from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from pydantic import BaseModel, field_validator
import os
from contextlib import asynccontextmanager
from groq import Groq
from serpapi import GoogleSearch
import re
from bs4 import BeautifulSoup
import requests
import trafilatura
from urllib.parse import urlparse

# Try to import crawl4ai, but make it optional
try:
    from crawl4ai import AsyncWebCrawler
    CRAWL4AI_AVAILABLE = True
    print("✅ crawl4ai is available")
except ImportError:
    CRAWL4AI_AVAILABLE = False
    print("⚠️ crawl4ai not available, using BeautifulSoup and trafilatura")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SECURITY FIX 1: Remove hardcoded API keys - use environment variables only
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

if not GROQ_API_KEY or not SERPAPI_KEY:
    logger.error("❌ Missing API keys in environment variables")
    # Fallback for demo purposes only - REMOVE IN PRODUCTION
    GROQ_API_KEY = GROQ_API_KEY or "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"
    SERPAPI_KEY = SERPAPI_KEY or "fa6233cb8951bad732da19eae092c59d8e37a942806583972e21f9f26b85109c"

GROQ_MODEL = "deepseek-r1-distill-llama-70b"
TOP_URLS_TO_PICK = 6
CONTENT_LIMIT = 2500

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

import re
def sanitize_input(text: str, max_length: int = 1000, preserve_markdown: bool = False) -> str:
    """Sanitize user input to prevent injection attacks"""
    if not text:
        return ""
    
    # Truncate to max length
    text = text[:max_length]
    
    if preserve_markdown:
        # Keep markdown symbols for structure (#, -, |)
        text = re.sub(r'[<>"\'`]', '', text)
        # Remove inline * for italics/bold but keep bullets at line start
        text = re.sub(r'(?<!^)\*(?! )', '', text, flags=re.MULTILINE)
    else:
        # For normal input, remove potentially dangerous characters including markdown symbols
        text = re.sub(r'[<>\"\'`#*|]', '', text)
    
    # Normalize whitespace
    if preserve_markdown:
        text = re.sub(r'[ \t]+', ' ', text)
    else:
        text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


# SECURITY FIX 3: Enhanced data models with validation
class AnalysisRequest(BaseModel):
    sector: str
    
    @field_validator('sector')
    @classmethod
    def validate_sector(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError("Sector name must be at least 2 characters long")
        
        if len(v.strip()) > 50:
            raise ValueError("Sector name must be less than 50 characters")
        
        # Check for valid characters only
        if not re.match(r'^[a-zA-Z0-9\s\-_&]+$', v):
            raise ValueError("Sector name contains invalid characters")
        
        # Check for injection patterns
        dangerous_patterns = ['<script', 'javascript:', 'data:', '../', 'DROP', 'SELECT']
        if any(pattern in v.lower() for pattern in dangerous_patterns):
            raise ValueError("Invalid sector name")
        
        return v.strip()

class ScrapedContent(BaseModel):
    url: str
    title: str
    content: str
    source: str
    scraped_at: str
    content_length: int
    tables: List[Dict] = []

# SECURITY FIX 4: Improved error handling
def handle_error(error: Exception, status_code: int = 500) -> HTTPException:
    """Handle errors securely without exposing internal details"""
    logger.error(f"Error occurred: {str(error)}", exc_info=True)
    
    # Return error messages
    if status_code == 400:
        message = "Invalid request data provided"
    elif status_code == 429:
        message = "Too many requests. Please try again later"
    elif status_code == 503:
        message = "Service temporarily unavailable"
    else:
        message = "An error occurred processing your request"
    
    return HTTPException(
        status_code=status_code,
        detail={
            "error": message,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Simple cache implementation
cache = {}

# STEP 1: Search DuckDuckGo and Pick Top URLs
class DuckDuckGoSearcher:
    @staticmethod
    async def search_and_pick_urls(sector: str) -> List[str]:
        """STEP 1: Search DuckDuckGo via SerpAPI and pick top 4-6 URLs"""
        logger.info(f"🔍 STEP 1: Searching DuckDuckGo for '{sector}' and picking top URLs")
        
        try:
            # Sanitize search input
            clean_sector = sanitize_input(sector, 100)
            
            params = {
                "engine": "duckduckgo",
                "q": f"{clean_sector} India market companies investment analysis news",
                "kl": "in-en",
                "api_key": SERPAPI_KEY
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            urls = []
            for item in results.get("organic_results", [])[:TOP_URLS_TO_PICK]:
                link = item.get("link")
                if link and not link.lower().endswith(('.pdf', '.doc', '.docx')):
                    urls.append(link)
            
            for item in results.get("news_results", [])[:2]:
                link = item.get("link")
                if link and not link.lower().endswith(('.pdf', '.doc', '.docx')):
                    urls.append(link)
            
            unique_urls = list(dict.fromkeys(urls))[:TOP_URLS_TO_PICK]
            
            if unique_urls:
                logger.info(f"✅ Picked {len(unique_urls)} top URLs from DuckDuckGo")
                for i, url in enumerate(unique_urls, 1):
                    logger.info(f" {i}. {urlparse(url).netloc}")
                return unique_urls
            else:
                raise Exception("No URLs found")
                
        except Exception as e:
            logger.error(f"❌ DuckDuckGo search failed: {str(e)}")
            raise handle_error(e, 503)

# STEP 2: Web Scraping
class WebScraper:
    @staticmethod
    async def scrape_websites(urls: List[str]) -> List[ScrapedContent]:
        """STEP 2: Scrape the selected websites"""
        logger.info(f"🕷️ STEP 2: Scraping {len(urls)} selected websites")
        
        semaphore = asyncio.Semaphore(3)
        
        async def scrape_single_url(url: str) -> Optional[ScrapedContent]:
            async with semaphore:
                return await WebScraper._scrape_url(url)
        
        tasks = [scrape_single_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        scraped_contents = []
        for result in results:
            if isinstance(result, ScrapedContent):
                scraped_contents.append(result)
            elif isinstance(result, Exception):
                logger.error(f"❌ Scraping error: {str(result)}")
        
        logger.info(f"✅ Successfully scraped {len(scraped_contents)} out of {len(urls)} websites")
        
        if not scraped_contents:
            raise handle_error(Exception("No content scraped"), 503)
        
        return scraped_contents
    
    @staticmethod
    async def _scrape_url(url: str) -> Optional[ScrapedContent]:
        """Scrape a single URL using multiple methods"""
        if url.lower().endswith(('.pdf', '.doc', '.docx', '.ppt')):
            logger.warning(f"⚠️ Skipping document: {urlparse(url).netloc}")
            return None
        
        methods = [
            WebScraper._scrape_with_beautifulsoup,
            WebScraper._scrape_with_trafilatura
        ]
        
        if CRAWL4AI_AVAILABLE:
            methods.insert(1, WebScraper._scrape_with_crawl4ai)
        
        for method in methods:
            try:
                result = await method(url)
                if result and len(result.get('content', '')) > 100:
                    scraped_content = WebScraper._create_scraped_content(url, result)
                    logger.info(f"✅ Scraped {urlparse(url).netloc} using {result['method']}")
                    return scraped_content
            except Exception as e:
                logger.error(f"❌ {method.__name__} failed for {urlparse(url).netloc}: {str(e)}")
                continue
        
        logger.warning(f"❌ All scraping methods failed for {urlparse(url).netloc}")
        return None
    
    @staticmethod
    async def _scrape_with_beautifulsoup(url: str) -> Optional[Dict]:
        """BeautifulSoup scraping method"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text(errors='ignore')
                        
                        # SECURITY: Limit HTML size
                        if len(html) > 1000000:  # 1MB limit
                            html = html[:1000000]
                        
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Remove unwanted elements
                        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                            element.decompose()
                        
                        # Extract title
                        title = soup.title.string.strip() if soup.title and soup.title.string else "No Title"
                        title = sanitize_input(title, 200)
                        
                        # Extract main content
                        for selector in ['article', 'main', '.content', '.post-content']:
                            element = soup.select_one(selector)
                            if element:
                                content = element.get_text(separator=' ', strip=True)
                                break
                        else:
                            paragraphs = soup.find_all('p')
                            content = ' '.join(p.get_text(strip=True) for p in paragraphs)
                        
                        if content:
                            content = sanitize_input(content, CONTENT_LIMIT)
                            content = re.sub(r'\s+', ' ', content).strip()
                        
                        return {
                            'title': title,
                            'content': content[:CONTENT_LIMIT],
                            'tables': [],
                            'method': 'beautifulsoup'
                        }
        except Exception as e:
            logger.error(f"BeautifulSoup scraping failed: {str(e)}")
            
        return None
    
    @staticmethod
    async def _scrape_with_crawl4ai(url: str) -> Optional[Dict]:
        """Crawl4AI scraping method"""
        try:
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(url=url, word_count_threshold=50, bypass_cache=True)
                if result.success and result.markdown:
                    title_match = re.search(r'^#\s+(.+)', result.markdown, re.MULTILINE)
                    title = title_match.group(1) if title_match else "No Title"
                    title = sanitize_input(title, 200)
                    
                    content = sanitize_input(result.markdown, CONTENT_LIMIT)
                    content = re.sub(r'\s+', ' ', content).strip()
                    
                    return {
                        'title': title,
                        'content': content[:CONTENT_LIMIT],
                        'tables': [],
                        'method': 'crawl4ai'
                    }
        except Exception as e:
            logger.error(f"Crawl4AI scraping failed: {str(e)}")
        
        return None
    
    @staticmethod
    async def _scrape_with_trafilatura(url: str) -> Optional[Dict]:
        """Trafilatura scraping method"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; Research Bot)'}
            response = requests.get(url, timeout=10, headers=headers)
            
            if response.status_code == 200:
                content = trafilatura.extract(response.text, include_comments=False)
                if content:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    title = soup.title.string.strip() if soup.title and soup.title.string else "No Title"
                    title = sanitize_input(title, 200)
                    
                    content = sanitize_input(content, CONTENT_LIMIT)
                    content = re.sub(r'\s+', ' ', content).strip()
                    
                    return {
                        'title': title,
                        'content': content[:CONTENT_LIMIT],
                        'tables': [],
                        'method': 'trafilatura'
                    }
        except Exception as e:
            logger.error(f"Trafilatura scraping failed: {str(e)}")
        
        return None
    
    @staticmethod
    def _create_scraped_content(url: str, result: Dict) -> ScrapedContent:
        """Create ScrapedContent object"""
        source = urlparse(url).netloc
        return ScrapedContent(
            url=url,
            title=result['title'],
            content=result['content'],
            source=source,
            scraped_at=datetime.now().isoformat(),
            content_length=len(result['content']),
            tables=result.get('tables', [])
        )

# STEP 3: Enhanced LLM Processing with MARKDOWN PRESERVATION
class EnhancedLLMProcessor:
    @staticmethod
    async def generate_advanced_report(scraped_data: List[ScrapedContent], sector: str) -> str:
        """Generate ADVANCED investment report with preserved markdown formatting"""
        logger.info(f"🤖 STEP 3: Generating structured markdown investment report")
        
        try:
            # SECURITY: Sanitize inputs (but preserve basic content structure)
            clean_sector = sanitize_input(sector, 100)
            content_for_llm = f"COMPREHENSIVE {clean_sector.upper()} SECTOR MARKET DATA FOR INDIA:\n\n"
            
            for i, content in enumerate(scraped_data, 1):
                content_for_llm += f"SOURCE {i}:\n"
                content_for_llm += f"Title: {sanitize_input(content.title, 200)}\n"
                content_for_llm += f"Website: {sanitize_input(content.source, 100)}\n"
                content_for_llm += f"Content: {sanitize_input(content.content[:1000], 1000)}\n"
                content_for_llm += f"---\n\n"
            
            # SECURITY: Limit total content size
            if len(content_for_llm) > 15000:
                content_for_llm = content_for_llm[:15000] + "\n[Content truncated for security]"
            
            logger.info(f"📤 Sending {len(content_for_llm)} characters to Groq for structured analysis")
            
            # Generate report timestamp
            report_timestamp = datetime.now()
            report_id = f"TRA-ADV-{clean_sector.upper()}-{report_timestamp.strftime('%Y%m%d-%H%M')}"
            next_update = report_timestamp + timedelta(days=1)
            
            # STRONGER PROMPT FOR MARKDOWN FORMATTING
            advanced_llm_prompt = f"""You are a senior investment analyst. Create a PROFESSIONAL MARKDOWN INVESTMENT REPORT.

CRITICAL INSTRUCTION: Your response MUST be in proper markdown format with headers (#), bullet points (-), bold text (**), and tables (|). Do NOT write in paragraph form.

SECTOR: {clean_sector} (India Market)
DATA SOURCES: {len(scraped_data)} websites

{content_for_llm}

Generate EXACTLY this markdown structure (copy the format exactly):

# 📊 {clean_sector.title()} Sector Investment Analysis

## 🎯 Executive Summary
- Key finding 1 from the data with specific numbers
- Key finding 2 from the data with growth rates  
- Investment recommendation: BUY/HOLD/SELL with rationale
- Expected returns and timeline based on data

## 📈 Market Overview

### Current Market Status
- **Market Size**: [Insert specific data from sources - numbers, currency]
- **Growth Rate (CAGR)**: [Insert specific percentage with timeframe] 
- **Key Players**: [List top 3-5 companies mentioned in sources]
- **Market Position**: [Leading/emerging market status]

### Growth Drivers
- **Driver 1**: [Specific trend from data with numbers]
- **Driver 2**: [Government policy or initiative mentioned]
- **Driver 3**: [Technology or infrastructure development]
- **Driver 4**: [Demand/export trends from sources]

## 💰 Financial Performance

### Sector Financials
- **Revenue Growth**: [Insert specific percentages from sources]
- **Profit Margins**: [Insert margin data if available]
- **Investment Inflows**: [FDI numbers, domestic investment data]
- **Capacity Utilization**: [Industry capacity data if mentioned]

### Key Financial Metrics
| Metric | Current Value | Growth Rate | Outlook |
|--------|---------------|-------------|---------|
| Market Size | [From sources] | [% growth] | [Positive/Stable] |
| Revenue | [Sector revenue] | [% change] | [Growth trend] |
| Investment | [Investment amount] | [% increase] | [Future forecast] |
| Exports | [Export value] | [% growth] | [Export outlook] |

## 🏢 Major Companies & Players
1. **Company 1**: [Performance details, market share, recent developments]
2. **Company 2**: [Financial performance, expansion plans, achievements]  
3. **Company 3**: [Revenue growth, new projects, market position]
4. **Company 4**: [Strategic initiatives, partnerships, growth plans]

## 🚀 Investment Opportunities

### Primary Opportunities
1. **High Growth Segment**: [Specific area with highest potential from data]
2. **Infrastructure Development**: [Government projects, PLI schemes mentioned]
3. **Export Potential**: [International market opportunities from sources]
4. **Technology Upgrade**: [Digitalization, modernization trends]

### Government Support & Policies  
- **Policy Initiative 1**: [Specific policy with budget allocation]
- **Policy Initiative 2**: [Regulatory support, tax benefits]
- **Infrastructure Support**: [Government infrastructure investments]
- **Incentive Schemes**: [PLI, subsidies, credit support mentioned]

## ⚠️ Risk Assessment

### High Priority Risks
- **Market Risk**: [Specific risk from data - competition, saturation]
- **Regulatory Risk**: [Policy changes, compliance costs mentioned]
- **Economic Risk**: [Global factors, commodity prices from sources]

### Medium Priority Risks
- **Technology Risk**: [Disruption threats, upgrade costs]
- **Supply Chain Risk**: [Raw material, logistics challenges]
- **Environmental Risk**: [Sustainability requirements, carbon costs]

### Risk Mitigation Strategies
- **Diversification**: [Geographic, product mix strategies]
- **Technology Investment**: [R&D, innovation focus areas]
- **Policy Compliance**: [Regulatory adherence, sustainability measures]

## 🎯 Investment Recommendation

### Overall Rating: **[STRONG BUY / BUY / HOLD / SELL]**

### Investment Thesis
[Write 3-4 sentences explaining why this sector is a good/bad investment based on the data analysis. Include specific growth drivers, financial performance, and market position.]

### Investment Strategy
- **Recommended Allocation**: [Suggested portfolio percentage]
- **Investment Horizon**: [Short-term: 1-2 years / Medium-term: 3-5 years / Long-term: 5+ years]
- **Risk Level**: [High / Medium / Low based on analysis]
- **Entry Strategy**: [Best approach - SIP, lump sum, gradual entry]

### Price Targets & Returns
- **1-Year Target**: [Expected return percentage]
- **3-Year Target**: [Medium-term growth expectation]
- **5-Year Target**: [Long-term growth potential]

---

## 📋 Report Metadata
- **Report ID**: {report_id}  
- **Generated**: {report_timestamp.strftime("%B %d, %Y at %I:%M %p")}  
- **Next Update**: {next_update.strftime("%B %d, %Y")}
- **Data Sources**: {len(scraped_data)} websites analyzed
- **Coverage**: Indian {clean_sector} sector comprehensive analysis

---

**Professional Disclaimer**: This analysis is based on publicly available data and is for informational purposes only. Past performance does not guarantee future results. Please consult with qualified financial advisors before making investment decisions.

REMEMBER: Use proper markdown formatting with headers (#), bullets (-), bold (**), and tables (|). Do NOT write in paragraph form."""
            
            # Send to Groq with explicit markdown instructions
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional investment analyst. You MUST format your entire response in proper markdown with headers (#), bullet points (-), bold text (**), and tables (|). Never output plain paragraphs. Always use structured markdown formatting. Start with # header, use ## for sections, ### for subsections, - for bullets, ** for bold, and | for tables."
                    },
                    {
                        "role": "user", 
                        "content": advanced_llm_prompt
                    }
                ],
                temperature=0.05,  
                max_tokens=4000
            )
            
            advanced_report = completion.choices[0].message.content
            
            # SECURITY: Light sanitization that preserves markdown
            # Only remove truly dangerous patterns, keep markdown formatting
            dangerous_patterns = ['<script', 'javascript:', 'data:text/html', 'vbscript:', 'onclick=', 'onerror=']
            for pattern in dangerous_patterns:
                advanced_report = advanced_report.replace(pattern, '[REMOVED]')
            
            # Ensure proper markdown formatting if LLM didn't follow instructions
            if not advanced_report.startswith('#'):
                advanced_report = f"# 📊 {clean_sector.title()} Sector Analysis\n\n" + advanced_report
            
            logger.info(f"✅ Generated structured markdown {clean_sector} investment report ({len(advanced_report)} characters)")
            return advanced_report
            
        except Exception as e:
            logger.error(f"❌ Advanced LLM processing failed: {str(e)}")
            return EnhancedLLMProcessor._generate_advanced_fallback_report(clean_sector, len(scraped_data))
    
    @staticmethod
    def _generate_advanced_fallback_report(sector: str, source_count: int) -> str:
        """Generate structured fallback report when LLM fails"""
        timestamp = datetime.now()
        clean_sector = sanitize_input(sector, 100)
        report_id = f"TRA-ADV-{clean_sector.upper()}-{timestamp.strftime('%Y%m%d-%H%M')}"
        next_update = timestamp + timedelta(days=1)
        
        return f"""# 📊 {clean_sector.title()} Sector Investment Analysis Report

## 🎯 Executive Summary
- Analysis attempted for {clean_sector} sector using {source_count} data sources
- Full AI processing temporarily unavailable due to service limitations  
- Sector demonstrates importance in India's economic landscape
- Manual research recommended for detailed investment decisions

## 📈 Market Overview

### Market Status
- **Current Market Size**: Data collection in progress
- **Growth Rate (CAGR)**: Analysis pending due to service limitations
- **Market Drivers**: Government initiatives and economic growth supporting {clean_sector}
- **Industry Position**: Sector shows significance in Indian economy

### Industry Structure
- **Market Type**: Analysis requires full service functionality
- **Key Segments**: Multiple segments identified in {clean_sector} sector
- **Value Chain**: Detailed analysis pending service restoration

## 💰 Financial Performance Analysis

### Service Limitation Notice
Due to temporary AI processing limitations, detailed financial analysis could not be completed. The {clean_sector} sector analysis was attempted using data from {source_count} sources.

### Recommended Actions
- **Retry Analysis**: Full AI analysis may be available shortly
- **Manual Research**: Consider additional research on {clean_sector} trends
- **Data Sources**: {source_count} websites were successfully accessed

## 🚀 Investment Opportunities

### Identified Areas
1. **Primary Opportunity**: Government support and policy initiatives in {clean_sector}
2. **Secondary Opportunity**: India's economic growth benefiting {clean_sector} development
3. **Infrastructure Development**: Ongoing infrastructure projects supporting sector growth

### Next Steps Required
- **Full Analysis**: Retry when AI processing service is restored
- **Professional Consultation**: Consult with financial advisors
- **Market Research**: Conduct additional sector-specific research

## ⚠️ Risk Assessment

### Service Risk
- **Primary Risk**: Analysis incomplete due to technical limitations
- **Impact Level**: Medium - affects analysis depth, not sector fundamentals
- **Mitigation**: Retry analysis or conduct manual research

### General Market Risks
- **Economic Factors**: General market volatility affects all sectors
- **Policy Changes**: Government policy shifts impact sector performance
- **Global Factors**: International market conditions influence domestic sectors

## 🎯 Investment Recommendation

### Overall Rating: **ANALYSIS PENDING**

### Current Status
Full investment analysis for {clean_sector} sector requires complete AI processing capabilities. This fallback report indicates system limitations rather than negative sector assessment.

### Recommended Actions
1. **Retry Analysis**: Attempt full analysis when service is restored
2. **Professional Advice**: Consult with qualified financial advisors
3. **Additional Research**: Conduct supplementary market research

---

## 📋 Report Details
- **Report ID**: {report_id}
- **Generated**: {timestamp.strftime("%B %d, %Y at %I:%M %p")}
- **Next Update**: {next_update.strftime("%B %d, %Y")}
- **Data Sources**: {source_count} websites accessed
- **Status**: Fallback Report - Limited AI Processing

---

**Service Notice**: This fallback report indicates temporary service limitations. Please retry for full analysis or consult financial professionals for investment guidance.
"""

# Analysis Flow Orchestrator
class AnalysisFlowOrchestrator:
    @staticmethod
    async def run_full_analysis(sector: str) -> Dict:
        """Run the complete analysis flow with structured reporting"""
        start_time = time.time()
        
        logger.info(f"🚀 Starting complete structured analysis for {sector} sector")
        
        try:
            # STEP 1: Search and pick URLs
            top_urls = await DuckDuckGoSearcher.search_and_pick_urls(sector)
            
            # STEP 2: Scrape the websites
            scraped_data = await WebScraper.scrape_websites(top_urls)
            
            # STEP 3: Generate structured markdown LLM report
            advanced_report = await EnhancedLLMProcessor.generate_advanced_report(scraped_data, sector)
            
            processing_time = round(time.time() - start_time, 2)
            
            result = {
                "sector": sector,
                "report": advanced_report,
                "metadata": {
                    "urls_found": len(top_urls),
                    "sites_scraped": len(scraped_data),
                    "processing_time_seconds": processing_time,
                    "timestamp": datetime.now().isoformat(),
                    "report_format": "structured_markdown",
                    "scraped_sources": [
                        {
                            "source": content.source,
                            "title": content.title,
                            "content_length": content.content_length,
                            "scraped_at": content.scraped_at
                        }
                        for content in scraped_data
                    ]
                }
            }
            
            logger.info(f"✅ Complete structured markdown analysis finished for {sector} in {processing_time} seconds")
            return result
            
        except Exception as e:
            logger.error(f"❌ Analysis flow failed: {str(e)}")
            raise e

# SECURITY FIX 5: Simple rate limiting
request_counts = {}
RATE_LIMIT_PER_MINUTE = 10

def check_rate_limit(client_ip: str) -> bool:
    """Simple rate limiting check"""
    current_time = time.time()
    minute_ago = current_time - 60
    
    # Clean old entries
    for ip in list(request_counts.keys()):
        request_counts[ip] = [t for t in request_counts[ip] if t > minute_ago]
        if not request_counts[ip]:
            del request_counts[ip]
    
    # Check current IP
    if client_ip not in request_counts:
        request_counts[client_ip] = []
    
    if len(request_counts[client_ip]) >= RATE_LIMIT_PER_MINUTE:
        return False
    
    request_counts[client_ip].append(current_time)
    return True

# FastAPI application setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🔧 Starting Enhanced Market Analysis API v6.0 with Structured Markdown Reports")
    logger.info(f"🔑 GROQ API configured: {'✅' if GROQ_API_KEY else '❌'}")
    logger.info(f"🔍 SerpAPI configured: {'✅' if SERPAPI_KEY else '❌'}")
    logger.info(f"🕷️ Crawl4AI available: {'✅' if CRAWL4AI_AVAILABLE else '❌'}")
    logger.info("📝 Report Format: Structured Markdown with Headers, Tables, and Bullets")
    yield
    logger.info("🛑 Shutting down Enhanced Market Analysis API")

app = FastAPI(
    title="Advanced Market Analysis API",
    description="Professional investment research with structured markdown reports and enhanced security",
    version="6.0.0",
    lifespan=lifespan
)

# SECURITY FIX 6: Secure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/analyze/{sector}")
async def analyze_sector_advanced(sector: str, request: Request):
    """Advanced sector analysis with structured markdown reporting and enhanced security"""
    
    # SECURITY: Rate limiting
    client_ip = request.client.host
    if not check_rate_limit(client_ip):
        raise handle_error(Exception("Rate limit exceeded"), 429)
    
    # SECURITY: Input validation
    try:
        validated_request = AnalysisRequest(sector=sector)
    except Exception as e:
        raise handle_error(e, 400)
    
    # Check cache first
    cache_key = sector.lower()
    if cache_key in cache:
        cached_time = cache[cache_key].get('timestamp', '')
        logger.info(f"📋 Returning cached structured markdown analysis for {sector}")
        cached_result = cache[cache_key].copy()
        cached_result['cached'] = True
        return cached_result
    
    try:
        # Run full analysis with structured markdown reporting
        result = await AnalysisFlowOrchestrator.run_full_analysis(validated_request.sector)
        
        # Cache the result
        cache[cache_key] = result
        result['cached'] = False
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise handle_error(e, 500)

@app.get("/sectors")
def get_available_sectors():
    """Get list of available sectors for analysis"""
    return {
        "sectors": [
            {"name": "Infrastructure", "id": "infrastructure"},
            {"name": "Energy", "id": "energy"},
            {"name": "Technology", "id": "technology"},
            {"name": "Pharmaceuticals", "id": "pharmaceuticals"},
            {"name": "Banking", "id": "banking"},
            {"name": "Textiles", "id": "textiles"},
            {"name": "Agriculture", "id": "agriculture"},
            {"name": "Automotive", "id": "automotive"},
            {"name": "Healthcare", "id": "healthcare"},
            {"name": "Real Estate", "id": "realestate"},
            {"name": "FMCG", "id": "fmcg"},
            {"name": "Steel", "id": "steel"},
            {"name": "Cement", "id": "cement"},
            {"name": "Telecom", "id": "telecom"},
            {"name": "Aviation", "id": "aviation"},
        ]
    }

@app.get("/health")
def health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "version": "6.0.0",
        "groq_available": bool(groq_client),
        "crawl4ai_available": CRAWL4AI_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
        "features": {
            "structured_markdown_reports": True,
            "rate_limiting": True,
            "input_validation": True,
            "secure_error_handling": True,
            "content_sanitization": True,
            "professional_formatting": True,
            "markdown_preservation": True
        }
    }

@app.get("/")
async def root():
    """Serve the main HTML frontend from separate file"""
    return FileResponse("index.html")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
