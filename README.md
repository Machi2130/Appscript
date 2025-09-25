# 📊 Enhanced Market Analysis API v6.0

A professional FastAPI-based web service for comprehensive market sector analysis in India, featuring AI-powered structured markdown reports, web scraping, and advanced security measures.

## 🚀 Features

- **AI-Powered Analysis**: Leverages Groq's deepseek-r1-distill-llama-70b model for intelligent market insights
- **Web Scraping**: Multi-method scraping using BeautifulSoup, Trafilatura, and Crawl4AI
- **Structured Markdown Reports**: Professional investment reports with headers, tables, and bullet points
- **Search Integration**: DuckDuckGo search via SerpAPI for comprehensive data collection
- **Security First**: Input validation, rate limiting, content sanitization, and secure error handling
- **Caching System**: Efficient caching to reduce API calls and improve performance
- **CORS Support**: Configurable CORS for web frontend integration
- **Health Monitoring**: Built-in health check endpoints

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python async web framework)
- **AI**: Groq API for LLM processing
- **Search**: SerpAPI (Google Search API)
- **Scraping**: BeautifulSoup4, Trafilatura, Crawl4AI
- **Data Validation**: Pydantic
- **Async HTTP**: aiohttp, httpx
- **Deployment**: Uvicorn ASGI server

## 📋 Prerequisites

- Python 3.10+
- GROQ API Key
- SerpAPI Key

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Machi2130/Appscript.git
   cd Appscript
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   export SERPAPI_KEY="your_serpapi_key_here"
   ```

## 🚀 Usage

### Running the Server

```bash
python ap.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### GET `/`
Serves the main HTML frontend interface.

#### GET `/analyze/{sector}`
Performs comprehensive market analysis for a specific sector.

**Parameters:**
- `sector` (path): Sector name (e.g., "technology", "banking", "infrastructure")

**Response:**
```json
{
  "sector": "technology",
  "report": "# 📊 Technology Sector Investment Analysis\n\n## 🎯 Executive Summary...",
  "metadata": {
    "urls_found": 6,
    "sites_scraped": 5,
    "processing_time_seconds": 45.2,
    "timestamp": "2024-01-15T10:30:00",
    "report_format": "structured_markdown",
    "scraped_sources": [...]
  }
}
```

#### GET `/sectors`
Returns list of available sectors for analysis.

**Response:**
```json
{
  "sectors": [
    {"name": "Infrastructure", "id": "infrastructure"},
    {"name": "Energy", "id": "energy"},
    {"name": "Technology", "id": "technology"},
    ...
  ]
}
```

#### GET `/health`
API health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "6.0.0",
  "groq_available": true,
  "crawl4ai_available": true,
  "timestamp": "2024-01-15T10:30:00",
  "features": {
    "structured_markdown_reports": true,
    "rate_limiting": true,
    "input_validation": true,
    "secure_error_handling": true,
    "content_sanitization": true,
    "professional_formatting": true,
    "markdown_preservation": true
  }
}
```

## 🔍 Analysis Process

The API follows a 3-step analysis process:

1. **Search & Discovery**: Searches DuckDuckGo for relevant news and company information
2. **Web Scraping**: Scrapes top websites using multiple methods for comprehensive data
3. **AI Analysis**: Processes collected data through Groq LLM to generate structured investment reports

## 🛡️ Security Features

- **Input Validation**: Pydantic models with custom validators
- **Rate Limiting**: 10 requests per minute per IP
- **Content Sanitization**: Prevents XSS and injection attacks
- **Error Handling**: Secure error responses without information disclosure
- **API Key Protection**: Environment variable-based configuration

## 📊 Report Format

Reports are generated in structured markdown with the following sections:

- Executive Summary
- Market Overview
- Financial Performance
- Major Companies & Players
- Investment Opportunities
- Risk Assessment
- Investment Recommendation
- Report Metadata

## 🌐 Frontend

The API includes a simple HTML frontend (`index.html`) for easy interaction. Access it at the root endpoint `/`.

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | API key for Groq AI service | Yes |
| `SERPAPI_KEY` | API key for SerpAPI | Yes |

### Optional Configuration

- **Rate Limit**: Modify `RATE_LIMIT_PER_MINUTE` in `ap.py`
- **Content Limit**: Adjust `CONTENT_LIMIT` for scraped content size
- **URLs to Pick**: Change `TOP_URLS_TO_PICK` for search results

## 🧪 Testing

Run tests with pytest:
```bash
pytest
```

## 📦 Dependencies

Key dependencies are listed in `requirements.txt`:

- `fastapi==0.104.1` - Web framework
- `groq==0.4.1` - AI API client
- `google-search-results==2.4.2` - Search API
- `beautifulsoup4==4.12.2` - HTML parsing
- `crawl4ai==0.7.4` - Advanced scraping
- `pydantic==2.5.0` - Data validation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Groq for providing powerful AI models
- SerpAPI for search functionality
- FastAPI community for excellent documentation
- Open source contributors to web scraping libraries

## 📞 Support

For support, please open an issue on GitHub or contact the maintainers.

---

**Disclaimer**: This tool provides general market analysis for informational purposes only. Not financial advice. Always consult with qualified financial advisors before making investment decisions.
