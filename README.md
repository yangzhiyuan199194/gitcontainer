# Gitcontainer 🐳

**Turn any GitHub repository into a production-ready Docker container with AI-powered Dockerfile generation.**

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-00a393.svg)](https://fastapi.tiangolo.com/)

Gitcontainer is an AI-powered web application that automatically generates production-ready Dockerfiles by analyzing GitHub repositories. Simply paste a GitHub URL and get a tailored Dockerfile with intelligent base image selection, dependency management, and Docker best practices.

## ✨ Features

- **🤖 AI-Powered Analysis**: Uses OpenAI GPT-4 to analyze repository structure and generate intelligent Dockerfiles
- **⚡ Real-time Streaming**: Watch the AI generate your Dockerfile in real-time with WebSocket streaming
- **🎯 Smart Detection**: Automatically detects technology stacks (Python, Node.js, Java, Go, etc.)
- **🔧 Production-Ready**: Generates Dockerfiles following best practices with proper security, multi-stage builds, and optimization
- **📋 Additional Instructions**: Add custom requirements for specialized environments
- **📄 Docker Compose**: Automatically suggests docker-compose.yml for complex applications
- **🎨 Modern UI**: Clean, responsive interface with Monaco editor for syntax highlighting
- **📱 Mobile Friendly**: Works seamlessly on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Git
- OpenAI API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cyclotruc/gitcontainer.git
   cd gitcontainer
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Open your browser:**
   Navigate to `http://localhost:8000`

## 🐳 Docker Deployment

### Build and run with Docker:

```bash
# Build the image
docker build -t gitcontainer .

# Run the container
docker run -p 8000:8000 -e OPENAI_API_KEY=your_api_key gitcontainer
```

### Using docker-compose:

```yaml
version: '3.8'
services:
  gitcontainer:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=your_openai_api_key_here
    volumes:
      - ./repos:/app/repos  # Persist cloned repositories
```

## 🛠️ How It Works

1. **Repository Cloning**: Gitcontainer clones the GitHub repository locally using Git
2. **Code Analysis**: Uses [gitingest](https://github.com/cyclotruc/gitingest) to analyze the repository structure and extract relevant information
3. **AI Generation**: Sends the analysis to OpenAI GPT-4 with specialized prompts for Dockerfile generation
4. **Smart Optimization**: The AI considers:
   - Technology stack detection
   - Dependency management
   - Security best practices
   - Multi-stage builds when beneficial
   - Port configuration
   - Environment variables
   - Health checks

## 📁 Project Structure

```
cyclotruc-gitcontainer/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── static/               # Static assets (icons, CSS)
├── templates/
│   └── index.jinja       # Main HTML template
└── tools/                # Core functionality modules
    ├── __init__.py
    ├── create_container.py  # AI Dockerfile generation
    ├── git_operations.py    # GitHub repository cloning
    └── gitingest.py        # Repository analysis
```

## 🔧 API Reference

### WebSocket Streaming

Connect to `/ws/{session_id}` for real-time generation updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/session_123');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data.type, data.content);
};
```

### Health Check

```bash
curl http://localhost:8000/health
```

## 🎛️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |
| `PORT` | Server port (default: 8000) | No |
| `HOST` | Server host (default: 0.0.0.0) | No |

### Advanced Usage

You can use the tools programmatically:

```python
from tools import clone_repo_tool, gitingest_tool, create_container_tool
import asyncio

async def generate_dockerfile(github_url):
    # Clone repository
    clone_result = await clone_repo_tool(github_url)
    
    # Analyze with gitingest
    analysis = await gitingest_tool(clone_result['local_path'])
    
    # Generate Dockerfile
    dockerfile = await create_container_tool(
        gitingest_summary=analysis['summary'],
        gitingest_tree=analysis['tree'],
        gitingest_content=analysis['content']
    )
    
    return dockerfile

# Usage
result = asyncio.run(generate_dockerfile("https://github.com/user/repo"))
print(result['dockerfile'])
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test them
4. Commit with clear messages: `git commit -m "Add feature X"`
5. Push to your fork: `git push origin feature-name`
6. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## 🧪 Testing

Test with example repositories:

- **Simple Python app**: `https://github.com/cyclotruc/gitingest`
- **This project**: `https://github.com/cyclotruc/gitcontainer`
- **Complex Node.js app**: Any Express.js repository
- **Multi-service app**: Repositories with multiple services

## 🎨 Customization

### Adding Custom Instructions

Use the "Additional instructions" feature to customize generation:

- `"Use Alpine Linux for smaller image size"`
- `"Include Redis and PostgreSQL"`
- `"Optimize for production deployment"`
- `"Add development tools for debugging"`

### Extending Technology Support

Add new technology detection in `tools/create_container.py`:

```python
# Add your technology patterns to the AI prompt
technology_patterns = {
    "rust": ["Cargo.toml", "src/main.rs"],
    "ruby": ["Gemfile", "app.rb", "config.ru"],
    # Add more...
}
```

## 🐛 Troubleshooting

### Common Issues

**"OPENAI_API_KEY not found"**
- Ensure your `.env` file contains the API key
- Check that the environment variable is properly set

**"Failed to clone repository"**
- Verify the GitHub URL is correct and public
- Check your internet connection
- Ensure Git is installed on your system

**"Generation timeout"**
- Large repositories may take longer to process
- Check your OpenAI API quota and limits

**Monaco Editor not loading**
- Ensure you have internet connection for CDN resources
- Check browser console for JavaScript errors

### Performance Tips

- **Large repositories**: Consider adding `.gitignore` patterns to exclude large files
- **Private repositories**: Currently only public GitHub repositories are supported
- **API limits**: Monitor your OpenAI API usage to avoid rate limits

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[OpenAI](https://openai.com/)** for providing the GPT-4 API
- **[gitingest](https://github.com/cyclotruc/gitingest)** for repository analysis capabilities
- **[FastAPI](https://fastapi.tiangolo.com/)** for the excellent web framework
- **[Monaco Editor](https://microsoft.github.io/monaco-editor/)** for code syntax highlighting

## 🔗 Links

- **GitHub Repository**: [https://github.com/cyclotruc/gitcontainer](https://github.com/cyclotruc/gitcontainer)
- **Demo**: Try it live with example repositories
- **Issues**: [Report bugs or request features](https://github.com/cyclotruc/gitcontainer/issues)

---

**Made with ❤️ by [Romain Courtois](https://github.com/cyclotruc)**

*Turn any repository into a container in seconds!*