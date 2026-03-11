# VLM MCP Server

[中文版本](./README_zh.md)

## Why This Project?

When using Claude Code with third-party models, they are typically text-only models without image processing capabilities. Adding an MCP server with image processing capability is essential for tasks that require visual understanding.

This project enables users to select their own Vision-Language Model (VLM) for image processing.

## Features

- **extract_text_from_image**: Extract text from images (OCR)
- **ui_to_artifact**: Convert UI screenshots to code, prompts, design specs, or descriptions
- **extract_text_from_screenshot**: Extract text from screenshots with code recognition support
- **diagnose_error_screenshot**: Analyze error screenshots and diagnose issues
- **understand_technical_diagram**: Analyze technical diagrams (architecture, flowcharts, UML, etc.)
- **analyze_data_visualization**: Analyze data visualization charts
- **ui_diff_check**: UI comparison to detect visual differences
- **analyze_image**: General-purpose image analysis

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `VLM_API_KEY` | Yes | API key |
| `VLM_BASE_URL` | No | Custom API endpoint (default: https://api.openai.com/v1) |
| `VLM_MODEL` | No | Model to use (default: gpt-4o) |
| `VLM_MAX_IMAGE_SIZE` | No | Maximum image size (default: 3MB). Images exceeding this size will be automatically compressed before processing. Supported formats: `3MB`, `3M`, `3145728` (bytes), `1024KB`, etc. |

## Quick Start

### Using uvx (Recommended)

```bash
# Copy config template and fill in your API Key
cp .env.example .env
# Edit .env file and fill in VLM_API_KEY

# Run directly (will automatically load .env file)
uvx vlm-mcp
```

### Using pip

```bash
# Install
pip install vlm-mcp

# Or install in development mode
pip install -e .
```

### Configure Environment Variables

```bash
# OpenAI
export VLM_API_KEY=sk-xxx
export VLM_MODEL=gpt-4o

# Custom API (e.g., Ollama)
export VLM_API_KEY=your-api-key
export VLM_BASE_URL=http://localhost:11434/v1
export VLM_MODEL=qwen2.5-vl
```

### Run the Server

```bash
# Run directly
python -m vlm_mcp

# Or use installed command
vlm-mcp
```

## Supported Models

Any VLM model compatible with OpenAI Chat Completions API:

- gpt-4o
- gpt-4o-mini
- gpt-4-turbo
- qwen2.5-vl series
- Other OpenAI API compatible models

## Claude Code Configuration

### 1. Configure MCP Server

Add the following to your Claude Code configuration:

```json
{
  "mcpServers": {
    "vlm-mcp": {
      "command": "uvx",
      "args": ["vlm-mcp"],
      "env": {
        "VLM_API_KEY": "your-api-key",
        "VLM_BASE_URL": "https://api.openai.com/v1",
        "VLM_MODEL": "gpt-4o",
        "VLM_MAX_IMAGE_SIZE": "5MB"
      }
    }
  }
}
```

### 2. Configure CLAUDE.md

To ensure Claude Code uses MCP tools for reading images instead of the built-in Read tool, add the following to your project or global CLAUDE.md:

```markdown
## MCP Priority

1. Use mcp tools to read images instead of claude code's read tool.
```

## Usage Examples

In Claude Code:

```
Please use extract_text_from_image tool to analyze this image /path/to/image.jpg and extract the text.
```

```
Please use ui_to_artifact tool to convert this UI screenshot to React code.
```

---

Inspired by the approach used by Zhipu AI.
