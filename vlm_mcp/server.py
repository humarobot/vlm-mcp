"""VLM MCP Server - Image understanding service based on OpenAI protocol"""

import os
import json
import base64
import asyncio
from pathlib import Path
from typing import Any, Optional
from io import BytesIO
from dotenv import load_dotenv

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from mcp.server import InitializationOptions

import httpx
from PIL import Image


class VLMClient:
    """VLM Client based on OpenAI protocol"""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self._client: Optional[httpx.AsyncClient] = None

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create shared httpx client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def close(self):
        """Close the httpx client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def chat(self, model: str, messages: list, **kwargs) -> str:
        """Send chat request"""
        url = f"{self.base_url}/chat/completions"

        # Process messages with images
        processed_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            # Filter unsupported roles
            if role not in ("system", "developer", "user", "assistant"):
                continue

            processed_msg = {"role": role, "content": msg.get("content", "")}

            if "content" in msg and isinstance(msg["content"], list):
                processed_content = []
                for item in msg["content"]:
                    if isinstance(item, dict):
                        if item.get("type") == "image_url":
                            # base64 image
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict):
                                url_data = image_url.get("url", "")
                                if url_data.startswith("data:image"):
                                    # data:image/jpeg;base64,xxx
                                    b64_data = url_data.split(",", 1)[1]
                                    processed_content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{b64_data}"
                                        }
                                    })
                                else:
                                    processed_content.append(item)
                            else:
                                processed_content.append(item)
                        elif item.get("type") == "text":
                            processed_content.append(item)
                    elif isinstance(item, str):
                        processed_content.append({"type": "text", "text": item})
                processed_msg["content"] = processed_content
            processed_messages.append(processed_msg)

        # Ensure at least one message
        if not processed_messages:
            processed_messages = [{"role": "user", "content": "Hello"}]

        payload = {
            "model": model,
            "messages": processed_messages,
            **kwargs
        }

        client = await self._get_client()
        response = await client.post(url, json=payload, headers=self._get_headers())
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        return str(result)


# Global client instance
_vlm_client: Optional[VLMClient] = None
_default_model: Optional[str] = None
_max_image_size: int = 3 * 1024 * 1024  # Default 3MB


def get_vlm_client() -> VLMClient:
    """Get VLM client instance"""
    global _vlm_client
    if _vlm_client is None:
        raise RuntimeError("VLM client not initialized. Please set VLM_API_KEY and optionally VLM_BASE_URL environment variables.")
    return _vlm_client


def get_default_model() -> Optional[str]:
    """Get default model"""
    return _default_model


def get_max_image_size() -> int:
    """Get max image size in bytes"""
    return _max_image_size


def init_vlm_client(api_key: str, base_url: Optional[str] = None, default_model: Optional[str] = None, max_image_size: Optional[int] = None):
    """Initialize VLM client"""
    global _vlm_client, _default_model, _max_image_size
    _vlm_client = VLMClient(api_key, base_url)
    _default_model = default_model
    if max_image_size is not None:
        _max_image_size = max_image_size


# Create MCP Server
app = Server("vlm-mcp")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="extract_text_from_image",
            description="[OCR Text Recognition] Extract text from images. Supports Chinese and English recognition. Input image path or base64 data, returns all text content in the image.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Local file path or URL of the image"
                    },
                    "image_data": {
                        "type": "string",
                        "description": "Base64 encoded image data (use when no image path, can be pure base64 or data URI format)"
                    }
                }
            }
        ),
        # ZAI MCP Server style tools
        Tool(
            name="ui_to_artifact",
            description="Convert UI screenshots to various artifacts: generate frontend code, create AI prompts for UI recreation, extract design specifications, generate natural language UI descriptions",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_source": {
                        "type": "string",
                        "description": "Local file path or remote URL of the image"
                    },
                    "output_type": {
                        "type": "string",
                        "enum": ["code", "prompt", "spec", "description"],
                        "description": "Output type: 'code' (generate frontend code), 'prompt' (generate AI prompt), 'spec' (generate design spec), 'description' (natural language description)"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Detailed instructions describing what to generate from this UI image. Should clearly state the desired output and any specific requirements."
                    }
                },
                "required": ["image_source", "output_type", "prompt"]
            }
        ),
        Tool(
            name="extract_text_from_screenshot",
            description="Extract and recognize text from screenshots using advanced OCR capabilities. Specialized for code, terminal output, documentation, and general text extraction.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_source": {
                        "type": "string",
                        "description": "Local file path or remote URL of the image"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Instructions for text extraction. Specify what type of text to extract and any formatting requirements."
                    },
                    "programming_language": {
                        "type": "string",
                        "description": "Optional: specify the programming language if the screenshot contains code (e.g., 'python', 'javascript', 'java') for better accuracy. Leave empty for auto-detection or non-code text."
                    }
                },
                "required": ["image_source", "prompt"]
            }
        ),
        Tool(
            name="diagnose_error_screenshot",
            description="Error diagnosis and troubleshooting: analyze error messages and stack traces, identify root causes, provide actionable solutions, suggest prevention strategies",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_source": {
                        "type": "string",
                        "description": "Local file path or remote URL of the image"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Description of what you need help with regarding this error. Include any relevant context about when the error occurred."
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional: additional context about when the error occurred (e.g., 'during npm install', 'when running the app', 'after deployment'). Helps with more accurate diagnosis."
                    }
                },
                "required": ["image_source", "prompt"]
            }
        ),
        Tool(
            name="understand_technical_diagram",
            description="Technical diagram analysis: analyze architecture diagrams, understand flowcharts and UML diagrams, explain ER diagrams and sequence diagrams, identify design patterns",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_source": {
                        "type": "string",
                        "description": "Local file path or remote URL of the image"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "What you want to understand or extract from this diagram."
                    },
                    "diagram_type": {
                        "type": "string",
                        "description": "Optional: specify the diagram type if known (e.g., 'architecture', 'flowchart', 'uml', 'er-diagram', 'sequence'). Leave empty for auto-detection."
                    }
                },
                "required": ["image_source", "prompt"]
            }
        ),
        Tool(
            name="analyze_data_visualization",
            description="Data visualization insights: extract insights from charts and graphs, identify trends and patterns, detect anomalies, provide business implications",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_source": {
                        "type": "string",
                        "description": "Local file path or remote URL of the image"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "What insights or information you want to extract from this visualization."
                    },
                    "analysis_focus": {
                        "type": "string",
                        "description": "Optional: specify what to focus on (e.g., 'trends', 'anomalies', 'comparisons', 'performance metrics'). Leave empty for comprehensive analysis."
                    }
                },
                "required": ["image_source", "prompt"]
            }
        ),
        Tool(
            name="ui_diff_check",
            description="UI comparison for visual regression: compare expected/reference UI with actual implementation, identify visual discrepancies, provide detailed difference reports, prioritize issues by severity",
            inputSchema={
                "type": "object",
                "properties": {
                    "expected_image_source": {
                        "type": "string",
                        "description": "Expected/reference UI image path or remote URL"
                    },
                    "actual_image_source": {
                        "type": "string",
                        "description": "Actual implementation image path or remote URL"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Instructions for the comparison. Specify what aspects to focus on or what level of detail is needed."
                    }
                },
                "required": ["expected_image_source", "actual_image_source", "prompt"]
            }
        ),
        Tool(
            name="analyze_image",
            description="General-purpose image analysis (fallback): use when specialized tools don't fit your needs. Flexible image understanding for any visual content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_source": {
                        "type": "string",
                        "description": "Local file path or remote URL of the image"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of what you want to analyze, extract, or understand from the image. Be specific about your requirements."
                    }
                },
                "required": ["image_source", "prompt"]
            }
        ),
    ]


def get_image_b64(image_source: str, max_size: int = 2048, quality: int = 85) -> str:
    """Get image base64 encoding, supports path, URL, or direct base64 data.

    Args:
        image_source: Image path, URL, or base64 data
        max_size: Maximum dimension (width or height) in pixels. Default 2048.
        quality: JPEG quality (1-100). Default 85.

    Returns:
        Base64 encoded image data (without data URI prefix)

    Raises:
        ValueError: If image_source is empty or invalid
    """
    if not image_source or not image_source.strip():
        raise ValueError("image_source cannot be empty")
    # Check if it's data URI format
    if image_source.startswith("data:"):
        b64_data = image_source.split(",", 1)[1]
        image_bytes = base64.b64decode(b64_data)
        image = Image.open(BytesIO(image_bytes))
        return _compress_and_encode(image)

    # Check if it's pure base64
    try:
        image_bytes = base64.b64decode(image_source)
        image = Image.open(BytesIO(image_bytes))
        return _compress_and_encode(image)
    except Exception as e:
        # Not valid base64, continue to check if it's a URL or file path
        pass

    # Check if it's a URL
    if image_source.startswith("http://") or image_source.startswith("https://"):
        response = httpx.get(image_source, timeout=30.0)
        response.raise_for_status()
        image_data = response.content
        image = Image.open(BytesIO(image_data))
        return _compress_and_encode(image)
    else:
        # Local file
        image = Image.open(image_source)
        return _compress_and_encode(image)


def _compress_and_encode(image: Image.Image) -> str:
    """Compress image until it meets size requirements.

    Args:
        image: PIL Image object

    Returns:
        Base64 encoded compressed image
    """
    max_size_bytes = get_max_image_size()

    # Convert to RGB mode if needed
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    # Start with aggressive compression
    quality = 85
    max_dimension = 2048

    for _ in range(10):  # Max 10 iterations to prevent infinite loop
        # Resize if needed
        width, height = image.size
        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Encode and check size
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        jpeg_bytes = buffer.getvalue()
        encoded = base64.b64encode(jpeg_bytes).decode("utf-8")

        # Check actual JPEG bytes size (not base64 string length)
        # Base64 adds ~37% overhead, so we compare raw bytes to max_size_bytes
        if len(jpeg_bytes) <= max_size_bytes:
            return encoded

        # Reduce quality and dimension for next iteration
        quality = max(30, quality - 15)
        max_dimension = max(512, int(max_dimension * 0.7))

    # Last resort: return what we have even if still too large
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=30, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def handle_extract_text(image_path: Optional[str] = None, image_data: Optional[str] = None) -> str:
    """Handle text extraction request"""
    client = get_vlm_client()

    # Encode image
    image_source = image_path or image_data
    if not image_source:
        raise ValueError("Either image_path or image_data must be provided")
    b64_image = get_image_b64(image_source)

    question = "Extract all text content from this image, including Chinese and English. If there is no text in the image, please state so."

    # Build messages
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_image}"
                }
            }
        ]
    }]
    max_tokens = 4096

    # Use default model
    default_model = get_default_model() or "gpt-4o"

    result = await client.chat(default_model, messages, max_tokens=max_tokens)
    return result


# ============ ZAI MCP Server style handler functions ============

async def handle_ui_to_artifact(image_source: str, output_type: str, prompt: str) -> str:
    """Handle UI to artifact conversion request"""
    client = get_vlm_client()
    b64_image = get_image_b64(image_source)

    # Build different prompts based on output_type
    output_prompts = {
        "code": "Generate the corresponding frontend code (React/Vue/HTML/CSS/etc.) based on this UI screenshot.",
        "prompt": "Generate an AI prompt for this UI design to recreate the UI.",
        "spec": "Extract the design specifications of this UI, including layout, colors, fonts, spacing and other detailed information.",
        "description": "Describe this UI interface in natural language."
    }

    full_prompt = f"{output_prompts.get(output_type, '')} {prompt}"

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": full_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_image}"
                }
            }
        ]
    }]

    default_model = get_default_model() or "gpt-4o"
    result = await client.chat(default_model, messages, max_tokens=4096)
    return result


async def handle_extract_text_from_screenshot(image_source: str, prompt: str, programming_language: Optional[str] = None) -> str:
    """Handle screenshot text extraction request"""
    client = get_vlm_client()
    b64_image = get_image_b64(image_source)

    lang_hint = f"(This is {programming_language} code)" if programming_language else ""
    full_prompt = f"Extract all text content from this screenshot. {prompt} {lang_hint}"

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": full_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_image}"
                }
            }
        ]
    }]

    default_model = get_default_model() or "gpt-4o"
    result = await client.chat(default_model, messages, max_tokens=4096)
    return result


async def handle_diagnose_error_screenshot(image_source: str, prompt: str, context: Optional[str] = None) -> str:
    """Handle error screenshot diagnosis request"""
    client = get_vlm_client()
    b64_image = get_image_b64(image_source)

    context_info = f"Error context: {context}" if context else ""
    full_prompt = f"Analyze this error screenshot, diagnose the problem cause and provide solutions. {context_info} {prompt}"

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": full_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_image}"
                }
            }
        ]
    }]

    default_model = get_default_model() or "gpt-4o"
    result = await client.chat(default_model, messages, max_tokens=4096)
    return result


async def handle_understand_technical_diagram(image_source: str, prompt: str, diagram_type: Optional[str] = None) -> str:
    """Handle technical diagram analysis request"""
    client = get_vlm_client()
    b64_image = get_image_b64(image_source)

    type_info = f"Diagram type: {diagram_type}" if diagram_type else "Please auto-detect the diagram type"
    full_prompt = f"Analyze this technical diagram. {type_info} {prompt}"

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": full_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_image}"
                }
            }
        ]
    }]

    default_model = get_default_model() or "gpt-4o"
    result = await client.chat(default_model, messages, max_tokens=4096)
    return result


async def handle_analyze_data_visualization(image_source: str, prompt: str, analysis_focus: Optional[str] = None) -> str:
    """Handle data visualization analysis request"""
    client = get_vlm_client()
    b64_image = get_image_b64(image_source)

    focus_info = f"Analysis focus: {analysis_focus}" if analysis_focus else "Please provide comprehensive analysis"
    full_prompt = f"Analyze this data visualization chart. {focus_info} {prompt}"

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": full_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_image}"
                }
            }
        ]
    }]

    default_model = get_default_model() or "gpt-4o"
    result = await client.chat(default_model, messages, max_tokens=4096)
    return result


async def handle_ui_diff_check(expected_image_source: str, actual_image_source: str, prompt: str) -> str:
    """Handle UI comparison request"""
    client = get_vlm_client()
    b64_expected = get_image_b64(expected_image_source)
    b64_actual = get_image_b64(actual_image_source)

    full_prompt = f"Compare these two UI screenshots and identify visual differences. {prompt}"

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"{full_prompt}\n\n[Expected UI]"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_expected}"
                }
            },
            {"type": "text", "text": "\n\n[Actual UI]"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_actual}"
                }
            }
        ]
    }]

    default_model = get_default_model() or "gpt-4o"
    result = await client.chat(default_model, messages, max_tokens=4096)
    return result


async def handle_analyze_image(image_source: str, prompt: str) -> str:
    """Handle general image analysis request"""
    client = get_vlm_client()
    b64_image = get_image_b64(image_source)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_image}"
                }
            }
        ]
    }]

    default_model = get_default_model() or "gpt-4o"
    result = await client.chat(default_model, messages, max_tokens=4096)
    return result


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Call tool to handle request"""
    try:
        if name == "extract_text_from_image":
            result = await handle_extract_text(
                image_path=arguments.get("image_path"),
                image_data=arguments.get("image_data")
            )
        # ZAI MCP Server style tools
        elif name == "ui_to_artifact":
            result = await handle_ui_to_artifact(
                image_source=arguments["image_source"],
                output_type=arguments["output_type"],
                prompt=arguments["prompt"]
            )
        elif name == "extract_text_from_screenshot":
            result = await handle_extract_text_from_screenshot(
                image_source=arguments["image_source"],
                prompt=arguments["prompt"],
                programming_language=arguments.get("programming_language")
            )
        elif name == "diagnose_error_screenshot":
            result = await handle_diagnose_error_screenshot(
                image_source=arguments["image_source"],
                prompt=arguments["prompt"],
                context=arguments.get("context")
            )
        elif name == "understand_technical_diagram":
            result = await handle_understand_technical_diagram(
                image_source=arguments["image_source"],
                prompt=arguments["prompt"],
                diagram_type=arguments.get("diagram_type")
            )
        elif name == "analyze_data_visualization":
            result = await handle_analyze_data_visualization(
                image_source=arguments["image_source"],
                prompt=arguments["prompt"],
                analysis_focus=arguments.get("analysis_focus")
            )
        elif name == "ui_diff_check":
            result = await handle_ui_diff_check(
                expected_image_source=arguments["expected_image_source"],
                actual_image_source=arguments["actual_image_source"],
                prompt=arguments["prompt"]
            )
        elif name == "analyze_image":
            result = await handle_analyze_image(
                image_source=arguments["image_source"],
                prompt=arguments["prompt"]
            )
        else:
            raise ValueError(f"Unknown tool: {name}")

        return [TextContent(type="text", text=result)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Main function"""
    # Load .env file if exists
    load_dotenv()

    # Initialize client from environment variables
    api_key = os.environ.get("VLM_API_KEY", "")
    base_url = os.environ.get("VLM_BASE_URL")
    default_model = os.environ.get("VLM_MODEL", "")
    max_image_size_str = os.environ.get("VLM_MAX_IMAGE_SIZE", "")

    # Parse max image size (supports formats like "3MB", "3M", "3145728")
    max_image_size = None
    if max_image_size_str:
        max_image_size_str = max_image_size_str.strip().upper()
        if max_image_size_str.endswith("MB"):
            max_image_size = int(float(max_image_size_str[:-2]) * 1024 * 1024)
        elif max_image_size_str.endswith("M"):
            max_image_size = int(float(max_image_size_str[:-1]) * 1024 * 1024)
        elif max_image_size_str.endswith("KB"):
            max_image_size = int(float(max_image_size_str[:-2]) * 1024)
        elif max_image_size_str.endswith("K"):
            max_image_size = int(float(max_image_size_str[:-1]) * 1024)
        else:
            try:
                max_image_size = int(max_image_size_str)
            except ValueError:
                pass

    if not api_key:
        raise ValueError("VLM_API_KEY environment variable is required")

    init_vlm_client(api_key, base_url, default_model or None, max_image_size)

    # Run server
    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="vlm-mcp",
                    server_version="0.2.0",
                    capabilities={
                        "tools": {}
                    }
                )
            )
    finally:
        # Close httpx client on exit
        client = get_vlm_client()
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())


def run():
    """Command line entry point"""
    asyncio.run(main())
