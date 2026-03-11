# VLM MCP Server

支持 OpenAI 协议兼容的 VLM 图片理解 MCP Server。

## 功能

- **extract_text_from_image**: 从图片中提取文字 (OCR)
- **ui_to_artifact**: 将 UI 截图转换为代码、提示词、设计规格或描述
- **extract_text_from_screenshot**: 从截图提取文本，支持代码识别
- **diagnose_error_screenshot**: 分析错误截图，诊断问题原因
- **understand_technical_diagram**: 分析技术图表（架构图、流程图、UML等）
- **analyze_data_visualization**: 分析数据可视化图表
- **ui_diff_check**: UI 对比检测，找出视觉差异
- **analyze_image**: 通用图片分析

## 环境变量

| 变量名 | 必填 | 说明 |
|--------|------|------|
| `VLM_API_KEY` | 是 | API 密钥 |
| `VLM_BASE_URL` | 否 | 自定义 API 地址（默认: https://api.openai.com/v1） |
| `VLM_MODEL` | 否 | 使用的模型（默认: gpt-4o） |

## 快速开始

### 使用 uvx 运行（推荐）

```bash
# 设置环境变量
export VLM_API_KEY=your-api-key
export VLM_MODEL=gpt-4o

# 直接运行
uvx vlm-mcp
```

### 使用 pip 安装

```bash
# 安装
pip install vlm-mcp

# 或开发模式安装
pip install -e .
```

### 配置环境变量

```bash
# OpenAI
export VLM_API_KEY=sk-xxx
export VLM_MODEL=gpt-4o

# 自定义 API (如 Ollama)
export VLM_API_KEY=your-api-key
export VLM_BASE_URL=http://localhost:11434/v1
export VLM_MODEL=qwen2.5-vl
```

### 运行服务

```bash
# 直接运行
python -m vlm_mcp

# 或使用安装的命令
vlm-mcp
```

## 支持的模型

任何兼容 OpenAI Chat Completions API 的 VLM 模型：

- gpt-4o
- gpt-4o-mini
- gpt-4-turbo
- qwen2.5-vl 系列
- 及其他兼容 OpenAI API 的模型

## Claude Code 配置

在 Claude Code 中配置 MCP 服务器：

```json
{
  "mcpServers": {
    "vlm-mcp": {
      "command": "uvx",
      "args": ["vlm-mcp"],
      "env": {
        "VLM_API_KEY": "your-api-key",
        "VLM_MODEL": "gpt-4o"
      }
    }
  }
}
```

## 使用示例

在 Claude Code 中使用：

```
请用 extract_text_from_image 工具分析这张图片 /path/to/image.jpg，提取其中的文字。
```

```
请用 ui_to_artifact 工具将这个UI截图转换为 React 代码。
```
