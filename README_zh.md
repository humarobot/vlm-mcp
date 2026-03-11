# VLM MCP Server

[English Version](./README.md)

## 为什么做这个项目?

当使用 Claude Code 接入第三方模型时,往往都是文本模型,没有图片的处理能力。因此,增加一个具有图片处理能力的 MCP 服务器对于需要视觉理解的任务来说十分有必要。

这个项目的目的就是让用户自己选择要处理图片的 VLM (Vision-Language Model) 模型。

## 功能

- **extract_text_from_image**: 从图片中提取文字 (OCR)
- **ui_to_artifact**: 将 UI 截图转换为代码、提示词、设计规格或描述
- **extract_text_from_screenshot**: 从截图提取文本,支持代码识别
- **diagnose_error_screenshot**: 分析错误截图,诊断问题原因
- **understand_technical_diagram**: 分析技术图表(架构图、流程图、UML等)
- **analyze_data_visualization**: 分析数据可视化图表
- **ui_diff_check**: UI 对比检测,找出视觉差异
- **analyze_image**: 通用图片分析

## 环境变量

| 变量名 | 必填 | 说明 |
|--------|------|------|
| `VLM_API_KEY` | 是 | API 密钥 |
| `VLM_BASE_URL` | 否 | 自定义 API 地址(默认: https://api.openai.com/v1) |
| `VLM_MODEL` | 否 | 使用的模型(默认: gpt-4o) |
| `VLM_MAX_IMAGE_SIZE` | 否 | 最大图片大小(默认: 3MB). 超过此大小的图片会自动压缩后再处理. 支持格式: `3MB`, `3M`, `3145728`(字节), `1024KB` 等 |

## 快速开始

### 使用 uvx 运行(推荐)

```bash
# 复制配置模板并填入你的 API Key
cp .env.example .env
# 编辑 .env 文件填入 VLM_API_KEY

# 直接运行 (会自动加载 .env 文件)
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

任何兼容 OpenAI Chat Completions API 的 VLM 模型:

- gpt-4o
- gpt-4o-mini
- gpt-4-turbo
- qwen2.5-vl 系列
- 及其他兼容 OpenAI API 的模型

## Claude Code 配置

### 1. 配置 MCP 服务器

在 Claude Code 中配置 MCP 服务器:

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

### 2. 配置 CLAUDE.md

为确保 Claude Code 使用 MCP 工具读取图片而非内置的 Read 工具,请在你的项目或全局 CLAUDE.md 中添加:

```markdown
## MCP Priority

1. Use mcp tools to read images instead of claude code's read tool.
```

## 使用示例

在 Claude Code 中使用:

```
请用 extract_text_from_image 工具分析这张图片 /path/to/image.jpg,提取其中的文字。
```

```
请用 ui_to_artifact 工具将这个UI截图转换为 React 代码。
```

---

设计参考了智谱 AI 的方案。
