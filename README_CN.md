# 懒人路由 (LazyRouter)

[English](README.md) | [中文](README_CN.md)

<p align="center">
  <img src="assets/lazyrouter_logo.png" alt="LazyRouter Logo" width="280"/>
</p>

懒人路由是一个轻量级的 OpenAI 兼容路由器，能够为每个请求自动选择最合适的模型。

它的设计理念是简单易用：在 YAML 中定义提供商和模型，调用 `model: "auto"`，让路由器自动选择。

## 为什么需要它

在智能体工作流中，上下文增长很快，token 消耗也会变得昂贵。如果没有智能路由，像 "hi" 或 "hello" 这样的简单提示也会调用高端模型（比如 Opus），这并不经济。

懒人路由通过在中间放置一个便宜、快速的路由模型作为守门人来解决这个问题：

- 它为每个请求选择合适的模型，而不是总是使用最贵的那个。
- 它减少了长时间运行的智能体会话中的不必要开支（特别是 OpenClaw 风格的工作流）。
- 它保持单一的 OpenAI 兼容接口，同时在后台处理不同提供商的差异。

它还帮助在不同 API 风格（OpenAI、Gemini 和 Anthropic）之间进行转换。

## 特性亮点

- OpenAI 兼容的 `/v1/chat/completions` 端点
- 基于 LLM 的路由，无需额外的训练流程
- 单一配置支持多个提供商（OpenAI、Anthropic、Gemini、OpenAI 兼容网关）
- 可作为 OpenClaw 等智能体框架的成本控制守门人
- 内置 OpenAI、Gemini 和 Anthropic 风格之间的兼容性处理
- 支持流式和非流式响应
- 健康检查和基准测试端点，提供运维可见性

## 快速开始

### 方式一：直接从 GitHub 运行（无需克隆）

1. 安装 `uv`：<https://docs.astral.sh/uv/getting-started/installation/>
2. 创建配置文件（下载 [config.example.yaml](https://github.com/mysteriousHerb/lazyrouter/blob/main/config.example.yaml) 作为起点）。你可以直接在 config.yaml 中填写 API 密钥，无需 .env 文件。
3. 运行：

```bash
uvx --from git+https://github.com/mysteriousHerb/lazyrouter.git lazyrouter --config config.yaml
```

### 方式二：克隆并本地运行

1. 安装 `uv`：<https://docs.astral.sh/uv/getting-started/installation/>
2. 克隆仓库并安装依赖：

```bash
git clone https://github.com/mysteriousHerb/lazyrouter
cd lazyrouter
uv sync
cp .env.example .env
cp config.example.yaml config.yaml
```

3. 编辑 `.env` 和 `config.yaml`，填入你的 API 密钥、提供商和模型。
4. 启动服务器：

```bash
uv run python main.py --config config.yaml
```

5. 向 `http://localhost:1234/v1/chat/completions` 发送请求。

## 配置说明

使用 `config.example.yaml` 作为基础。API 密钥从 `.env` 加载。

- `llms` 中的 `coding_elo` / `writing_elo` 是质量信号，可以从 `https://arena.ai/leaderboard` 获取。
- `context_compression` 控制在长时间智能体运行期间如何积极地修剪旧历史记录，以控制 token 使用和成本。

## OpenClaw 集成

编辑你的 `.openclaw/openclaw.json`，在 `models.providers` 下添加 LazyRouter 提供商：

```json
{
  "models": {
    "providers": {
      "lazyrouter": {
        "baseUrl": "http://server-address:port/v1",
        "apiKey": "not-needed",
        "api": "openai-completions",
        "models": []
      }
    }
  }
}
```

然后将智能体主模型设置为：

```json
 "agents": {
    "defaults": {
      "model": {
        "primary": "lazyrouter/auto"
      },
    }
  }
```

### 请求示例

```bash
curl -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "简要解释向量数据库"}]
  }'
```

## API 端点

- `GET /health`
- `GET /v1/models`
- `GET /v1/health-status`
- `GET /v1/benchmark`
- `POST /v1/chat/completions`

## 技术实现

懒人路由使用轻量级的基于 LLM 的路由架构：

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  客户端请求      │────▶│  路由模型         │────▶│  上下文裁剪       │────▶│  LLM 提供商      │
│  (model: auto)  │     │  (便宜且快速)     │     │  (token 控制)    │     │  (via LiteLLM)  │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └─────────────────┘
                               │                                                   │
                               │ 选择最佳模型                                        │
                               ▼                                                   ▼
                        ┌──────────────────┐                              ┌─────────────────┐
                        │ OpenAI/Anthropic │                              │    返回响应      │
                        │ Gemini/Custom    │                              │    给客户端      │
                        └──────────────────┘                              └─────────────────┘
```

核心组件：

- **LLMRouter** (`router.py`)：使用便宜/快速的模型（如 GPT-4o-mini、Gemini Flash）分析请求，根据 Elo 评分、定价和任务复杂度选择最优模型。返回带有推理过程的结构化 JSON。

- **FastAPI 服务器** (`server.py`)：OpenAI 兼容的 `/v1/chat/completions` 端点，支持流式传输。处理 Gemini/Anthropic 的特定消息格式转换。

- **上下文压缩** (`context_compressor.py`)：修剪对话历史以控制长智能体会话中的 token 使用。可通过 `max_history_tokens` 和 `keep_recent_exchanges` 配置。

- **健康检查器** (`health_checker.py`)：后台任务，定期 ping 模型并将不健康的模型从路由决策中排除。

- **工具缓存** (`tool_cache.py`)：按会话缓存工具调用 ID 到模型的映射，使工具延续时可以绕过路由器以降低延迟。

- **LiteLLM 集成**：所有提供商调用都通过 LiteLLM 进行，设置 `drop_params=True` 以自动处理 OpenAI、Anthropic 和 Gemini API 之间的兼容性。

## 开发

```bash
uv run python tests/test_setup.py
uv run pytest -q
```

## 文档

- `docs/README.md`（文档索引）
- `docs/QUICKSTART.md`
- `docs/API_STYLES.md`
- `docs/QUICKSTART_API_STYLES.md`
- `docs/UV_GUIDE.md`

## 许可证

GNU 通用公共许可证第 3 版，2007 年 6 月 29 日
