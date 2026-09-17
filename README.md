# agent-sight

多模态 RAG 两级检索 · 以图搜图（Go 实现，CLI + Web UI）

基于云厂商多模态嵌入模型 + VLM 大模型精排的两级检索架构，支持硅基流动 / 模力方舟 双平台，开箱即用。

## 架构

```
查询图 ──▶ 【第一级】多模态嵌入粗召回（毫秒级、低成本）
              │  Qwen3-VL-Embedding-8B / BGE-M3 / CLIP
              │  向量库快速 Top-K 召回
              ▼
           【第二级】VLM 精排（高精度、按需触发）
              │  Qwen3-VL-8B-Instruct 等
              │  视觉相似度打分 + 文字条件筛选 + 匹配理由
              ▼
           最终排序 + 匹配解释
```

90% 请求走嵌入模型毫秒级返回，只有需要精准判断或图文混合时才触发 VLM 精排。

## 功能

- **两级检索**：`search`（嵌入）+ `search-pro`（嵌入 + VLM 精排）
- **图文混合**：`search-pro ./q.jpg "有人在笑"` 支持文字筛选
- **自适应阈值**：`-t auto` 基于库内向量分布动态判定，无需人工调参
- **双平台**：SiliconFlow / 模力方舟一键切换
- **Web UI**：Kimi 风格控制台，`web` 命令一键启动
- **本地持久化**：JSON 向量库，增量建库 + 强制重建
- **模型一致性校验**：换模型时自动提示重建
- **跨平台二进制**：linux / darwin / windows × amd64 / arm64

## 快速开始

### 1. 配置

```bash
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY
```

### 2. 编译

```bash
go build -o agent-sight .
```

### 3. 建库

```bash
./agent-sight build ./input         # 增量
./agent-sight build -f ./input      # 强制重建
```

### 4. 检索

```bash
# 快速检索（嵌入）
./agent-sight search ./q.jpg 5
./agent-sight search -t auto ./q.jpg    # 自适应阈值

# 精准检索（嵌入 + VLM 精排）
./agent-sight search-pro ./q.jpg 5
./agent-sight search-pro ./q.jpg "有猫" 3
```

### 5. Web UI

```bash
./agent-sight web          # 默认 http://127.0.0.1:8080
./agent-sight web -p 9000  # 自定义端口
```

自动打开浏览器，左侧栏 4 个 Tab：快速检索 / 精准检索 / 建库 / 图库信息。

## 命令

| 命令 | 说明 |
|---|---|
| `build [-f] [-w N] <目录>` | 批量建库（`-f` 强制重建） |
| `add [-f] <图片>` | 追加单张 |
| `search [-t T] [-k N] [-v] <图片>` | 快速检索（`-t auto` 自适应阈值） |
| `search-pro [-t T] [-k N] <图片> [文字]` | 精准检索（+ VLM 精排） |
| `info` | 查看图库信息 |
| `web [-p PORT]` | 启动 Web UI |

## 环境变量（.env）

| 变量 | 说明 | 默认 |
|---|---|---|
| `OPENAI_API_KEY` | API 密钥 | 必填 |
| `OPENAI_BASE_URL` | API 端点 | `https://api.siliconflow.cn/v1` |
| `OPENAI_MODEL` | 嵌入模型 | `Qwen/Qwen3-VL-Embedding-8B` |
| `VLM_MODEL` | 精排模型 | `Qwen/Qwen3-VL-8B-Instruct` |
| `AUTO_RERANK_THRESHOLD` | 阈值（数字或 `auto`） | `auto` |
| `RECALL_CANDIDATES` | 嵌入召回候选数 | `50` |
| `VLM_BATCH_SIZE` | VLM 送图数 | `6` |
| `VLM_TIMEOUT` | VLM 超时（秒） | `120` |

## 多平台发布

使用 GitHub Releases，二进制命名：`agent-sight_<os>_<arch>.zip`

## 项目结构

```
.
├── main.go           # CLI 入口 + Web 命令分发
├── web.go            # Web 服务器 + 内嵌 UI
├── embed.go          # 多模态嵌入（SiliconFlow / 模力方舟）
├── reranker.go       # VLM 精排
├── vectorstore.go    # JSON 向量库
├── go.mod
├── .env.example
└── README.md
```

## License

MIT
