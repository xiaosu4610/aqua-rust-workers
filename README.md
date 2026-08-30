# AQUA — 多协议 AI API 网关

<div align="center">

**免费 · 极速 · 免注册的 OpenAI 兼容 AI 网关**

Rust → WebAssembly · Cloudflare Workers 边缘运行 · 多上游聚合 · 任意密钥即用

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-stable-orange.svg)](https://rustup.rs/)
[![Cloudflare Workers](https://img.shields.io/badge/Cloudflare-Workers-f38020.svg)](https://workers.cloudflare.com/)

**开源仓库：[gitee.com/xiaosu4610/aqua-rust-workers](https://gitee.com/xiaosu4610/aqua-rust-workers) · 觉得不错请点个 Star ⭐**

</div>

---

## 这是什么

AQUA 把 **Nvidia NIM、Gitee AI、SiliconFlow、智谱 GLM、讯飞星火、Cloudflare Workers AI** 等多家 AI 上游，聚合成一套 **OpenAI 兼容 API**。用任何支持 OpenAI SDK 的客户端（ChatGPT 客户端、LobeChat、NextChat、Dify、沉浸式翻译等），填一个 Base URL 就能用上全线模型。

网关本体用 **Rust 编译为 WebAssembly**，跑在 Cloudflare 边缘节点上：全球就近接入、无冷启动、免费套餐即可运行。

**任意密钥即可调用**——不需要注册、不需要申请，填任意非空 Key（`sk-****`、甚至中文）就能用。

## 模型通道

| 通道 | 模型名前缀 | 说明 |
|---|---|---|
| Nvidia NIM | 无前缀 / `nvidia/` | 默认通道；数百密钥池轮询，限流自动换 Key 重试 |
| Gitee AI | `gitee-ai/` | 含 IP 归属地查询等特色端点 |
| SiliconFlow | `siliconflow/` | — |
| 智谱 GLM | `zhipu/` | 含 CogView 绘图、CogVideo 视频 |
| 讯飞星火 | `spark/` | — |
| Workers AI | `workers-ai/`、`workers-ai-tts/` | Cloudflare 自家 AI，免费日额度管理 |
| 自定义上游 | `acu/` | 直连部署者自己的专属上游（地址+密钥均由环境变量配置） |

模型名带前缀自动路由到对应上游；不带前缀的走静态模型目录识别，未识别的兜底到 Nvidia。

## 工具箱与游戏（网页 + API 双形态）

网关不只是 AI 代理，还内置了一套「工具 API」与配套网页应用（部署前台后访问「工具箱」/「在线体验」页）：

- **AI 对弈游戏**：井字棋（minimax 必不败引擎 / 可选 LLM 对手）、五子棋 9×9（棋型评分引擎 / LLM 落子）、猜数字 1A2B（人猜 AI / LLM 挑战模式）、成语接龙（LLM 驱动 + 首尾字校验）
- **AI 赋能工具**：IP 归属地查询（AI 解读）、文本内容审核（AI 风险分析）、AI 翻译、AI 摘要
- **纯算法工具**：文本统计、UUID、时间戳互转、Base64/URL 编解码、JSON 格式化

所有工具能力同时以 REST API 开放（`/v1/tools/*` 命名空间），鉴权与主 API 一致——**我们不只提供 AI API，也提供工具 API**。

## API 端点（OpenAI 兼容）

| 端点 | 方法 | 说明 |
|---|---|---|
| `/v1/models` | GET | 模型列表（公开访问，实时反映上游可用性） |
| `/v1/chat/completions` | POST | 对话补全，`stream: true` 时 SSE 流式透传 |
| `/v1/embeddings` | POST | 文本向量化 |
| `/v1/rerank` | POST | 重排序 |
| `/v1/moderations` | POST | 内容审核 |
| `/v1/images/generations` | POST | 图像生成 |
| `/v1/videos/generations` | POST | 视频生成 |
| `/v1/audio/speech` | POST | 语音合成 TTS |
| `/v1/audio/transcriptions` | POST | 语音识别 ASR（multipart 上传） |
| `/v1/ip_location` | POST | IP 归属地查询（双通道：Gitee AI 主 + ip-api.com 免费备用，大模型掉线依然可用） |
| `/v1/tools/text-stats` | POST | 文本统计（字数/词频/阅读时长，纯算法） |
| `/v1/tools/dice` | POST | 随机骰子（可指定面数与数量） |
| `/v1/tools/uuid` | GET | 生成 UUID v4 |
| `/v1/tools/timestamp` | GET/POST | 当前时间戳查询 / 时间戳互转 |
| `/v1/tools/base64` | POST | Base64 编解码 |
| `/v1/tools/subnet` | POST | IPv4 子网计算器（网络/广播地址、主机范围、掩码，纯算法） |
| `/assets/*` | GET | 生成图片的 R2 缓存（24h 自动清理） |

所有错误响应为 OpenAI 兼容结构，并附带 `help` 字段（官网、QQ 频道/群引导），方便客户端直接展示排障信息。

## 工程实现

三个 **Durable Object** 负责有状态调度（单线程模型，天然无竞态）：

- **NvKeyPool** — Nvidia 密钥池：随机轮询、每 Key 每分钟 38 次限速（避开上游 429）、限流冷却、失效隔离；模型封禁只针对 client_error（400/404），5xx/429 仅冷却 Key 不误杀模型
- **WaiBudget** — Workers AI 免费日额度：原子计数与每日重置，超额自动熔断返回 429，保护部署者的账号额度不被打爆
- **AcuConcurrency** — 自定义上游全局并发闸：并发满时排队等待（最长 10s），保护后端

其他特性：

- **供应商隔离**：封禁/健康状态按通道隔离，一个上游故障不影响其他通道
- **密钥分片合并**：Cloudflare 单环境变量上限 5.1KB，`NVIDIA_KEYS` 支持自动切分为 `NVIDIA_KEYS_2..N`，运行时透明合并成完整池
- **鉴权双模式**：`AUTH_MODE` 一键切换开放/密钥制（见下）

## 鉴权双模式

| AUTH_MODE | 行为 | 适用场景 |
|---|---|---|
| `open`（默认，未配置时） | 任意非空密钥均可使用，中英文皆可 | 公益开放 / 个人自用 |
| `key` | 仅 `GATEWAY_KEYS` 列表中的密钥可用，其余返回 401 | 防滥用 / 私有部署 |

401 响应会附官网与 QQ 频道/群引导字段，客户端可直接取用展示。

## 快速部署（保姆级）

> 全程约 15 分钟。不需要付费：Cloudflare 免费套餐即可运行整个网关。

### 前置条件（逐个检查）

```bash
# 1. 安装 Rust（Windows 用户下载 rustup-init.exe；已有跳过）
#    https://rustup.rs/
rustup --version

# 2. 添加 Wasm 编译目标（必做，否则 cargo build 报错）
rustup target add wasm32-unknown-unknown

# 3. 安装 Node.js ≥ 18（https://nodejs.org/ 下载 LTS 版）
node --version

# 4. 安装 wrangler 并登录 Cloudflare（会弹浏览器授权）
npm install -g wrangler
wrangler login
wrangler whoami   # 显示你的账户邮箱即成功
```

> 中国大陆网络建议保留 `.cargo/config.toml`（rsproxy 镜像加速）；Windows 用户若未装 pwsh，把命令里的 `powershell` 换成 `pwsh` 或安装 PowerShell 7。

### 1. 配置上游密钥

```bash
cd gateway
cp vars.example.toml vars.toml   # vars.toml 已被 .gitignore 排除，绝不入库
```

编辑 `vars.toml`，填入你自己的上游密钥（Nvidia / Gitee / SiliconFlow / 智谱 / 星火 / 自定义上游，有几项填几项，没配的通道自动禁用）。

### 2. 替换 Cloudflare 资源

编辑 `gateway/wrangler.toml`，替换为你自己的资源 ID 与域名：

| 占位符 | 获取方式 |
|---|---|
| `REPLACE_WITH_KV_ID` | `wrangler kv namespace create MODEL_CACHE` |
| `REPLACE_WITH_D1_ID` | `wrangler d1 create aqua_logs` |
| `your-gateway-domain.example` | 你的网关域名（frontend/wrangler.toml 同理） |

### 3. 构建

```powershell
powershell -File scripts/build.ps1 gateway    # 构建网关
powershell -File scripts/build.ps1 frontend   # 构建前台
```

流程：cargo 编译 wasm32 → wasm-bindgen 生成胶水 → esbuild 打包 legacy shim。

### 4. 注入密钥并部署

```bash
# 方式一：wrangler secret（推荐，密钥存加密存储不落文件）
wrangler secret put NVIDIA_KEYS
wrangler secret put AUTH_MODE

# 方式二：写入 wrangler.toml [env.production.vars] 后部署（勿提交该文件）
wrangler deploy --env production
```

### 5. 验证

```bash
curl https://your-gateway-domain.example/v1/models
curl https://your-gateway-domain.example/v1/chat/completions \
  -H "Authorization: Bearer whatever-you-like" \
  -H "Content-Type: application/json" \
  -d '{"model":"acu/deepseek-v4-flash","messages":[{"role":"user","content":"你好"}]}'
```

返回模型列表 JSON / 对话回复即为成功。客户端（LobeChat、NextChat、沉浸式翻译等）填：

- **API 地址**：`https://你的网关域名/v1`（或根地址 `https://你的网关域名`，多数客户端自动补 `/v1`）
- **API Key**：任意非空字符串（默认 open 模式）

## 常见问题（部署排障）

<details>
<summary><b>cargo build 报错：target wasm32-unknown-unknown not installed</b></summary>

执行 `rustup target add wasm32-unknown-unknown` 后重新构建。
</details>

<details>
<summary><b>构建时报 wasm-bindgen 版本不匹配</b></summary>

本项目使用 wasm-bindgen 0.2.127。确认 `.tools/wasm-bindgen/` 下有对应版本可执行文件，或用 `cargo install wasm-bindgen-cli --version 0.2.127` 安装后修改 build.ps1 中的路径。
</details>

<details>
<summary><b>wrangler deploy 报错 10054 / 变量过大</b></summary>

单个环境变量超过 CF 5.1KB 上限。NVIDIA_KEYS 密钥太多时，手动分成 `NVIDIA_KEYS`、`NVIDIA_KEYS_2`、`NVIDIA_KEYS_3`... 多片（每片 ≤4800 字符，按逗号边界切），网关运行时自动合并。
</details>

<details>
<summary><b>部署成功但请求返回 502</b></summary>

502 = 对应上游未配置或不可达。检查 `vars.toml` 对应通道的密钥是否已通过 secret 或 vars 注入（占位符 `REPLACE_WITH_REAL_KEY` 会被视为未配置）。
</details>

<details>
<summary><b>部署成功但请求返回 401</b></summary>

当前为 `AUTH_MODE=key` 模式且密钥不在 `GATEWAY_KEYS` 列表。要么用列表内的密钥，要么把 `AUTH_MODE` 改为 `open`（或直接删除该变量）后重新部署。
</details>

<details>
<summary><b>模型列表能出来但对话报 429</b></summary>

Workers AI 日额度用尽（WaiBudget 熔断）或上游限流。等待每天 00:00 UTC 自动重置，或改用其他平台模型。
</details>

<details>
<summary><b>前台模型列表加载失败</b></summary>

检查 `frontend/public/index.html` 中的 `var GATEWAY` 是否已改成你的网关地址（含 `/v1`）。
</details>

<details>
<summary><b>想只给自己用，不让别人调用</b></summary>

`AUTH_MODE = "key"`，`GATEWAY_KEYS = "我的密钥"`，重新部署。只有知道这把密钥的人能用。
</details>

## 环境变量

完整样例见 [gateway/vars.example.toml](gateway/vars.example.toml)：

| 变量 | 默认 | 说明 |
|---|---|---|
| `AUTH_MODE` | `open` | `open` 任意密钥可用；`key` 指定密钥制 |
| `GATEWAY_KEYS` | — | `AUTH_MODE=key` 时的合法密钥列表，逗号分隔多把平滑轮换 |
| `NVIDIA_KEYS` | — | Nvidia 密钥池（逗号分隔可上百个；支持 `_2.._N` 分片） |
| `NVIDIA_BASE` 等 `*_BASE` | 官方地址 | 各上游 Base URL，一般不用改 |
| `GITEE_KEY` / `SILICONFLOW_KEY` / `ZHIPU_KEY` / `SPARK_KEY` | — | 对应上游密钥，未配置则该通道 502 |
| `ACU_BASE` / `ACU_KEY` | — | 自定义专属上游地址与密钥 |
| `WAI_ACCOUNT_ID` / `WAI_API_TOKEN` / `WAI_CAP_GLOBAL` | — | Workers AI 凭据（**部署者自用，配合部署者自己的 Cloudflare 账号**；请勿填入他人账号） |

## 项目结构（每个文件夹都有独立 README 详解）

> 🔧 想二次开发 / 扩展功能 / 深入理解代码？请阅读 **[DEVELOPMENT.md](DEVELOPMENT.md)**——完整的架构导读、扩展实操（新增模型/供应商/工具）、本地调试与排障手册。

```
aqua-worker/
├── gateway/                 # 网关核心（Rust → Wasm）→ 详见 gateway/README.md
│   ├── src/                 #   Rust 源码（路由/鉴权/三个 DO）
│   ├── vars.example.toml    #   环境变量样例（真实值不入库）
│   └── wrangler.toml        #   Workers 配置
├── frontend/                # 用户前台（Rust Worker）→ 详见 frontend/README.md
│   ├── src/lib.rs           #   静态资源 + SPA 路由
│   └── public/index.html    #   全部前端内容（单文件单页应用）
├── scripts/                 # 构建脚本 → 详见 scripts/README.md
│   ├── build.ps1            #   一键构建（cargo → wasm-bindgen → esbuild）
│   └── shim.legacy.template.js
└── .cargo/                  # Rust 编译配置（国内镜像加速）→ 详见 .cargo/README.md
```

## 隐私与安全

- 仓库**不含任何真实密钥、上游地址、Cloudflare 资源 ID**，全部经环境变量或 secret 注入
- `vars.toml`、`wrangler.local.toml`、`*.env` 均被 `.gitignore` 排除
- 未配置的上游通道返回 502 且不泄露任何内部信息
- 构建产物（`build/`）不入库

## 社区

- **QQ 频道（官方主阵地）**：大版本更新与重要公告均在此通知 → [点击加入](https://pd.qq.com/s/e4ktxw1b8)（频道号 `pd57362562`）
- **QQ 群（休闲交流）**：日常闲聊、技术交流 → 群号 `1103667832`

## 开源协议

本项目采用 **[GNU AGPL-3.0](LICENSE)** 协议开源。

> **请注意：不同的开源协议所赋予的权利与约束是不同的。**
>
> - 本项目此前使用的 **MIT** 协议最为宽松：允许任意使用、修改、闭源甚至商用，唯一义务是保留版权声明。
> - 现行的 **AGPL-3.0** 是强保护（copyleft）协议：任何人**修改本项目的代码后对外提供（包括仅部署为线上服务、不分发二进制的情况）**，都必须以 AGPL-3.0 协议向其用户**开放完整源码**，并保留原版权声明与协议文本。
>
> 因此，如果你打算基于 AQUA 二次开发：
> - **个人学习、内部使用** → 完全自由，无任何额外义务；
> - **二开后对外提供服务或分发** → 必须同样以 AGPL-3.0 开源你的修改版本。想闭源商用需联系作者获得**商业授权**。

---

<div align="center">

如果 AQUA 对你有帮助，欢迎到 [Gitee 仓库](https://gitee.com/xiaosu4610/aqua-rust-workers) 点个 **Star ⭐** 支持

</div>
