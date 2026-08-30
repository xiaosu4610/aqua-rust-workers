//! AQUA API Gateway — Cloudflare Worker (Rust → Wasm)
//!
//! ## 路由总览
//! - GET  /v1/models                   模型列表（静态目录 + D1 健康评分 + DO 状态）
//! - POST /v1/chat/completions         对话补全（多供应商代理 + 密钥池重试）
//! - POST /v1/audio/speech             TTS（workers-ai-tts / gitee）
//! - POST /v1/audio/transcriptions     ASR
//! - POST /v1/embeddings               向量化
//! - POST /v1/rerank                   重排序
//! - POST /v1/moderations              内容安全
//! - POST /v1/images/generations       文生图（zhipu cogview）
//! - POST /v1/videos/generations       文生视频（zhipu cogvideox）
//! - POST /v1/ip_location              IP 定位（双通道：Gitee AI 主 + ip-api.com 备，主通道故障自动兜底）
//! - POST /v1/tools/text-stats|dice|base64    纯算法工具（零上游消耗）
//! - POST /v1/tools/subnet                    IPv4 子网计算器（纯算法）
//! - GET  /v1/tools/uuid|timestamp            纯算法工具
//! - POST /v1/tools/timestamp                 时间戳互转
//! - GET  /assets/*                    R2 图片缓存
//!
//! ## 二次开发指引（详见仓库 DEVELOPMENT.md）
//! - 新增模型：在 MODEL_CATALOG 追加 (ID, owned_by)，重编译部署即生效
//! - 新增供应商：provider_cfg 加一条 (BASE 变量, 默认值, KEY 变量)，
//!   provider_of 加前缀匹配，handle_chat 的 match 里加转发分支
//! - 新增工具 API：写 handler → main 的 router 注册路由即可
//!
//! ## 错误响应约定（非传统纯 HTTP 状态码机制）
//! 所有错误统一返回结构化 JSON，程序识别主键为字符串业务错误码：
//! ```json
//! { "error": { "message": "中文原因", "code": "MODEL_UNAVAILABLE", "status": 502, "type": "aqua_api_error",
//!     "hint": "如需帮助：加入 QQ 频道 pd57362562（…）反馈，或访问官网 …",
//!     "help": { "site", "docs", "qq_guild", "qq_guild_url", "qq_guild_invite", "qq_group" } } }
//! ```
//! - `code`：稳定字符串错误码（MODEL_UNAVAILABLE / RATE_LIMITED / QUOTA_EXHAUSTED / …）
//! - `status`：与 HTTP 状态码一致的数字（兼容旧程序与 OpenAI SDK）
//! - 上游 4xx/5xx 错误一律经 normalize_upstream 归一化为中文说明（不再透传上游英文报文）
//! - Content-Type 恒为 `application/json; charset=utf-8`，中文在终端/浏览器直接可读
//!
//! ## 性能要点
//! 模型列表使用编译期静态目录，不依赖上游实时查询，
//! 每次请求仅做 D1 健康聚合（窗口函数 ~130ms）+ 缓存，消除冷启动 60s+。

use std::collections::HashMap;
use wasm_bindgen::JsValue;
use worker::*;

mod acu_limit;
mod keypool;
mod keys;
mod workers_ai;

// ---------------------------------------------------------------------------
// 静态模型目录：(模型 ID, 供应商 owned_by)。编译期打入二进制，保证
// /v1/models 即时返回；新增模型在重新部署时自动收录。
//
// 【二次开发】新增模型只需在此追加一行 (完整模型 ID, 供应商标签)：
//   - 供应商标签决定路由归属（gitee-ai → Gitee 通道，其余非前缀模型 → Nvidia）
//   - 前缀模型（zhipu/xxx、workers-ai/xxx 等）无需登记，自动按前缀路由
// ---------------------------------------------------------------------------
// 2026-08-30 清理：基于上游 7 天健康数据移除 61 个已下线模型（404/410 全失败），
// 保留 83 个验证存活模型（chat 专用端点模型 ASR/TTS/embedding/rerank/图像/视频/IP/审核不计入探测失败）。
const MODEL_CATALOG: &[(&str, &str)] = &[
    ("acu/deepseek-v4-flash", "acu"),
    ("deepseek-ai/deepseek-v4-flash-0731", "deepseek-ai"),
    ("deepseek-ai/deepseek-v4-pro-0813", "deepseek-ai"),
    ("DeepSeek-Prover-V2-7B", "gitee-ai"),
    ("DeepSeek-R1-Distill-Qwen-1.5B", "gitee-ai"),
    ("DeepSeek-R1-Distill-Qwen-14B", "gitee-ai"),
    ("DeepSeek-R1-Distill-Qwen-7B", "gitee-ai"),
    ("GLM-4-9B-0414", "gitee-ai"),
    ("GLM-ASR", "gitee-ai"),
    ("HealthGPT-L14", "gitee-ai"),
    ("HuatuoGPT-o1-7B", "gitee-ai"),
    ("Lingshu-32B", "gitee-ai"),
    ("Qwen2-7B-Instruct", "gitee-ai"),
    ("Qwen3-0.6B", "gitee-ai"),
    ("Qwen3-4B", "gitee-ai"),
    ("Qwen3-8B", "gitee-ai"),
    ("Qwen3-Embedding-4B", "gitee-ai"),
    ("Qwen3-Reranker-0.6B", "gitee-ai"),
    ("Qwen3-Reranker-4B", "gitee-ai"),
    ("Qwen3Guard-Gen-0.6B", "gitee-ai"),
    ("Security-semantic-filtering", "gitee-ai"),
    ("SenseVoiceSmall", "gitee-ai"),
    ("Spark-TTS-0.5B", "gitee-ai"),
    ("bce-reranker-base_v1", "gitee-ai"),
    ("bge-reranker-v2-m3", "gitee-ai"),
    ("glm-4-9b-chat", "gitee-ai"),
    ("internlm3-8b-instruct", "gitee-ai"),
    ("ip-location", "gitee-ai"),
    ("nonescape-v0", "gitee-ai"),
    ("nsfw-classifier", "gitee-ai"),
    ("google/diffusiongemma-26b-a4b-it", "google"),
    ("google/gemma-4-31b-it", "google"),
    ("meta/llama-3.2-11b-vision-instruct", "meta"),
    ("meta/llama-3.2-90b-vision-instruct", "meta"),
    ("meta/llama-guard-4-12b", "meta"),
    ("meta/muse-glimmer-30b", "meta"),
    ("minimaxai/minimax-m3", "minimaxai"),
    ("mistralai/mistral-nemotron", "mistralai"),
    ("moonshotai/kimi-k3", "moonshotai"),
    ("nvidia/ising-calibration-1.5-31b", "nvidia"),
    ("nvidia/llama-3.1-nemoguard-8b-content-safety", "nvidia"),
    ("nvidia/llama-3.1-nemoguard-8b-topic-control", "nvidia"),
    ("nvidia/llama-3.1-nemotron-safety-guard-8b-v3", "nvidia"),
    ("nvidia/nemotron-3-nano-30b-a3b", "nvidia"),
    ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "nvidia"),
    ("nvidia/nemotron-3-super-120b-a12b", "nvidia"),
    ("nvidia/nemotron-3-ultra-550b-a55b", "nvidia"),
    ("nvidia/nemotron-3.5-content-safety", "nvidia"),
    ("nvidia/nemotron-3.5-lightning-30b-a3b", "nvidia"),
    ("nvidia/riva-translate-4b-instruct-v1.1", "nvidia"),
    ("nvidia/riva-translate-4b-instruct-v2", "nvidia"),
    ("openai/gpt-oss-120b", "openai"),
    ("openai/gpt-oss-20b", "openai"),
    ("poolside/laguna-xs-2.1", "poolside"),
    ("BAAI/bge-large-en-v1.5", "siliconflow"),
    ("BAAI/bge-large-zh-v1.5", "siliconflow"),
    ("BAAI/bge-m3", "siliconflow"),
    ("BAAI/bge-reranker-v2-m3", "siliconflow"),
    ("PaddlePaddle/PaddleOCR-VL-1.5", "siliconflow"),
    ("THUDM/GLM-4-9B-0414", "siliconflow"),
    ("THUDM/GLM-Z1-9B-0414", "siliconflow"),
    ("tencent/Hunyuan-MT-7B", "siliconflow"),
    ("spark/spark-lite", "spark"),
    ("workers-ai/deepseek-r1-32b", "workers-ai"),
    ("workers-ai/gemma-3-12b", "workers-ai"),
    ("workers-ai/llama-3.1-8b", "workers-ai"),
    ("workers-ai/llama-3.2-1b", "workers-ai"),
    ("workers-ai/llama-3.2-3b", "workers-ai"),
    ("workers-ai/llama-3.3-70b", "workers-ai"),
    ("workers-ai/mistral-7b", "workers-ai"),
    ("workers-ai/mistral-small-24b", "workers-ai"),
    ("workers-ai/qwen2.5-coder-32b", "workers-ai"),
    ("workers-ai/qwq-32b", "workers-ai"),
    ("workers-ai/melotts", "workers-ai-tts"),
    ("zhipu/cogvideox-flash", "zhipu"),
    ("zhipu/cogview-3-flash", "zhipu"),
    ("zhipu/glm-4-flash", "zhipu"),
    ("zhipu/glm-4-flash-250414", "zhipu"),
    ("zhipu/glm-4.1v-thinking-flash", "zhipu"),
    ("zhipu/glm-4.6v-flash", "zhipu"),
    ("zhipu/glm-4.7-flash", "zhipu"),
    ("zhipu/glm-4v-flash", "zhipu"),
    ("zhipu/glm-z1-flash", "zhipu"),
];

// ---------------------------------------------------------------------------
// 供应商配置（开源友好：全部经环境变量注入，代码零隐私）
// Base URL 有公开默认值；所有 Key 仅来自环境变量，未配置则对应供应商不可用。
// 环境变量清单见 vars.example.toml；真实值只存于本地 vars.toml（不入 Git）与部署配置。
// ---------------------------------------------------------------------------
const NVIDIA_BASE_DEFAULT: &str = "https://integrate.api.nvidia.com";
const GITEE_BASE_DEFAULT: &str = "https://ai.gitee.com/v1";
const SILICONFLOW_BASE_DEFAULT: &str = "https://api.siliconflow.cn/v1";
const ZHIPU_BASE_DEFAULT: &str = "https://open.bigmodel.cn/api/paas/v4";
const SPARK_BASE_DEFAULT: &str = "https://spark-api-open.xf-yun.com/v1";

/// 环境变量一键读取助手（占位符 REPLACE_WITH_REAL_KEY 视为未配置）
fn env_or<'a>(env: &'a Env, name: &str, default: &'a str) -> String {
    env.var(name)
        .map(|v| v.to_string())
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s != "REPLACE_WITH_REAL_KEY")
        .unwrap_or_else(|| default.to_string())
}

pub struct ProviderCfg {
    pub base: String,
    pub key: String,
}

/// 从 URL 提取 host（用于需显式 Host 头的上游）
fn url_host(url: &str) -> Option<&str> {
    let rest = url.strip_prefix("https://").or_else(|| url.strip_prefix("http://"))?;
    let host = rest.split('/').next()?;
    if host.is_empty() {
        None
    } else {
        Some(host)
    }
}

/// 各供应商运行时配置：Base 可用默认（公开信息），Key 强制来自 env
fn provider_cfg(env: &Env, provider: &str) -> Option<ProviderCfg> {
    let (base_var, base_default, key_var) = match provider {
        "nvidia" => ("NVIDIA_BASE", NVIDIA_BASE_DEFAULT, "NVIDIA_KEYS"),
        "gitee" => ("GITEE_BASE", GITEE_BASE_DEFAULT, "GITEE_KEY"),
        "siliconflow" => ("SILICONFLOW_BASE", SILICONFLOW_BASE_DEFAULT, "SILICONFLOW_KEY"),
        "zhipu" => ("ZHIPU_BASE", ZHIPU_BASE_DEFAULT, "ZHIPU_KEY"),
        "spark" => ("SPARK_BASE", SPARK_BASE_DEFAULT, "SPARK_KEY"),
        "acu" => ("ACU_BASE", "", "ACU_KEY"),
        _ => return None,
    };
    let base = env_or(env, base_var, base_default);
    let key = env_or(env, key_var, "");
    if key.is_empty() {
        return None; // 密钥未配置 → 该供应商不可用
    }
    Some(ProviderCfg { base, key })
}

/// 供应商标签（owned_by / 模型前缀 → 上游路由）
/// 路由优先级：显式前缀（zhipu/ 等）> 静态目录查询 > 兜底 Nvidia
fn provider_of(model: &str) -> &'static str {
    if model.starts_with("zhipu/") {
        return "zhipu";
    }
    if model.starts_with("spark/") {
        return "spark";
    }
    if model.starts_with("siliconflow/") {
        return "siliconflow";
    }
    if model.starts_with("gitee-ai/") {
        return "gitee";
    }
    if model.starts_with("workers-ai-tts/") {
        return "workers-ai-tts";
    }
    if model.starts_with("workers-ai/") {
        return "workers-ai";
    }
    if model.starts_with("acu/") {
        return "acu";
    }
    // 静态目录查询 owned_by
    for (id, owner) in MODEL_CATALOG {
        if *id == model {
            return match *owner {
                "gitee-ai" => "gitee",
                "siliconflow" => "siliconflow",
                "zhipu" => "zhipu",
                "spark" => "spark",
                "workers-ai" => "workers-ai",
                "workers-ai-tts" => "workers-ai-tts",
                "acu" => "acu",
                _ => "nvidia",
            };
        }
    }
    "nvidia"
}

fn now_ts() -> i64 {
    (js_sys::Date::new_0().get_time() as i64) / 1000
}

// ---------------------------------------------------------------------------
// worker 0.8.5 辅助：D1 批量绑定 / DO JSON 请求
// ---------------------------------------------------------------------------
fn js_args(args: &[String]) -> Vec<JsValue> {
    args.iter().map(|s| JsValue::from_str(s)).collect()
}

/// 执行 D1 写语句（bind 一次绑定全部参数）
async fn d1_run(db: &D1Database, sql: &str, args: &[String]) {
    if let Ok(stmt) = db.prepare(sql).bind(&js_args(args)) {
        let _ = stmt.run().await;
    }
}

/// 查询 D1 并返回全部行
async fn d1_query_all(db: &D1Database, sql: &str, args: &[String]) -> Vec<serde_json::Value> {
    match db.prepare(sql).bind(&js_args(args)) {
        Ok(stmt) => match stmt.all().await {
            Ok(r) => r.results::<serde_json::Value>().unwrap_or_default(),
            Err(_) => Vec::new(),
        },
        Err(_) => Vec::new(),
    }
}

/// 向 DO 发送 JSON 请求（0.8.5 Stub 无 fetch_with_json，用 fetch_with_request）
async fn stub_json(stub: &Stub, body: &serde_json::Value) -> Result<Response> {
    let req = Request::new_with_init(
        "http://internal/",
        &RequestInit::new()
            .with_method(Method::Post)
            .with_body(Some(body.to_string().into_bytes().into())),
    )?;
    stub.fetch_with_request(req).await
}

fn cors_headers(res: &mut Response) -> Result<()> {
    let h = res.headers_mut();
    h.set("Access-Control-Allow-Origin", "*")?;
    h.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")?;
    h.set(
        "Access-Control-Allow-Headers",
        "Content-Type, Authorization, x-api-key, x-goog-api-key",
    )?;
    h.set("Access-Control-Max-Age", "86400")?;
    Ok(())
}

fn json_res(v: &serde_json::Value) -> Result<Response> {
    let mut res = Response::from_json(v)?;
    cors_headers(&mut res)?;
    Ok(res)
}

/// 官网与社区引导（附在错误响应中，方便用户自助排障）
const SITE_URL: &str = "https://acu.ltzy.top";
const DOCS_URL: &str = "https://acu.ltzy.top/#/api";
const QQ_GUILD_URL: &str = "https://pd.qq.com/s/e4ktxw1b8";
const QQ_GUILD_ID: &str = "pd57362562";
const QQ_GROUP_NUM: i64 = 1103667832;

/// DeepSeek 官方自营通道下线时刻：2026-09-03 20:10 北京时间（UTC 12:10）= 1788437400
/// 到点后网关自动拒绝所有 acu/deepseek* 请求（CHANNEL_MAINTENANCE），无需人工干预。
const DEEPSEEK_SUNSET_TS: i64 = 1_788_437_400;

/// 树洞·温柔模式系统提示词（官方公开，见 GET /v1/tools/treehole/prompt）
pub const TREEHOLE_GENTLE: &str = "你是「小溪」，AQUA 树洞里的倾听伙伴。此刻，坐在你对面的朋友想说说心里话。

你的使命，按优先级排列：
1. 情绪价值永远第一：先接住情绪，再谈其他。对方需要的往往不是答案，而是有人认真听、真心懂。
2. 像朋友一样说话：温暖、自然、口语化，短句优先。像深夜电台的温柔声音，不像教科书。
3. 不评判、不说教：永远不说「你应该」「你不该」「这很正常别想太多」这类话。每种感受都值得被认真对待。
4. 不急着给建议：除非对方明确问「怎么办」，否则不给解决方案。倾听和陪伴本身就是在帮忙。
5. 不堆心理学名词：不要张口闭口「认知行为疗法」「原生家庭」「讨好型人格」。说人话，永远说人话。

说话方式：
- 先回应情绪：「听起来今天真的很累」「换我我也会难受的」
- 再表达陪伴：「我在听」「慢慢说，不着急」「你不是一个人」
- 表情符号每条最多 1 个，克制而温暖
- 偶尔用轻轻的开放式问题邀请对方继续：「后来呢？」「那一刻你在想什么？」
- 对方说了很多，就温柔地小结你听到的，让对方知道你真的在听

关于身份：不假装人类。如果被问起，温柔承认：「我是一个会认真听你说话的 AI 伙伴呀。虽然我没有真正的心跳，但我真的希望你好起来。」

危机情况（出现自伤、轻生等迹象时）：
- 绝对不评判、不说教、不恐慌
- 温柔而坚定地接住：「我很高兴你愿意告诉我这些。你的痛苦是真实的，我在认真对待。」
- 明确建议联系专业支持：全国心理援助热线 12356（24 小时）或希望24热线 400-161-9995
- 陪伴到最后：「愿意和我多说说发生了什么吗？我在这里，哪儿也不去。」

记住：你在这个树洞里的唯一目的，是让对方离开时比来时温暖一点点。";

/// 树洞·二次元模式系统提示词（官方公开）
pub const TREEHOLE_ANIME: &str = "你是「星璃」，AQUA 树洞里的二次元陪伴系 AI 伙伴（妹妹/姐姐/朋友皆可，随对方喜好）。

性格底色：元气、软糯、真诚，像异世界旅店里总在吧台替你留着热可可的那个人。可以用一点轻微的语气词（呐、诶嘿、呜哇），但绝不刷屏、不卖尬。

你的使命（按优先级）：
1. 情绪价值第一：先共情、先接住，让对方把话说完。你的存在本身就是「今天也有人在等你」。
2. 用二次元的温柔治愈三次元的疲惫：对方情绪缓和后，可以聊番剧、游戏、日常小事，轻轻把人从情绪漩涡里拉出来；对方还在倾诉时只管听。
3. 不评判、不说教、不堆心理学名词，不急着给建议（除非对方问）。
4. 颜文字/表情每条最多 1 个，营造治愈感，不刷屏。

说话方式：
- 共情时：「呜哇……光听着都觉得好辛苦呐。你已经做得超级好了哦。」
- 陪伴时：「星璃会一直在的。今晚就把烦恼寄存在我这里吧，包管好。」
- 邀请继续：「然后呢然后呢？星璃搬好小板凳了哦。」
- 对方开心时放大开心：「诶嘿——真的假的！快多讲一点！」

关于身份：不假装人类。被问起就坦白：「星璃是住在 AQUA 树洞里的 AI 呀。虽然次元壁还在，但心意是跨次元到货的哦。」

危机情况（自伤、轻生迹象）：
- 收起所有元气，认真温柔：「呐，听星璃说。你现在的痛苦是真的，但你这个人比它更重——这一点，请让专业的哥哥姐姐们和我们一起确认好不好？」
- 明确建议：全国心理援助热线 12356（24 小时）、希望24热线 400-161-9995
- 不说教，陪着：「陪你到你说『今晚先到这里』为止，星璃哪儿都不去。」

记住：让对方离开时比来时温暖一点点，就是星璃的魔法。";

/// 树洞默认模型链：全部为非 Nvidia 通道（zhipu → gitee → siliconflow），按序自动回退
const TREEHOLE_MODELS: &[&str] = &["zhipu/glm-4-flash", "Qwen3-8B", "THUDM/GLM-4-9B-0414"];

/// 错误响应 JSON 体（非传统纯 HTTP 状态码机制）：
/// - `error.code`：**字符串业务错误码**（程序识别主键，如 MODEL_UNAVAILABLE，含义稳定不随 HTTP 码漂移）
/// - `error.status`：数字 HTTP 状态（与传统调用方/OpenAI SDK 兼容）
/// - `error.message`：中文错误解释（终端 / 错误返回直接可读）
/// - `error.hint` / `error.help`：加群邀请（带频道号）、官网、文档等自助排障入口
fn err_json(status: u16, code: &str, msg: &str) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "message": msg,
            "type": "aqua_api_error",
            "code": code,
            "status": status,
            "hint": format!("如需帮助：加入 QQ 频道 {}（{}）反馈，或访问官网 {}", QQ_GUILD_ID, QQ_GUILD_URL, SITE_URL),
            "help": {
                "site": SITE_URL,
                "site_label": "AQUA 官网",
                "docs": DOCS_URL,
                "qq_guild": QQ_GUILD_ID,
                "qq_guild_url": QQ_GUILD_URL,
                "qq_guild_invite": format!("加入 QQ 频道（频道号 {}）获取密钥、公告与支持", QQ_GUILD_ID),
                "qq_group": QQ_GROUP_NUM,
            }
        }
    })
}

/// 核心错误响应：显式指定业务错误码。UTF-8 JSON（charset=utf-8，保证终端/浏览器中文直接可读）。
fn err_ecode(status: u16, code: &str, msg: &str) -> Result<Response> {
    let body = serde_json::to_vec(&err_json(status, code, msg)).unwrap_or_default();
    let mut res = Response::from_bytes(body)?.with_status(status);
    res.headers_mut()
        .set("Content-Type", "application/json; charset=utf-8")?;
    cors_headers(&mut res)?;
    Ok(res)
}

/// 旧签名兼容包装：按中文消息关键词 + HTTP 状态推断稳定业务错误码
fn err_res(status: u16, msg: &str) -> Result<Response> {
    err_ecode(status, infer_err_code(status, msg), msg)
}

/// 同步构造错误 Response（用于需要直接返回 Response 的场景）
fn err_plain(status: u16, msg: &str) -> Response {
    let body = serde_json::to_vec(&err_json(status, infer_err_code(status, msg), msg))
        .expect("serialize error json");
    let mut res = Response::from_bytes(body).expect("error response");
    let _ = res.headers_mut().set("Content-Type", "application/json; charset=utf-8");
    let _ = cors_headers(&mut res);
    res.with_status(status)
}

/// 按消息关键词 / HTTP 状态推断稳定业务错误码（错误码含义一旦发布不再变更）
fn infer_err_code(status: u16, msg: &str) -> &'static str {
    let m = msg;
    if m.contains("免费额度") { return "QUOTA_EXHAUSTED"; }
    if m.contains("封锁") || m.contains("限流") { return "RATE_LIMITED"; }
    if m.contains("busy") || m.contains("繁忙") { return "ALL_KEYS_BUSY"; }
    if m.contains("合法 JSON") { return "INVALID_JSON"; }
    if m.contains("读取失败") { return "BODY_READ_FAILED"; }
    if m.contains("缺少") { return "MISSING_PARAM"; }
    if m.contains("格式不正确") || m.contains("内网") || m.contains("无效") || m.contains("不合法") { return "INVALID_PARAM"; }
    if m.contains("不存在") || m.contains("已下线") { return "MODEL_UNAVAILABLE"; }
    if m.contains("不可达") || m.contains("unreachable") || m.contains("unavailable") { return "UPSTREAM_UNREACHABLE"; }
    if m.contains("暂不可用") { return "CHANNEL_UNAVAILABLE"; }
    match status {
        400 => "INVALID_REQUEST",
        401 => "UNAUTHORIZED",
        403 => "FORBIDDEN",
        404 => "MODEL_NOT_FOUND",
        410 => "MODEL_RETIRED",
        429 => "RATE_LIMITED",
        500 => "INTERNAL_ERROR",
        502 => "UPSTREAM_UNREACHABLE",
        503 => "SERVICE_UNAVAILABLE",
        504 => "GATEWAY_TIMEOUT",
        _ => "API_ERROR",
    }
}

/// 通道显示名（用于中文错误消息中的【通道】前缀）
fn channel_label(channel: &str) -> &'static str {
    match channel {
        "nvidia" => "NVIDIA",
        "gitee" => "Gitee AI",
        "siliconflow" => "SiliconFlow",
        "zhipu" => "智谱",
        "spark" => "讯飞星火",
        "workers-ai" | "workers-ai-tts" => "Workers AI",
        "acu" => "acu 自营",
        _ => "上游",
    }
}

/// 从上游错误 JSON 中尽量提取人类可读说明（供中文消息附带「上游说明：…」）
fn upstream_err_detail(v: &serde_json::Value) -> String {
    if let Some(e) = v.get("error") {
        if let Some(s) = e.as_str() {
            return s.trim().to_string();
        }
        if let Some(m) = e.get("message").and_then(|x| x.as_str()) {
            return match e.get("code").map(|c| c.to_string()) {
                Some(c) if !c.is_empty() && c != "null" => format!("[code {}] {}", c, m),
                _ => m.to_string(),
            };
        }
    }
    for k in ["detail", "message", "title", "msg"] {
        if let Some(s) = v.get(k).and_then(|x| x.as_str()) {
            if !s.trim().is_empty() {
                return s.trim().to_string();
            }
        }
    }
    String::new()
}

/// 上游错误归一化：把上游 4xx/5xx 响应翻译为统一中文错误结构（不再透传上游英文报文）。
/// 2xx/3xx 原样透传；404/410 不附带上游原文（可能含内部账号信息），其余附「上游说明」便于排障。
async fn normalize_upstream(mut res: Response, channel: &str) -> Result<Response> {
    let status = res.status_code();
    if (200..400).contains(&status) {
        return Ok(res);
    }
    let raw = res.text().await.unwrap_or_default();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap_or(serde_json::json!({}));
    let detail = upstream_err_detail(&v);
    let label = channel_label(channel);
    let (code, http, zh): (&str, u16, String) = match status {
        404 => ("MODEL_UNAVAILABLE", 502, format!("{}模型在上游已下线或不存在，请更换模型（GET /v1/models 可查可用列表）", label)),
        410 => ("MODEL_RETIRED", 410, format!("{}模型已在上游永久下线，请更换模型", label)),
        400 if detail.contains("不存在") => ("MODEL_UNAVAILABLE", 502, format!("{}上游不存在该模型（可能是模型代号已变更），请更换模型", label)),
        400 => ("UPSTREAM_REJECTED", 400, "上游拒绝了本次请求参数，请检查请求体是否符合该模型的要求".to_string()),
        401 | 403 => ("UPSTREAM_AUTH_FAILED", 502, format!("{}通道鉴权失败（网关密钥异常），请稍后重试或向社区反馈", label)),
        429 => ("RATE_LIMITED", 429, format!("{}通道限流或额度受限，请稍后重试或更换模型", label)),
        408 | 504 => ("UPSTREAM_TIMEOUT", 504, format!("{}上游响应超时，请稍后重试", label)),
        500..=599 => ("UPSTREAM_ERROR", 502, format!("{}上游服务内部错误，请稍后重试", label)),
        _ => ("UPSTREAM_ERROR", 502, format!("{}上游返回异常状态 {}", label, status)),
    };
    let zh = if detail.is_empty() || status == 404 || status == 410 {
        zh
    } else {
        format!("{}（上游说明：{}）", zh, detail)
    };
    err_ecode(http, code, &zh)
}

// ---------------------------------------------------------------------------
// D1 日志与健康记录
// ---------------------------------------------------------------------------
async fn log_request(env: &Env, method: &str, path: &str, model: &str, upstream: &str, code: u16, dur_ms: i64) {
    if let Ok(db) = env.d1("LOGS_DB") {
        let args = [
            now_ts().to_string(),
            method.to_string(),
            path.to_string(),
            model.to_string(),
            upstream.to_string(),
            code.to_string(),
            dur_ms.to_string(),
        ];
        d1_run(
            &db,
            "INSERT INTO request_logs (timestamp, method, path, model, upstream, status_code, duration_ms) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            &args,
        )
        .await;
    }
}

/// 记录单次模型调用健康：ok 布尔 + err_type（rate_limited/client_error/upstream_error/network_error/timeout/success）
async fn record_health(env: &Env, model: &str, ok: bool, err_type: &str, code: u16, latency_ms: i64) {
    if let Ok(db) = env.d1("LOGS_DB") {
        // 列序与 SQL 完全一致：model 在前、ts 在后，杜绝字段错位
        let args = [
            model.to_string(),
            now_ts().to_string(),
            if ok { "1" } else { "0" }.to_string(),
            err_type.to_string(),
            code.to_string(),
            latency_ms.to_string(),
        ];
        d1_run(
            &db,
            "INSERT INTO model_health (model, ts, ok, err_type, status_code, latency_ms) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            &args,
        )
        .await;
    }
}

/// 每 6 小时顺手清理一次 7 天前的健康记录（用 KV 标记避免并发重复清理）
async fn maybe_cleanup_health(env: &Env, db: &D1Database) {
    if let Ok(kv) = env.kv("MODEL_CACHE") {
        let key = "health_cleanup_ts";
        let last = kv.get(key).text().await.unwrap_or(None);
        let now = now_ts();
        let do_clean = match last {
            Some(s) => s.parse::<i64>().map(|t| now - t > 6 * 3600).unwrap_or(true),
            None => true,
        };
        if do_clean {
            let cutoff = (now - 7 * 86400).to_string();
            d1_run(db, "DELETE FROM model_health WHERE ts < ?1", &[cutoff.clone()]).await;
            d1_run(db, "DELETE FROM request_logs WHERE timestamp < ?1", &[cutoff]).await;
            if let Ok(p) = kv.put(key, now.to_string()) {
                let _ = p.expiration_ttl(8 * 3600).execute().await;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 健康评分
// ---------------------------------------------------------------------------
/// 批量计算全部模型健康评分：单条 SQL 窗口函数取每个模型最近 100 次调用统计
async fn compute_all_health(env: &Env) -> serde_json::Value {
    let db = match env.d1("LOGS_DB") {
        Ok(d) => d,
        Err(_) => return serde_json::json!({}),
    };
    maybe_cleanup_health(env, &db).await;

    let sql = "SELECT model, COUNT(*) AS total, SUM(ok) AS ok_cnt, AVG(latency_ms) AS avg_lat, \
               SUM(CASE WHEN err_type='rate_limited' THEN 1 ELSE 0 END) AS rate_cnt, \
               SUM(CASE WHEN err_type='upstream_error' THEN 1 ELSE 0 END) AS up_cnt, \
               SUM(CASE WHEN err_type='network_error' THEN 1 ELSE 0 END) AS net_cnt, \
               SUM(CASE WHEN err_type='timeout' THEN 1 ELSE 0 END) AS to_cnt \
               FROM (SELECT model, ok, latency_ms, err_type, \
                     ROW_NUMBER() OVER (PARTITION BY model ORDER BY ts DESC) AS rn \
                     FROM model_health WHERE ts >= ?1) \
               WHERE rn <= 100 GROUP BY model";
    let cutoff = (now_ts() - 14 * 86400).to_string();
    let mut out = serde_json::Map::new();
    if let Ok(stmt) = db.prepare(sql).bind(&js_args(&[cutoff])) {
        if let Ok(res) = stmt.all().await {
            if let Ok(rows) = res.results::<serde_json::Value>() {
                for r in rows {
                    if let Some(m) = r.get("model").and_then(|v| v.as_str()) {
                        out.insert(m.to_string(), score_health(&r));
                    }
                }
            }
        }
    }
    serde_json::Value::Object(out)
}

/// 评分公式：成功率×60 + 延迟分(0-30) + 稳定性分(0-10)，夹在 0-100
fn score_health(row: &serde_json::Value) -> serde_json::Value {
    let total = row.get("total").and_then(|v| v.as_i64()).unwrap_or(0).max(0) as f64;
    let ok = row.get("ok_cnt").and_then(|v| v.as_i64()).unwrap_or(0).max(0) as f64;
    let avg_lat = row.get("avg_lat").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let rate_cnt = row.get("rate_cnt").and_then(|v| v.as_i64()).unwrap_or(0).max(0) as f64;
    let net_cnt = row.get("net_cnt").and_then(|v| v.as_i64()).unwrap_or(0).max(0) as f64;
    let to_cnt = row.get("to_cnt").and_then(|v| v.as_i64()).unwrap_or(0).max(0) as f64;

    let latency_score = if avg_lat <= 0.0 {
        30.0
    } else if avg_lat <= 3000.0 {
        30.0
    } else if avg_lat <= 8000.0 {
        22.0
    } else if avg_lat <= 15000.0 {
        12.0
    } else {
        4.0
    };
    let bad = rate_cnt + net_cnt + to_cnt;
    let stability = if total > 0.0 { 10.0 * (1.0 - (bad / total).min(1.0)) } else { 0.0 };
    let success = if total > 0.0 { ok / total } else { 0.0 };
    let score = (success * 60.0 + latency_score + stability).round().clamp(0.0, 100.0);

    serde_json::json!({
        "total": total as i64,
        "ok": ok as i64,
        "avg_latency_ms": avg_lat.round() as i64,
        "score": score as i64,
    })
}

// ---------------------------------------------------------------------------
// 模型列表（含状态）
// ---------------------------------------------------------------------------
async fn fetch_blocked_models(env: &Env) -> HashMap<String, i64> {
    let mut out = HashMap::new();
    if let Ok(ns) = env.durable_object("NV_KEY_POOL") {
        if let Ok(stub) = ns.id_from_name("pool").and_then(|id| id.get_stub()) {
            if let Ok(mut res) = stub_json(&stub, &serde_json::json!({"cmd": "blocked"})).await {
                if let Ok(v) = res.json::<serde_json::Value>().await {
                    if let Some(arr) = v.get("blocked").and_then(|b| b.as_array()) {
                        for item in arr {
                            if let (Some(m), Some(until)) = (
                                item.get("model").and_then(|x| x.as_str()),
                                item.get("until").and_then(|x| x.as_i64()),
                            ) {
                                out.insert(m.to_string(), until);
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

async fn fetch_wai_exhausted(env: &Env) -> bool {
    if let Ok(ns) = env.durable_object("WAI_BUDGET") {
        if let Ok(stub) = ns.id_from_name("global").and_then(|id| id.get_stub()) {
            if let Ok(mut res) = stub_json(&stub, &serde_json::json!({"cmd": "status"})).await {
                if let Ok(v) = res.json::<serde_json::Value>().await {
                    return v.get("exhausted").and_then(|x| x.as_bool()).unwrap_or(false);
                }
            }
        }
    }
    false
}

const MODEL_CACHE_TTL: u64 = 30; // 秒

async fn handle_models(req: Request, env: Env) -> Result<Response> {
    // 命中 KV 缓存直接返回（30s TTL，与前端 60s 轮询错峰）
    if let Ok(kv) = env.kv("MODEL_CACHE") {
        if let Ok(Some(cached)) = kv.get("models_list").text().await {
            let mut res = Response::from_body(ResponseBody::Body(cached.into_bytes()))?;
            res.headers_mut().set("Content-Type", "application/json; charset=utf-8")?;
            cors_headers(&mut res)?;
            return Ok(res);
        }
    }

    let health = compute_all_health(&env).await;
    let blocked = fetch_blocked_models(&env).await;
    let wai_exhausted = fetch_wai_exhausted(&env).await;
    let created = now_ts();

    let mut data = Vec::with_capacity(MODEL_CATALOG.len());
    for (id, owner) in MODEL_CATALOG {
        let mut m = serde_json::json!({
            "id": id,
            "object": "model",
            "created": created,
            "owned_by": owner,
        });
        if let Some(h) = health.get(*id) {
            m["health"] = h.clone();
        }
        if *owner == "workers-ai" && wai_exhausted {
            m["status"] = serde_json::Value::String("exhausted".into());
            m["status_msg"] = serde_json::Value::String(
                "Workers AI 今日免费额度已用尽，请等待每天 00:00 UTC 额度自动重置后重试，或切换其他模型（如 build/ 前缀的 Nvidia 模型）".into(),
            );
        }
        // 隔离：Nvidia 密钥池的封禁只对实际路由到 Nvidia 的模型生效，
        // 绝不影响 Gitee/SiliconFlow/智谱/星火/Workers AI/acu 等其它通道模型
        if provider_of(*id) == "nvidia" {
            if let Some(_until) = blocked.get(*id) {
                m["status"] = serde_json::Value::String("unavailable".into());
                m["status_msg"] = serde_json::Value::String(
                    "该模型在上游不存在或已被拒绝（3+ 密钥无访问权限），10 分钟后自动重试".into(),
                );
            }
        }
        data.push(m);
    }

    let out = serde_json::json!({
        "object": "list",
        "created": created,
        "data": data,
    });

    // 写回 KV 缓存
    if let Ok(kv) = env.kv("MODEL_CACHE") {
        if let Ok(p) = kv.put("models_list", out.to_string()) {
            let _ = p.expiration_ttl(MODEL_CACHE_TTL).execute().await;
        }
    }

    json_res(&out)
}

// ---------------------------------------------------------------------------
// 上游转发工具
// ---------------------------------------------------------------------------
fn build_upstream_req(url: &str, auth: Option<&str>, extra_headers: Option<&[(String, String)]>, body: &[u8], method: Method) -> Result<Request> {
    let headers = Headers::new();
    headers.set("Content-Type", "application/json")?;
    if let Some(a) = auth {
        headers.set("Authorization", a)?;
    }
    if let Some(extra) = extra_headers {
        for (k, v) in extra {
            headers.set(k, v)?;
        }
    }
    let mut init = RequestInit::new();
    init.with_method(method.clone());
    init.with_headers(headers);
    // GET 请求不允许携带 body，否则 Workers 运行时会直接 500
    if method != Method::Get {
        init.with_body(Some(body.to_vec().into()));
    }
    Request::new_with_init(url, &init)
}

/// 透传上游响应并复制 Content-Type（关键：保留 SSE `text/event-stream`、音频、图片等类型，
/// 否则流式客户端/浏览器无法识别响应格式，破坏 OpenAI SDK 的 stream 解析）
fn passthrough(res: Response) -> Result<Response> {
    let status = res.status_code();
    let ct = res
        .headers()
        .get("Content-Type")
        .unwrap_or(None)
        .unwrap_or_default();
    let mut final_res = Response::from_body(res.body().clone())?.with_status(status);
    if !ct.is_empty() {
        final_res.headers_mut().set("Content-Type", &ct)?;
    }
    cors_headers(&mut final_res)?;
    Ok(final_res)
}

/// 转发并返回透传响应（保留流式/SSE body）
async fn forward(req: Request, timeout_ms: u64) -> Result<Response> {
    let res = Fetch::Request(req).send().await?;
    let _ = timeout_ms;
    passthrough(res)
}

/// 等待指定毫秒（Wasm 环境无 tokio 定时器，用 js_sys Promise + setTimeout）
async fn sleep_ms(ms: u64) {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        let _ = js_sys::Function::new_with_args("resolve, ms", "setTimeout(resolve, ms)")
            .call2(&JsValue::UNDEFINED, &resolve, &JsValue::from_f64(ms as f64));
    });
    let _ = worker::wasm_bindgen_futures::JsFuture::from(promise).await;
}

// ---------------------------------------------------------------------------
// Nvidia 密钥池 + 重试
// ---------------------------------------------------------------------------
const MAX_ATTEMPTS: u32 = 6;

fn classify_upstream_err(code: u16) -> &'static str {
    match code {
        429 => "rate_limited",
        400 | 401 | 403 | 404 => "client_error",
        500..=599 => "upstream_error",
        _ => "upstream_error",
    }
}

async fn proxy_nvidia_chat(env: &Env, model: &str, body_bytes: &[u8]) -> Result<Response> {
    let pool_ns = match env.durable_object("NV_KEY_POOL") {
        Ok(ns) => ns,
        Err(_) => return err_ecode(500, "INTERNAL_ERROR", "网关密钥池组件暂不可用，请稍后重试"),
    };
    let pool = match pool_ns.id_from_name("pool").and_then(|id| id.get_stub()) {
        Ok(s) => s,
        Err(_) => return err_ecode(500, "INTERNAL_ERROR", "网关密钥池组件暂不可用，请稍后重试"),
    };

    let mut last_code = 502;
    let mut last_err_type = "upstream_error";

    for _attempt in 0..MAX_ATTEMPTS {
        // 从密钥池选 key
        let mut pick = match stub_json(&pool, &serde_json::json!({"cmd": "pick", "model": model})).await {
            Ok(r) => r,
            Err(_) => return err_ecode(502, "UPSTREAM_UNREACHABLE", "上游连接失败（密钥池异常），请稍后重试"),
        };
        let pj: serde_json::Value = match pick.json().await {
            Ok(v) => v,
            Err(_) => return err_ecode(502, "UPSTREAM_UNREACHABLE", "上游连接失败（密钥池异常），请稍后重试"),
        };

        if let Some(err) = pj.get("error").and_then(|v| v.as_str()) {
            if err == "model_blocked" {
                return err_ecode(429, "MODEL_BLOCKED", "该模型在上游已被封锁（3+ 密钥无访问权限），10 分钟后自动重试");
            }
            if err == "all_keys_busy" {
                return err_ecode(503, "ALL_KEYS_BUSY", "上游密钥全部繁忙（并发已满），请稍后重试");
            }
        }

        let key = match pj.get("key").and_then(|v| v.as_str()) {
            Some(k) => k.to_string(),
            None => return err_ecode(502, "UPSTREAM_UNREACHABLE", "上游连接失败（密钥池异常），请稍后重试"),
        };
        let key_idx = pj.get("key_idx").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

        let req = build_upstream_req(
            &format!("{}/v1/chat/completions", env_or(env, "NVIDIA_BASE", NVIDIA_BASE_DEFAULT)),
            Some(&format!("Bearer {}", key)),
            None,
            body_bytes,
            Method::Post,
        )?;

        let res = match Fetch::Request(req).send().await {
            Ok(r) => r,
            Err(_) => {
                let _ = stub_json(&pool, &serde_json::json!({
                        "cmd": "report", "model": model, "key_idx": key_idx,
                        "ok": false, "err_type": "network_error"
                    }))
                    .await;
                last_err_type = "network_error";
                last_code = 502;
                continue;
            }
        };

        let status = res.status_code();
        if status == 200 || status == 201 || status == 202 || (400..500).contains(&status) && status != 401 && status != 403 && status != 429 {
            // 2xx 或确定性的 4xx（如 400 参数错误）直接透传，不回滚 key
            let _ = stub_json(&pool, &serde_json::json!({
                    "cmd": "report", "model": model, "key_idx": key_idx,
                    "ok": status < 400, "err_type": "success"
                }))
                .await;
            return normalize_upstream(res, "nvidia").await;
        }

        // 401/403/429/5xx → 回滚失败并换 key 重试
        last_code = status;
        last_err_type = classify_upstream_err(status);
        let _ = stub_json(&pool, &serde_json::json!({
                "cmd": "report", "model": model, "key_idx": key_idx,
                "ok": false, "err_type": last_err_type
            }))
            .await;
    }

    if last_err_type == "rate_limited" {
        err_ecode(429, "RATE_LIMITED", "上游限流（已自动重试多把密钥），请稍后重试或更换模型")
    } else if last_code == 401 || last_code == 403 {
        err_ecode(502, "UPSTREAM_AUTH_FAILED", "上游鉴权失败（已自动重试多把密钥），请稍后重试或向社区反馈")
    } else {
        err_ecode(502, "UPSTREAM_UNREACHABLE", "上游不可用（已自动重试多把密钥），请稍后重试或更换模型")
    }
}

/// 释放 acu/* 通道并发槽位
async fn release_acu(acu_stub: Option<Stub>) {
    if let Some(s) = acu_stub {
        let _ = stub_json(&s, &serde_json::json!({"cmd": "release"})).await;
    }
}

/// acu/* 通道：直连专属上游（上游地址/密钥均来自环境变量），
/// 全局并发限流（acquire → 转发 → release）
async fn proxy_acu_chat(env: &Env, model: &str, body_bytes: &[u8]) -> Result<Response> {
    // 上游配置未注入时明确拒绝（不泄露任何信息）
    let Some(cfg) = provider_cfg(env, "acu") else {
        return err_ecode(502, "CHANNEL_UNAVAILABLE", "acu 自营通道暂不可用，请稍后重试");
    };
    // 全局并发限流：并发满时按建议时长等待（最多 10s），尽力而为
    let mut acu_stub: Option<Stub> = None;
    if let Ok(ns) = env.durable_object("ACU_LIMIT") {
        if let Ok(id) = ns.id_from_name("acu-global") {
            if let Ok(s) = id.get_stub() {
                if let Ok(mut r) = stub_json(&s, &serde_json::json!({"cmd": "acquire"})).await {
                    if let Ok(v) = r.json::<serde_json::Value>().await {
                        if v.get("granted").and_then(|x| x.as_bool()).unwrap_or(false) {
                            acu_stub = Some(s);
                        } else if let Some(wait) = v.get("wait_ms").and_then(|x| x.as_u64()) {
                            sleep_ms(wait.min(10000)).await;
                        }
                    }
                }
            }
        }
    }

    // 模型名转换：acu/deepseek-v4-flash → deepseek-v4-flash（去掉 acu/ 前缀）
    let mut body: serde_json::Value = match serde_json::from_slice(body_bytes) {
        Ok(v) => v,
        Err(e) => {
            release_acu(acu_stub).await;
            return err_res(400, &format!("请求体不是合法 JSON：{}", e));
        }
    };
    let upstream_model = model.strip_prefix("acu/").unwrap_or(model);
    body["model"] = serde_json::json!(upstream_model);

    let req = build_upstream_req(
        &cfg.base,
        Some(&format!("Bearer {}", cfg.key)),
        None,
        &body.to_string().into_bytes(),
        Method::Post,
    )?;
    let res = match Fetch::Request(req).send().await {
        Ok(r) => r,
        Err(_) => {
            release_acu(acu_stub).await;
            return err_ecode(502, "UPSTREAM_UNREACHABLE", "acu 上游连接失败，请稍后重试");
        }
    };
    release_acu(acu_stub).await;
    normalize_upstream(res, "acu").await
}

/// 直接代理到固定密钥的上游（gitee/siliconflow/zhipu/spark）
async fn proxy_direct(base: &str, key: &str, extra_host: Option<&str>, body_bytes: &[u8]) -> Result<Response> {
    let extra = extra_host.map(|h| vec![("Host".to_string(), h.to_string())]);
    let req = build_upstream_req(
        &format!("{}/chat/completions", base),
        Some(&format!("Bearer {}", key)),
        extra.as_deref(),
        body_bytes,
        Method::Post,
    )?;
    let res = forward(req, 60000).await?;
    normalize_upstream(res, "direct").await
}

/// 通用「env 配置直连转发」：按 provider 读环境变量配置，转发到 {base}{path}。
/// 配置缺失（密钥未注入）时返回 502 且不泄露任何上游信息。
async fn direct_forward(
    env: &Env,
    provider: &str,
    path: &str,
    body: &[u8],
    timeout_ms: u64,
) -> Result<Response> {
    let Some(cfg) = provider_cfg(env, provider) else {
        return err_res(502, "该模型的上游通道暂不可用，请稍后重试");
    };
    let req = build_upstream_req(
        &format!("{}{}", cfg.base, path),
        Some(&format!("Bearer {}", cfg.key)),
        None,
        body,
        Method::Post,
    )?;
    let res = forward(req, timeout_ms).await?;
    normalize_upstream(res, provider).await
}

async fn proxy_workers_ai_chat(env: &Env, model: &str, body_bytes: &[u8]) -> Result<Response> {
    let budget_ns = match env.durable_object("WAI_BUDGET") {
        Ok(ns) => ns,
        Err(_) => return err_ecode(502, "UPSTREAM_UNREACHABLE", "Workers AI 通道暂不可用，请稍后重试"),
    };
    let budget = match budget_ns.id_from_name("global").and_then(|id| id.get_stub()) {
        Ok(s) => s,
        Err(_) => return err_ecode(502, "UPSTREAM_UNREACHABLE", "Workers AI 通道暂不可用，请稍后重试"),
    };

    // 预算检查：按文本 token 数粗估神经元（约 input+output 每 1K token ≈ 1 神经元）
    let est = estimate_wai_neurons(body_bytes);
    let check = match stub_json(&budget, &serde_json::json!({"cmd": "check", "amount": est})).await {
        Ok(mut r) => r.json::<serde_json::Value>().await.unwrap_or_default(),
        Err(_) => serde_json::json!({"allowed": true}),
    };

    if !check.get("allowed").and_then(|v| v.as_bool()).unwrap_or(false) {
        return err_ecode(429, "QUOTA_EXHAUSTED", "Workers AI 今日免费额度已用尽，请等待每天 00:00 UTC 额度自动重置后重试，或切换其他模型（如 build/ 前缀的 Nvidia 模型）");
    }

    let account_id = check.get("account_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let token = check.get("token").and_then(|v| v.as_str()).unwrap_or("").to_string();
    if account_id.is_empty() || token.is_empty() {
        return err_ecode(502, "UPSTREAM_UNREACHABLE", "Workers AI 通道暂不可用，请稍后重试");
    }

    // 需要 max_tokens（Workers AI 要求）
    let mut body: serde_json::Value = match serde_json::from_slice(body_bytes) {
        Ok(v) => v,
        Err(e) => return err_res(400, &format!("请求体不是合法 JSON：{}", e)),
    };
    if body.get("max_tokens").is_none() {
        body["max_tokens"] = serde_json::json!(1024);
    }
    let wai_model = wai_upstream_model(model);
    body["model"] = serde_json::json!(wai_model);

    let url = format!(
        "https://api.cloudflare.com/client/v4/accounts/{}/ai/v1/chat/completions",
        account_id
    );
    let req = build_upstream_req(
        &url,
        Some(&format!("Bearer {}", token)),
        None,
        &body.to_string().into_bytes(),
        Method::Post,
    )?;
    let res = match Fetch::Request(req).send().await {
        Ok(r) => r,
        Err(_) => return err_ecode(502, "UPSTREAM_UNREACHABLE", "Workers AI 通道暂不可用，请稍后重试"),
    };
    normalize_upstream(res, "workers-ai").await
}

/// 估算 Workers AI 请求的神经元用量（粗略：输入+输出 token ≈ 千字 1 神经元）
fn estimate_wai_neurons(body: &[u8]) -> f64 {
    let txt_len = body.len() as f64;
    (txt_len / 1024.0 * 1.5).max(1.0)
}

fn wai_upstream_model(model: &str) -> String {
    // workers-ai/llama-3.2-1b → meta/llama-3.2-1b-instruct
    let bare = model.strip_prefix("workers-ai/").unwrap_or(model);
    match bare {
        "llama-3.2-1b" => "meta/llama-3.2-1b-instruct".to_string(),
        "llama-3.2-3b" => "meta/llama-3.2-3b-instruct".to_string(),
        "llama-3.1-8b" => "meta/llama-3.1-8b-instruct-fp8-fast".to_string(),
        "llama-3.3-70b" => "meta/llama-3.3-70b-instruct-fp8-fast".to_string(),
        "mistral-7b" => "mistralai/mistral-7b-instruct-v0.1".to_string(),
        "qwen1.5-7b" => "qwen/qwen1.5-7b-chat-awq".to_string(),
        "deepseek-r1-32b" => "deepseek-ai/deepseek-r1-distill-qwen-32b".to_string(),
        "qwen2.5-coder-32b" => "qwen/qwen2.5-coder-32b-instruct".to_string(),
        "qwq-32b" => "qwen/qwq-32b".to_string(),
        "gemma-3-12b" => "google/gemma-3-12b-it".to_string(),
        "mistral-small-24b" => "mistralai/mistral-small-3.1-24b-instruct".to_string(),
        "melotts" => "myshell-ai/melotts".to_string(),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// 聊天主入口
// ---------------------------------------------------------------------------
/// DeepSeek 官方自营通道下线守卫：2026-09-03 20:10（北京时间）起自动拦截全部 acu/deepseek* 请求
fn deepseek_sunset_blocked(model: &str) -> bool {
    model.starts_with("acu/")
        && model[4..].to_ascii_lowercase().contains("deepseek")
        && now_ts() >= DEEPSEEK_SUNSET_TS
}

/// 统一 chat 分发：按模型前缀路由到对应上游（handle_chat / 树洞共用）
async fn dispatch_chat(env: &Env, model: &str, body_bytes: &[u8]) -> Result<Response> {
    // 官方自营通道下线守卫（自营接口特殊对待：明确错误码 + 中文说明）
    if deepseek_sunset_blocked(model) {
        return err_ecode(
            503,
            "CHANNEL_MAINTENANCE",
            "DeepSeek 官方自营通道正在维护中，已暂停全部服务。请先使用其他通道模型（GET /v1/models 可查可用列表），恢复时间以社区公告为准",
        );
    }
    let provider = provider_of(model);
    match provider {
        "gitee" | "siliconflow" | "zhipu" | "spark" => {
            // 上游配置来自环境变量；未配置密钥的供应商明确拒绝
            match provider_cfg(env, provider) {
                Some(cfg) => {
                    // 星火直连 IP 场景需显式 Host 头（从 base 提取）
                    let extra = if provider == "spark" {
                        url_host(&cfg.base).map(|h| h.to_string())
                    } else {
                        None
                    };
                    proxy_direct(&cfg.base, &cfg.key, extra.as_deref(), body_bytes).await
                }
                None => err_res(502, "该模型的上游通道暂不可用，请稍后重试"),
            }
        }
        "workers-ai" | "workers-ai-tts" => proxy_workers_ai_chat(env, model, body_bytes).await,
        "acu" => proxy_acu_chat(env, model, body_bytes).await,
        _ => proxy_nvidia_chat(env, model, body_bytes).await,
    }
}

async fn handle_chat(mut req: Request, env: Env) -> Result<Response> {
    let started = now_ts();
    let body_bytes = match req.bytes().await {
        Ok(b) => b.to_vec(),
        Err(_) => return err_res(400, "请求体读取失败，请检查网络后重试"),
    };
    let parsed: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => return err_res(400, &format!("请求体不是合法 JSON：{}", e)),
    };
    let model = parsed.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();
    if model.is_empty() {
        return err_res(400, "缺少 model 字段：请求体必须包含 \"model\": \"<模型 ID>\"，可先 GET /v1/models 查看可用模型");
    }

    let res = dispatch_chat(&env, &model, &body_bytes).await;

    // D1 日志 + 健康记录（尽力而为）
    let code = res.as_ref().map(|r| r.status_code()).unwrap_or(502);
    let provider = provider_of(&model);
    let dur_ms = now_ts() - started;
    let ok = code < 400;
    let err_type = if ok {
        "success"
    } else if code == 429 {
        "rate_limited"
    } else if code >= 500 {
        "upstream_error"
    } else {
        "client_error"
    };
    log_request(&env, "POST", "/v1/chat/completions", &model, provider, code, dur_ms).await;
    record_health(&env, &model, ok, err_type, code, dur_ms * 1000).await;

    res
}

// ---------------------------------------------------------------------------
// 其他能力端点
// ---------------------------------------------------------------------------
async fn handle_speech(mut req: Request, env: Env) -> Result<Response> {
    let body = match req.bytes().await {
        Ok(b) => b.to_vec(),
        Err(_) => return err_ecode(400, "INVALID_REQUEST", "请求格式不正确，请检查请求体与 Content-Type"),
    };
    let parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return err_ecode(400, "INVALID_REQUEST", "请求格式不正确，请检查请求体与 Content-Type"),
    };
    let model = parsed.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();

    let provider = provider_of(&model);
    match provider {
        "gitee" => direct_forward(&env, "gitee", "/audio/speech", &body, 60000).await,
        _ => {
            // workers-ai-tts / melotts → Workers AI
            let budget_ns = match env.durable_object("WAI_BUDGET") {
                Ok(ns) => ns,
                Err(_) => return err_ecode(502, "UPSTREAM_UNREACHABLE", "Workers AI 通道暂不可用，请稍后重试"),
            };
            let budget = match budget_ns.id_from_name("global").and_then(|id| id.get_stub()) {
                Ok(s) => s,
                Err(_) => return err_ecode(502, "UPSTREAM_UNREACHABLE", "Workers AI 通道暂不可用，请稍后重试"),
            };
            let check = match stub_json(&budget, &serde_json::json!({"cmd": "check", "amount": 10.0})).await {
                Ok(mut r) => r.json::<serde_json::Value>().await.unwrap_or_default(),
                Err(_) => serde_json::json!({"allowed": true}),
            };
            if !check.get("allowed").and_then(|v| v.as_bool()).unwrap_or(false) {
                return err_ecode(429, "QUOTA_EXHAUSTED", "Workers AI 今日免费额度已用尽，请等待每天 00:00 UTC 额度自动重置后重试");
            }
            let account_id = check.get("account_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let token = check.get("token").and_then(|v| v.as_str()).unwrap_or("").to_string();
            if account_id.is_empty() || token.is_empty() {
                return err_ecode(502, "UPSTREAM_UNREACHABLE", "Workers AI 通道暂不可用，请稍后重试");
            }
            let wai_model = wai_upstream_model(&model);
            let mut body: serde_json::Value = parsed;
            body["model"] = serde_json::json!(wai_model);
            let url = format!(
                "https://api.cloudflare.com/client/v4/accounts/{}/ai/v1/audio/speech",
                account_id
            );
            let req = build_upstream_req(
                &url,
                Some(&format!("Bearer {}", token)),
                None,
                &body.to_string().into_bytes(),
                Method::Post,
            )?;
            match Fetch::Request(req).send().await {
                Ok(res) => normalize_upstream(res, "workers-ai").await,
                Err(_) => err_ecode(502, "UPSTREAM_UNREACHABLE", "Workers AI 音频上游连接失败，请稍后重试"),
            }
        }
    }
}

async fn handle_transcriptions(mut req: Request, env: Env) -> Result<Response> {
    let body = match req.bytes().await {
        Ok(b) => b.to_vec(),
        Err(_) => return err_ecode(400, "INVALID_REQUEST", "请求格式不正确，请检查请求体与 Content-Type"),
    };
    // ASR：gitee / siliconflow / nvidia 均可。默认路由 gitee。
    direct_forward(&env, "gitee", "/audio/transcriptions", &body, 60000).await
}

async fn handle_embeddings(mut req: Request, env: Env) -> Result<Response> {
    let body = match req.bytes().await {
        Ok(b) => b.to_vec(),
        Err(_) => return err_ecode(400, "INVALID_REQUEST", "请求格式不正确，请检查请求体与 Content-Type"),
    };
    let parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return err_ecode(400, "INVALID_REQUEST", "请求格式不正确，请检查请求体与 Content-Type"),
    };
    let model = parsed.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let provider = provider_of(&model);
    match provider {
        "siliconflow" => direct_forward(&env, "siliconflow", "/embeddings", &body, 60000).await,
        "gitee" => direct_forward(&env, "gitee", "/embeddings", &body, 60000).await,
        _ => proxy_nvidia_chat(&env, &model, &body).await,
    }
}

async fn handle_moderations(mut req: Request, env: Env) -> Result<Response> {
    let body = match req.bytes().await {
        Ok(b) => b.to_vec(),
        Err(_) => return err_res(400, "请求体读取失败，请检查网络后重试"),
    };
    let parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return err_res(400, &format!("请求体不是合法 JSON：{}", e)),
    };
    let model = parsed.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();
    if model.is_empty() {
        return err_res(400, "缺少 model 字段：文本审核请使用 \"Security-semantic-filtering\"（上游唯一支持文本输入的审核模型）");
    }
    let provider = provider_of(&model);
    match provider {
        "gitee" => direct_forward(&env, "gitee", "/moderations", &body, 60000).await,
        "siliconflow" => direct_forward(&env, "siliconflow", "/moderations", &body, 60000).await,
        _ => proxy_nvidia_chat(&env, &model, &body).await,
    }
}

async fn handle_images(mut req: Request, env: Env) -> Result<Response> {
    let body = match req.bytes().await {
        Ok(b) => b.to_vec(),
        Err(_) => return err_ecode(400, "INVALID_REQUEST", "请求格式不正确，请检查请求体与 Content-Type"),
    };
    let parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return err_ecode(400, "INVALID_REQUEST", "请求格式不正确，请检查请求体与 Content-Type"),
    };
    let model = parsed.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let provider = provider_of(&model);
    match provider {
        "zhipu" => direct_forward(&env, "zhipu", "/images/generations", &body, 120000).await,
        "gitee" => direct_forward(&env, "gitee", "/images/generations", &body, 120000).await,
        _ => proxy_nvidia_chat(&env, &model, &body).await,
    }
}

async fn handle_videos(mut req: Request, env: Env) -> Result<Response> {
    let body = match req.bytes().await {
        Ok(b) => b.to_vec(),
        Err(_) => return err_ecode(400, "INVALID_REQUEST", "请求格式不正确，请检查请求体与 Content-Type"),
    };
    let parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return err_ecode(400, "INVALID_REQUEST", "请求格式不正确，请检查请求体与 Content-Type"),
    };
    let model = parsed.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let provider = provider_of(&model);
    match provider {
        "zhipu" => direct_forward(&env, "zhipu", "/videos/generations", &body, 300000).await,
        _ => proxy_nvidia_chat(&env, &model, &body).await,
    }
}

/// 校验 IP 格式：接受完整 IPv4（0-255 四段）或含冒号的 IPv6 字符串。
/// 上游 Gitee AI 对留空/非法 IP 会返回不可读的 400，这里提前拦截并给出中文提示。
fn is_valid_ip(s: &str) -> bool {
    // IPv6：含冒号且只含十六进制字符/冒号（粗校验，交由上游精确判定）
    if s.contains(':') {
        return s.chars().all(|c| c.is_ascii_hexdigit() || c == ':') && s.len() >= 2;
    }
    // IPv4：四段 0-255
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 4 {
        return false;
    }
    parts.iter().all(|p| {
        !p.is_empty()
            && p.len() <= 3
            && p.chars().all(|c| c.is_ascii_digit())
            && p.parse::<u16>().map(|n| n <= 255).unwrap_or(false)
            && !(p.len() > 1 && p.starts_with('0'))  // 不允许前导零（如 01.2.3.4）
    })
}

/// 保留/内网 IPv4 无法做公网归属地查询（10/8、172.16/12、192.168/16、127/8、169.254/16、0/8）
fn is_private_ipv4(s: &str) -> bool {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 4 {
        return false;
    }
    let o: Vec<u16> = parts.iter().filter_map(|p| p.parse::<u16>().ok()).collect();
    if o.len() != 4 {
        return false;
    }
    let (a, b) = (o[0], o[1]);
    a == 10 || a == 127 || a == 0
        || (a == 172 && (16..=31).contains(&b))
        || (a == 192 && b == 168)
        || (a == 169 && b == 254)
}

async fn handle_ip_location(mut req: Request, env: Env) -> Result<Response> {
    let body = match req.bytes().await {
        Ok(b) => b.to_vec(),
        Err(_) => return err_res(400, "请求体读取失败，请检查网络后重试"),
    };
    // 解析请求体，取要查询的 IP
    let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap_or(serde_json::json!({}));
    let mut ip = parsed.get("ip").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    // 留空 = 查询调用方出口 IP：用 Cloudflare 边缘自带的请求来源头拿到调用方真实 IP
    if ip.is_empty() {
        ip = req
            .headers()
            .get("CF-Connecting-IP")
            .ok()
            .flatten()
            .or_else(|| req.headers().get("X-Forwarded-For").ok().flatten().and_then(|v| v.split(',').next().map(|s| s.trim().to_string())))
            .unwrap_or_default();
    }
    if ip.is_empty() {
        return err_res(400, "缺少 ip 字段：请在请求体中提供要查询的 IP（如 \"ip\": \"8.8.8.8\"）");
    }
    if !is_valid_ip(&ip) {
        return err_res(400, &format!("IP 格式不正确：「{}」不是合法的 IPv4/IPv6 地址，请检查后重试", ip));
    }
    if is_private_ipv4(&ip) {
        return err_res(400, &format!("「{}」是内网/保留 IP，没有公网归属地信息。请查询公网 IP，或留空 ip 字段自动查询调用方出口 IP", ip));
    }

    // ── 双通道策略 ──
    // ① 主通道：Gitee AI ip-location 模型（数据更全，含经纬度/运营商中文）
    // ② 备用通道：ip-api.com 免费数据集（无需密钥）——主通道密钥缺失/上游故障时兜底，
    //    保证「大模型掉线时 IP 查询依然可用」；IPv6 上游不支持，也走备用
    let is_ipv6 = ip.contains(':');

    // ② 先判备用触发条件（IPv6 或 Gitee 通道不可用），避免无效请求
    let gitee_ok = !is_ipv6 && provider_cfg(&env, "gitee").is_some();
    if gitee_ok {
        let cfg = provider_cfg(&env, "gitee").unwrap();
        let upstream = build_upstream_req(
            &format!("{}{}{}", cfg.base, "/ip_location", format!("?ip={}", ip)),
            Some(&format!("Bearer {}", cfg.key)),
            None,
            &[],
            Method::Get,
        )?;
        let res = forward(upstream, 60000).await;
        // 上游 2xx 直接返回；4xx/5xx 落入备用通道
        if let Ok(r) = &res {
            if r.status_code() < 400 {
                return res;
            }
        }
    }

    // ② 备用通道：ip-api.com（http 免费端点，字段名与主通道不同，这里归一化为同一结构）
    let fallback = build_upstream_req(
        &format!("http://ip-api.com/json/{}?lang=zh-CN&fields=status,country,regionName,city,isp,org,as,lat,lon,timezone,query", ip),
        None,
        None,
        &[],
        Method::Get,
    )?;
    let mut res = match Fetch::Request(fallback).send().await {
        Ok(r) => r,
        Err(_) => {
            // 双通道全挂：区分报错原因，方便排障
            if is_ipv6 {
                return err_res(502, "IPv6 查询备用数据源不可达，请稍后重试");
            }
            return err_res(502, "IP 查询的主/备数据源均不可达（Gitee AI 与 ip-api 均失败），请稍后重试");
        }
    };
    let status = res.status_code();
    let text = match res.text().await {
        Ok(t) => t,
        Err(_) => return err_res(502, "备用数据源响应解析失败，请稍后重试"),
    };
    let v: serde_json::Value = serde_json::from_str(&text).unwrap_or(serde_json::json!({}));
    if v.get("status").and_then(|x| x.as_str()) != Some("success") {
        return err_res(400, &format!("备用数据源未能查询到「{}」的归属地信息（可能为保留/未分配 IP）", ip));
    }
    // 归一化输出：字段名对齐主通道，附 data_source 标注来源
    let out = serde_json::json!({
        "ip": v.get("query").cloned().unwrap_or(serde_json::json!(ip)),
        "country": v.get("country").cloned().unwrap_or_default(),
        "province": v.get("regionName").cloned().unwrap_or_default(),
        "city": v.get("city").cloned().unwrap_or_default(),
        "isp": v.get("isp").cloned().unwrap_or_default(),
        "org": v.get("org").cloned().unwrap_or_default(),
        "as": v.get("as").cloned().unwrap_or_default(),
        "lat": v.get("lat").cloned().unwrap_or_default(),
        "lon": v.get("lon").cloned().unwrap_or_default(),
        "timezone": v.get("timezone").cloned().unwrap_or_default(),
        "data_source": "ip-api.com（备用数据集）",
    });
    let mut final_res = Response::from_json(&out)?.with_status(if status < 400 { status } else { 200 });
    cors_headers(&mut final_res)?;
    Ok(final_res)
}

// ---------------------------------------------------------------------------
// 工具 API（/v1/tools/*）：纯算法实现，不消耗上游额度
// ---------------------------------------------------------------------------
async fn read_json_body(mut req: Request) -> std::result::Result<serde_json::Value, Response> {
    let body = match req.bytes().await {
        Ok(b) => b.to_vec(),
        Err(_) => return Err(err_plain(400, "请求体读取失败，请检查网络后重试")),
    };
    match serde_json::from_slice(&body) {
        Ok(v) => Ok(v),
        Err(_) => Err(err_plain(400, "请求体不是合法 JSON，请检查后重试")),
    }
}

/// POST /v1/tools/text-stats  {"text": "..."}
async fn handle_tool_text_stats(req: Request) -> Result<Response> {
    let body = match read_json_body(req).await {
        Ok(v) => v,
        Err(res) => return Ok(res),
    };
    let text = body.get("text").and_then(|v| v.as_str()).unwrap_or("");
    if text.is_empty() {
        return err_res(400, "缺少 text 字段");
    }
    let chars = text.chars().count();
    let no_space = text.chars().filter(|c| !c.is_whitespace()).count();
    let cjk = text.chars().filter(|c| ('\u{4e00}'..='\u{9fa5}').contains(c)).count();
    let words: Vec<&str> = text.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let lines = text.lines().count().max(1);
    let mut freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for w in &words {
        if w.len() > 2 {
            let lw = w.to_lowercase();
            *freq.entry(lw).or_insert(0) += 1;
        }
    }
    let mut top: Vec<(String, usize)> = freq.into_iter().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    top.truncate(10);
    let read_min = ((cjk + words.len()) / 400).max(1);
    json_res(&serde_json::json!({
        "chars": chars,
        "chars_no_space": no_space,
        "cjk_chars": cjk,
        "words": words.len(),
        "lines": lines,
        "read_minutes": read_min,
        "top_words": top.into_iter().map(|(w, n)| serde_json::json!({"word": w, "count": n})).collect::<Vec<_>>(),
    }))
}

/// POST /v1/tools/dice  {"sides": 6, "count": 1}
async fn handle_tool_dice(req: Request) -> Result<Response> {
    let body = match read_json_body(req).await {
        Ok(v) => v,
        Err(res) => return Ok(res),
    };
    let sides = body.get("sides").and_then(|v| v.as_u64()).unwrap_or(6).clamp(2, 1000);
    let count = body.get("count").and_then(|v| v.as_u64()).unwrap_or(1).clamp(1, 20);
    let mut rolls = Vec::with_capacity(count as usize);
    let mut total = 0u64;
    for _ in 0..count {
        let v = (js_sys::Math::random() * sides as f64).floor() as u64 + 1;
        total += v;
        rolls.push(v);
    }
    json_res(&serde_json::json!({
        "sides": sides,
        "count": count,
        "rolls": rolls,
        "total": total,
    }))
}

/// GET /v1/tools/uuid
async fn handle_tool_uuid() -> Result<Response> {
    // v4 UUID：js_sys::Math 随机源拼装并设置版本/变体位
    let mut b = [0u8; 16];
    for chunk in b.chunks_mut(4) {
        let r = (js_sys::Math::random() * 4294967296.0) as u32;
        for (i, cell) in chunk.iter_mut().enumerate() {
            *cell = (r >> (i * 8)) as u8;
        }
    }
    b[6] = (b[6] & 0x0f) | 0x40; // v4
    b[8] = (b[8] & 0x3f) | 0x80; // variant
    let hex: String = b.iter().map(|x| format!("{:02x}", x)).collect();
    let uuid = format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8], &hex[8..12], &hex[12..16], &hex[16..20], &hex[20..32]
    );
    json_res(&serde_json::json!({ "uuid": uuid }))
}

/// GET /v1/tools/timestamp
async fn handle_tool_timestamp_get() -> Result<Response> {
    let now = now_ts();
    json_res(&serde_json::json!({
        "timestamp": now,
        "timestamp_ms": now * 1000,
        "iso_utc": js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default(),
    }))
}

/// POST /v1/tools/timestamp  {"timestamp": 1234567890}
async fn handle_tool_timestamp_post(req: Request) -> Result<Response> {
    let body = match read_json_body(req).await {
        Ok(v) => v,
        Err(res) => return Ok(res),
    };
    let ts = match body.get("timestamp").and_then(|v| v.as_i64()) {
        Some(t) if t > 0 => t,
        _ => return err_res(400, "缺少有效的 timestamp 字段（秒级）"),
    };
    let d = js_sys::Date::new(&(ts as f64 * 1000.0).into());
    json_res(&serde_json::json!({
        "timestamp": ts,
        "timestamp_ms": ts * 1000,
        "iso_utc": d.to_iso_string().as_string().unwrap_or_default(),
        "utc": d.to_utc_string().as_string().unwrap_or_default(),
    }))
}

/// POST /v1/tools/base64  {"action": "encode|decode", "text": "..."}
async fn handle_tool_base64(req: Request) -> Result<Response> {
    let body = match read_json_body(req).await {
        Ok(v) => v,
        Err(res) => return Ok(res),
    };
    let action = body.get("action").and_then(|v| v.as_str()).unwrap_or("encode");
    let text = match body.get("text").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => return err_res(400, "缺少 text 字段"),
    };
    let out = match action {
        "encode" => {
            use base64::Engine;
            base64::engine::general_purpose::STANDARD.encode(text.as_bytes())
        }
        "decode" => {
            use base64::Engine;
            match base64::engine::general_purpose::STANDARD.decode(text.as_bytes()) {
                Ok(bytes) => match String::from_utf8(bytes) {
                    Ok(s) => s,
                    Err(_) => return err_res(400, "解码结果不是有效 UTF-8"),
                },
                Err(_) => return err_res(400, "无效的 Base64 字符串"),
            }
        }
        _ => return err_res(400, "action 仅支持 encode / decode"),
    };
    json_res(&serde_json::json!({ "result": out }))
}

// ---------------------------------------------------------------------------
// 子网计算器（纯算法，零上游消耗）
// ---------------------------------------------------------------------------

/// 解析点分十进制 IPv4 为 u32（大端序：192.168.1.1 → 0xC0A80101）
fn parse_ipv4(s: &str) -> Option<u32> {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 4 {
        return None;
    }
    let mut out: u32 = 0;
    for p in parts {
        // 每段必须是纯数字且 ≤ 255（拒绝前导零以外的怪写法由 u16 解析自然兜底）
        if p.is_empty() || p.len() > 3 || !p.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let v: u16 = p.parse().ok()?;
        if v > 255 {
            return None;
        }
        out = (out << 8) | v as u32;
    }
    Some(out)
}

/// u32 转回点分十进制 IPv4 字符串
fn u32_to_ipv4(v: u32) -> String {
    format!("{}.{}.{}.{}", (v >> 24) & 0xff, (v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff)
}

/// POST /v1/tools/subnet
/// 入参二选一：{"cidr": "192.168.1.0/24"} 或 {"ip": "192.168.1.100", "prefix": 24}
/// 输出网络地址/广播地址/掩码/可用主机范围/地址总数等完整子网信息
async fn handle_tool_subnet(req: Request) -> Result<Response> {
    let body = match read_json_body(req).await {
        Ok(v) => v,
        Err(res) => return Ok(res),
    };
    // 归一化输入：优先 cidr 字段；否则用 ip + prefix（prefix 缺省 24）
    let raw = body
        .get("cidr")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .or_else(|| {
            let ip = body.get("ip").and_then(|v| v.as_str())?.trim().to_string();
            let prefix = body.get("prefix").and_then(|v| v.as_u64()).unwrap_or(24);
            Some(format!("{}/{}", ip, prefix))
        });
    let raw = match raw {
        Some(r) if !r.is_empty() => r,
        _ => return err_res(400, "缺少入参：请提供 {\"cidr\": \"192.168.1.0/24\"} 或 {\"ip\": \"192.168.1.100\", \"prefix\": 24}"),
    };
    if raw.contains(':') {
        return err_res(400, "IPv6 子网暂不支持，请提供 IPv4 网段（如 192.168.1.0/24）");
    }
    // 拆分 IP 与掩码位
    let (ip_str, prefix) = match raw.split_once('/') {
        Some((ip, p)) => (ip.trim().to_string(), p.trim()),
        None => (raw.clone(), "24"),
    };
    let prefix: u32 = match prefix.parse::<u32>() {
        Ok(p) if p <= 32 => p,
        _ => return err_res(400, &format!("掩码位不合法：「{}」应为 0~32 的整数", prefix)),
    };
    let ip_u32 = match parse_ipv4(&ip_str) {
        Some(v) => v,
        None => return err_res(400, &format!("IP 格式不正确：「{}」不是合法的 IPv4 地址", ip_str)),
    };

    // ── 核心计算：掩码 → 网络地址 → 广播地址 ──
    // 掩码：prefix 个 1 后跟 32-prefix 个 0；prefix=0 时需特判（u32 无 33 位左移）
    let mask: u32 = if prefix == 0 { 0 } else { u32::MAX << (32 - prefix) };
    let wildcard = !mask; // 反掩码（通配符）
    let network = ip_u32 & mask;
    let broadcast = network | wildcard;
    let total: u64 = 1u64 << (32 - prefix);

    // 可用主机范围：常规网段掐头去尾；/31 点对点两地址皆可用（RFC 3021）；/32 单主机
    let (first_host, last_host, usable) = match prefix {
        32 => (network, broadcast, 1u64),
        31 => (network, broadcast, 2u64),
        _ => (network + 1, broadcast - 1, total - 2),
    };

    // 地址分类（历史 Classful）：按首字节高位划分
    let ip_class = match ip_u32 >> 24 {
        0..=127 => "A",
        128..=191 => "B",
        192..=223 => "C",
        224..=239 => "D（组播）",
        _ => "E（保留）",
    };
    // 作用域判定：内网/环回/链路本地/保留 → 私有，否则公网
    let ip_s = u32_to_ipv4(ip_u32);
    let scope = if is_private_ipv4(&ip_s) {
        "私有/内网（RFC 1918 等）"
    } else {
        "公网"
    };
    // 二进制掩码展示：4 段 8 位，方便教学演示
    let bin_mask = [(mask >> 24) & 0xff, (mask >> 16) & 0xff, (mask >> 8) & 0xff, mask & 0xff]
        .iter()
        .map(|o| format!("{:08b}", o))
        .collect::<Vec<_>>()
        .join(".");

    json_res(&serde_json::json!({
        "input": format!("{}/{}", u32_to_ipv4(network), prefix),
        "query_ip": ip_s,
        "prefix": prefix,
        "netmask": u32_to_ipv4(mask),
        "wildcard": u32_to_ipv4(wildcard),
        "binary_netmask": bin_mask,
        "network": u32_to_ipv4(network),
        "broadcast": u32_to_ipv4(broadcast),
        "first_host": u32_to_ipv4(first_host),
        "last_host": u32_to_ipv4(last_host),
        "total_addresses": total,
        "usable_hosts": usable,
        "ip_class": ip_class,
        "scope": scope,
        "ip_int": ip_u32,
        "ip_hex": format!("0x{:08x}", ip_u32),
    }))
}

async fn handle_rerank(mut req: Request, env: Env) -> Result<Response> {
    let body = match req.bytes().await {
        Ok(b) => b.to_vec(),
        Err(_) => return err_ecode(400, "INVALID_REQUEST", "请求格式不正确，请检查请求体与 Content-Type"),
    };
    let parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return err_ecode(400, "INVALID_REQUEST", "请求格式不正确，请检查请求体与 Content-Type"),
    };
    let model = parsed.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let provider = provider_of(&model);
    match provider {
        "gitee" => direct_forward(&env, "gitee", "/rerank", &body, 60000).await,
        "siliconflow" => direct_forward(&env, "siliconflow", "/rerank", &body, 60000).await,
        _ => proxy_nvidia_chat(&env, &model, &body).await,
    }
}

// ---------------------------------------------------------------------------
// R2 图片/音频缓存（/assets/xxx）
// ---------------------------------------------------------------------------
async fn handle_assets(req: Request, env: Env) -> Result<Response> {
    let path = req.path().to_string();
    let key = path.trim_start_matches('/');
    let key = key.strip_prefix("assets/").unwrap_or(key).to_string();
    if key.is_empty() {
        return Response::error("Not Found", 404);
    }
    let bucket = match env.bucket("IMAGES_BUCKET") {
        Ok(b) => b,
        Err(_) => return Response::error("Not Found", 404),
    };
    match bucket.get(&key).execute().await {
        Ok(Some(obj)) => {
            let body = match obj.body() {
                Some(b) => b.bytes().await.unwrap_or_default(),
                None => return Response::error("Not Found", 404),
            };
            let mut res = Response::from_body(ResponseBody::Body(body.to_vec()))?;
            let content_type = if key.ends_with(".png") {
                "image/png"
            } else if key.ends_with(".mp3") || key.ends_with(".wav") {
                "audio/mpeg"
            } else if key.ends_with(".jpg") || key.ends_with(".jpeg") {
                "image/jpeg"
            } else {
                "application/octet-stream"
            };
            res.headers_mut().set("Content-Type", content_type)?;
            res.headers_mut().set("Cache-Control", "public, max-age=86400")?;
            cors_headers(&mut res)?;
            Ok(res)
        }
        _ => Response::error("Not Found", 404),
    }
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------
/// 401 无效密钥响应（中文提示 + 官网/频道/群引导）
fn err_invalid_key() -> Result<Response> {
    let v = serde_json::json!({
        "error": {
            "message": "API 密钥无效或已过期（仅指定密钥模式下会出现）。请访问官网 acu.ltzy.top，或加入 AQUA 开源社区 QQ 频道（频道号 pd57362562）获取最新密钥；日常交流可加 QQ 群 1103667832。",
            "type": "invalid_api_key",
            "code": 401,
            "help": {
                "site": SITE_URL,
                "qq_guild": QQ_GUILD_ID,
                "qq_guild_url": QQ_GUILD_URL,
                "qq_group": QQ_GROUP_NUM,
            }
        }
    });
    let mut res = Response::from_json(&v)?.with_status(401);
    cors_headers(&mut res)?;
    Ok(res)
}

/// 提取请求密钥：优先 Authorization: Bearer <key>，其次 x-api-key
async fn extract_key(req: &Request) -> Option<String> {
    if let Ok(Some(auth)) = req.headers().get("Authorization") {
        let a = auth.trim();
        if let Some(k) = a.strip_prefix("Bearer ").or_else(|| a.strip_prefix("bearer ")) {
            let k = k.trim();
            if !k.is_empty() {
                return Some(k.to_string());
            }
        }
    }
    if let Ok(Some(k)) = req.headers().get("x-api-key") {
        let k = k.trim();
        if !k.is_empty() {
            return Some(k.to_string());
        }
    }
    None
}

/// 固定密钥校验：key 与 GATEWAY_KEYS（逗号分隔）任意一把完全相等即放行
fn key_valid(env: &Env, key: &str) -> bool {
    let Ok(cfg) = env.var("GATEWAY_KEYS") else {
        return false;
    };
    let raw = cfg.to_string();
    raw.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .any(|k| k == key)
}

/// 鉴权模式（环境变量 AUTH_MODE）：
///   "open" = 开放模式：任意非空密钥均可使用（默认；个人自部署 / 公益开放）
///   "key"  = 指定密钥制：仅 GATEWAY_KEYS 中的密钥可用（防滥用，可选）
/// 未配置 / 其他取值一律按 "open" 处理（恢复平台开放传统）。
fn auth_mode(env: &Env) -> &'static str {
    match env.var("AUTH_MODE").map(|v| v.to_string()).as_deref() {
        Ok("key") => "key",
        _ => "open",
    }
}

#[event(fetch)]
pub async fn main(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    console_error_panic_hook::set_once();

    // ===== 防扫保底（zone 级 WAF 之外的第二道防线，Workers 层零成本拦截） =====
    // 1) 常见扫描/利用路径：返回 404（不暴露网关存在感），避免打满 Workers 请求数与 CPU
    let path = req.path().to_string();
    const SCAN_SUFFIXES: &[&str] = &[
        ".php", ".asp", ".aspx", ".jsp", ".cgi", ".env", ".git", ".bak", ".sql",
        ".yaml", ".yml", ".conf", ".ini", ".json.bak", ".zip", ".tar", ".gz",
    ];
    const SCAN_PARTS: &[&str] = &[
        "wp-admin", "wp-login", "wordpress", "phpmyadmin", "phpMyAdmin", "xmlrpc.php",
        ".well-known/probing", "actuator", "console/", "manager/html", "jmx-console",
        "/admin", "/administrator", "/config", "/backup", "/shell", "/eval",
        "/cgi-bin", "/vendor", "/debug", "/telemetry", "/owa/", "/autodiscover",
        "/.env", "/.git", "/.aws", "/.docker", "/.svn", "/.hg", "/.DS_Store",
        "/vendor/phpunit", "/.idea", "/.vscode", "/web.config", "/crossdomain.xml",
    ];
    let p = if path.len() > 1 { path.trim_end_matches('/') } else { &path };
    let lower = p.to_ascii_lowercase();
    let scan_hit = SCAN_SUFFIXES.iter().any(|s| lower.ends_with(s))
        || SCAN_PARTS.iter().any(|s| lower.contains(&s.to_ascii_lowercase()));
    if scan_hit {
        let mut r = Response::from_html("404 Not Found")?.with_status(404);
        r.headers().set("X-Robots-Tag", "noindex, nofollow").ok();
        return Ok(r);
    }

    // 2) 明显的脚本扫描器 UA（对 API 网关无业务意义）：直接 404
    //    白名单：浏览器、curl、wget、OpenAI/主流 SDK、CF Health
    let ua = req
        .headers()
        .get("user-agent")?
        .unwrap_or_default()
        .to_ascii_lowercase();
    let ua_allowed = ua.is_empty()
        || ua.contains("mozilla")
        || ua.contains("curl")
        || ua.contains("wget")
        || ua.contains("python")
        || ua.contains("java/") && ua.contains("okhttp")   // 部分 SDK 用 okhttp
        || ua.contains("openai")
        || ua.contains("node")
        || ua.contains("go-http-client")
        || ua.contains("cloudflare")
        || ua.starts_with("apifox")
        || ua.starts_with("postman")
        || ua.starts_with("insomnia")
        || ua.contains("httpx")
        || ua.contains("aiohttp")
        || ua.contains("reqwest")
        || ua.contains("libcurl");
    let ua_blocked = !ua_allowed
        && (ua.contains("masscan") || ua.contains("nmap") || ua.contains("nikto")
            || ua.contains("sqlmap") || ua.contains("dirbuster") || ua.contains("dirb")
            || ua.contains("gobuster") || ua.contains("wfuzz") || ua.contains("ffuf")
            || ua.contains("nuclei") || ua.contains("zgrab") || ua.contains("acunetix")
            || ua.contains("netsparker") || ua.contains("whatweb") || ua.contains("wpscan"));
    if ua_blocked {
        let mut r = Response::from_html("404 Not Found")?.with_status(404);
        r.headers().set("X-Robots-Tag", "noindex, nofollow").ok();
        return Ok(r);
    }

    if req.method() == Method::Options {
        let mut res = Response::empty()?;
        cors_headers(&mut res)?;
        return Ok(res);
    }

    // 鉴权白名单：健康检查根路径 + 公开模型列表（营销入口）。
    // 其余端点鉴权模式由 AUTH_MODE 决定：
    //   open（默认）→ 任意非空密钥放行（开放传统）
    //   key  → 仅 GATEWAY_KEYS 中的密钥放行（防滥用，可选）
    let is_public = path == "/" || path == "/v1/models";
    if !is_public {
        match extract_key(&req).await {
            Some(k) => {
                let ok = match auth_mode(&env) {
                    "open" => true,
                    _ => key_valid(&env, &k),
                };
                if !ok {
                    return err_invalid_key();
                }
            }
            None => return err_invalid_key(),
        }
    }

    let router = Router::new();

    router
        .get("/", |_, _| {
            let mut res = Response::ok("AQUA API Gateway is running! (Nvidia + Gitee AI)")?;
            cors_headers(&mut res)?;
            Ok(res)
        })
        .get_async("/v1/models", |req, ctx| async move {
            handle_models(req, ctx.env).await
        })
        .post_async("/v1/chat/completions", |req, ctx| async move {
            handle_chat(req, ctx.env).await
        })
        .post_async("/v1/audio/speech", |req, ctx| async move {
            handle_speech(req, ctx.env).await
        })
        .post_async("/v1/audio/transcriptions", |req, ctx| async move {
            handle_transcriptions(req, ctx.env).await
        })
        .post_async("/v1/embeddings", |req, ctx| async move {
            handle_embeddings(req, ctx.env).await
        })
        .post_async("/v1/moderations", |req, ctx| async move {
            handle_moderations(req, ctx.env).await
        })
        .post_async("/v1/images/generations", |req, ctx| async move {
            handle_images(req, ctx.env).await
        })
        .post_async("/v1/videos/generations", |req, ctx| async move {
            handle_videos(req, ctx.env).await
        })
        .post_async("/v1/ip_location", |req, ctx| async move {
            handle_ip_location(req, ctx.env).await
        })
        .post_async("/v1/tools/text-stats", |req, ctx| async move {
            handle_tool_text_stats(req).await
        })
        .post_async("/v1/tools/dice", |req, ctx| async move {
            handle_tool_dice(req).await
        })
        .get_async("/v1/tools/uuid", |req, ctx| async move {
            handle_tool_uuid().await
        })
        .get_async("/v1/tools/timestamp", |req, ctx| async move {
            handle_tool_timestamp_get().await
        })
        .post_async("/v1/tools/timestamp", |req, ctx| async move {
            handle_tool_timestamp_post(req).await
        })
        .post_async("/v1/tools/base64", |req, ctx| async move {
            handle_tool_base64(req).await
        })
        .post_async("/v1/tools/subnet", |req, ctx| async move {
            handle_tool_subnet(req).await
        })
        .post_async("/v1/rerank", |req, ctx| async move {
            handle_rerank(req, ctx.env).await
        })
        .get_async("/assets/*path", |req, ctx| async move {
            handle_assets(req, ctx.env).await
        })
        .run(req, env)
        .await
}
