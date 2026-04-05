# AutoClip Hand — 开发/运行环境

## Rust（构建 OpenFang / `openfang-hands`）

- 安装 [rustup](https://rustup.rs/)：`stable` 工具链，**rust-version ≥ 1.75**（见仓库根 `Cargo.toml`）。
- Linux 需 **C 语言链接器**，否则 `cargo build` 会报 `linker 'cc' not found`：

```bash
# Debian / Ubuntu
sudo apt-get update && sudo apt-get install -y build-essential pkg-config libssl-dev

# Fedora
sudo dnf install gcc openssl-devel
```

- 构建与测试示例：

```bash
. "$HOME/.cargo/env"
cd /path/to/openfang
cargo test -p openfang-hands
cargo build -p openfang-cli
```

## 运行时二进制（Hand 执行 ffmpeg/yt-dlp 等）

与 Clip Hand 相同，需在 **运行 Agent 的机器**上安装：

| 二进制 | 用途 |
|--------|------|
| `ffmpeg` | 转码、切条、竖屏、烧字幕、缩略图 |
| `ffprobe` | 元数据 |
| `yt-dlp` | URL 下载与字幕（可选路径） |
| `whisper` 或 API | 转写（按 Hand 设置） |

安装示例（Debian/Ubuntu）：`sudo apt install ffmpeg yt-dlp`

## 快速自检

```bash
command -v cargo rustc cc ffmpeg ffprobe || true
```

若 `cc` 缺失，请先安装 `build-essential`（仅构建需要）；仅运行预编译二进制时，可不要求本机 `cargo`。

## 视觉 / 视频理解（AutoClip Phase 3b）

在 OpenFang **内核配置**中开启：

- `media.video_description = true`（否则 `media_describe_video` 会报错）。

环境变量（至少其一，中国大陆优先）：

| 变量 | 说明 |
|------|------|
| `DASHSCOPE_API_KEY` | 阿里云 DashScope → 通义 **Qwen-VL**（默认 `qwen-vl-plus`） |
| `ZHIPU_API_KEY` | 智谱 **GLM-4V**（默认 `glm-4v-plus`） |
| `VOLCENGINE_API_KEY` | 火山引擎 **方舟 Ark** → **豆包 Seed 2.0** 多模态（默认 `doubao-seed-2-0-pro-260215`） |
| `OPENAI_API_KEY` | OpenAI 视觉（可选） |

可选：

- `OPENFANG_VISION_PROVIDER=qwen|zhipu|volcengine|openai` 在多个 Key 同时存在时强制选用。  
- `OPENFANG_VOLCENGINE_VISION_MODEL` — 任选其一（与方舟控制台模型 ID 一致）：  
  - `doubao-seed-2-0-pro-260215`（默认，理解质量优先）  
  - `doubao-seed-2-0-lite-260215`（均衡）  
  - `doubao-seed-2-0-mini-260215`（更快、更省）  
  - 或接入点 **`ep-xxxx`**
