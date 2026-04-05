//! Media understanding engine — image description, audio transcription, video analysis.
//!
//! Auto-cascades through available providers based on configured API keys.
//! Vision: prioritises mainland-compatible OpenAI-style APIs (DashScope Qwen-VL, Zhipu GLM-4V,
//! Volcengine Ark 豆包, then OpenAI); video uses ffmpeg frame sampling + the same vision stack.

use openfang_types::media::{
    MediaAttachment, MediaConfig, MediaSource, MediaType, MediaUnderstanding,
};
use openfang_types::model_catalog::VOLCENGINE_BASE_URL;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::info;

const QWEN_CHAT_COMPLETIONS_URL: &str =
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions";
const ZHIPU_CHAT_COMPLETIONS_URL: &str =
    "https://open.bigmodel.cn/api/paas/v4/chat/completions";
const OPENAI_CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";

/// Media understanding engine.
pub struct MediaEngine {
    config: MediaConfig,
    semaphore: Arc<Semaphore>,
}

impl MediaEngine {
    pub fn new(config: MediaConfig) -> Self {
        let max = config.max_concurrency.clamp(1, 8);
        Self {
            config,
            semaphore: Arc::new(Semaphore::new(max)),
        }
    }

    /// Describe an image using a vision-capable LLM (no custom prompt).
    pub async fn describe_image(
        &self,
        attachment: &MediaAttachment,
    ) -> Result<MediaUnderstanding, String> {
        self.describe_image_prompt(attachment, None).await
    }

    /// Describe an image with an optional user prompt (e.g. editing-oriented instructions).
    ///
    /// Provider order (unless `OPENFANG_VISION_PROVIDER` overrides): **Qwen-VL** (DashScope),
    /// **GLM-4V** (Zhipu), **豆包** (火山 Ark), OpenAI, then legacy stubs for Gemini/Anthropic.
    pub async fn describe_image_prompt(
        &self,
        attachment: &MediaAttachment,
        prompt: Option<&str>,
    ) -> Result<MediaUnderstanding, String> {
        attachment.validate()?;
        if attachment.media_type != MediaType::Image {
            return Err("Expected image attachment".into());
        }

        let _permit = self.semaphore.acquire().await.map_err(|e| e.to_string())?;

        let (image_bytes, mime) = read_image_bytes(attachment).await?;
        let provider = self
            .config
            .image_provider
            .as_deref()
            .or_else(|| detect_vision_provider())
            .ok_or(
                "No vision provider configured. For mainland CN: set DASHSCOPE_API_KEY (Qwen-VL), \
                 ZHIPU_API_KEY (GLM-4V), VOLCENGINE_API_KEY (火山豆包 Ark), or OPENAI_API_KEY. \
                 Optional: OPENFANG_VISION_PROVIDER=qwen|zhipu|volcengine|openai",
            )?;

        let default_prompt = "Describe this image in detail: main subjects, setting, actions, \
             on-screen text if any, mood, and anything relevant for video editing or highlight selection.";
        let text_prompt = prompt.unwrap_or(default_prompt);

        let (description, model_used) = match provider {
            "qwen" => {
                let key = std::env::var("DASHSCOPE_API_KEY")
                    .map_err(|_| "DASHSCOPE_API_KEY not set".to_string())?;
                let model = default_vision_model("qwen");
                let text = openai_style_vision(
                    QWEN_CHAT_COMPLETIONS_URL,
                    &key,
                    model,
                    text_prompt,
                    &mime,
                    &image_bytes,
                )
                .await?;
                (text, model.to_string())
            }
            "zhipu" => {
                let key = std::env::var("ZHIPU_API_KEY")
                    .map_err(|_| "ZHIPU_API_KEY not set".to_string())?;
                let model = default_vision_model("zhipu");
                let text = openai_style_vision(
                    ZHIPU_CHAT_COMPLETIONS_URL,
                    &key,
                    model,
                    text_prompt,
                    &mime,
                    &image_bytes,
                )
                .await?;
                (text, model.to_string())
            }
            "openai" => {
                let key = std::env::var("OPENAI_API_KEY")
                    .map_err(|_| "OPENAI_API_KEY not set".to_string())?;
                let model = default_vision_model("openai");
                let text = openai_style_vision(
                    OPENAI_CHAT_COMPLETIONS_URL,
                    &key,
                    model,
                    text_prompt,
                    &mime,
                    &image_bytes,
                )
                .await?;
                (text, model.to_string())
            }
            "volcengine" | "doubao" => {
                let key = std::env::var("VOLCENGINE_API_KEY")
                    .map_err(|_| "VOLCENGINE_API_KEY not set".to_string())?;
                let model = volcengine_vision_model();
                let url = volcengine_chat_completions_url();
                let text = openai_style_vision(
                    &url,
                    &key,
                    &model,
                    text_prompt,
                    &mime,
                    &image_bytes,
                )
                .await?;
                (text, model)
            }
            "gemini" => {
                return Err(
                    "Gemini vision is not wired in MediaEngine yet. Use DASHSCOPE_API_KEY (Qwen-VL), \
                     ZHIPU_API_KEY (GLM-4V), VOLCENGINE_API_KEY (豆包), or OPENAI_API_KEY."
                        .into(),
                );
            }
            "anthropic" => {
                return Err(
                    "Anthropic vision is not implemented in MediaEngine. Use DASHSCOPE_API_KEY, \
                     ZHIPU_API_KEY, VOLCENGINE_API_KEY, or OPENAI_API_KEY."
                        .into(),
                );
            }
            other => {
                return Err(format!(
                    "Unsupported vision provider '{other}'. Use qwen, zhipu, volcengine, or openai."
                ));
            }
        };

        info!(provider, model = %model_used, chars = description.len(), "Image description complete");

        Ok(MediaUnderstanding {
            media_type: MediaType::Image,
            description,
            provider: provider.to_string(),
            model: model_used,
        })
    }

    /// Transcribe audio using speech-to-text.
    /// Auto-cascade: Groq (whisper-large-v3-turbo) -> OpenAI (whisper-1).
    pub async fn transcribe_audio(
        &self,
        attachment: &MediaAttachment,
    ) -> Result<MediaUnderstanding, String> {
        attachment.validate()?;
        if attachment.media_type != MediaType::Audio {
            return Err("Expected audio attachment".into());
        }

        let provider = self
            .config
            .audio_provider
            .as_deref()
            .or_else(|| detect_audio_provider())
            .ok_or(
                "No audio transcription provider configured. Set GROQ_API_KEY or OPENAI_API_KEY",
            )?;

        let _permit = self.semaphore.acquire().await.map_err(|e| e.to_string())?;

        // Parakeet MLX — local transcription via uv + Python
        if provider == "parakeet-mlx" {
            return transcribe_with_parakeet_mlx(attachment).await;
        }

        // Derive a proper filename with extension from mime_type
        // (Whisper APIs require an extension to detect format)
        let ext = match attachment.mime_type.as_str() {
            "audio/wav" => "wav",
            "audio/mpeg" | "audio/mp3" => "mp3",
            "audio/ogg" => "ogg",
            "audio/webm" => "webm",
            "audio/mp4" | "audio/m4a" => "m4a",
            "audio/flac" => "flac",
            _ => "wav",
        };

        // Read audio bytes from source
        let audio_bytes = match &attachment.source {
            MediaSource::FilePath { path } => tokio::fs::read(path)
                .await
                .map_err(|e| format!("Failed to read audio file '{}': {}", path, e))?,
            MediaSource::Base64 { data, .. } => {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .map_err(|e| format!("Failed to decode base64 audio: {}", e))?
            }
            MediaSource::Url { url } => {
                return Err(format!(
                    "URL-based audio source not supported for transcription: {}",
                    url
                ));
            }
        };
        let filename = format!("audio.{}", ext);

        let model = default_audio_model(provider);

        // Build API request
        let (api_url, api_key) = match provider {
            "groq" => (
                "https://api.groq.com/openai/v1/audio/transcriptions",
                std::env::var("GROQ_API_KEY").map_err(|_| "GROQ_API_KEY not set")?,
            ),
            "openai" => (
                "https://api.openai.com/v1/audio/transcriptions",
                std::env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY not set")?,
            ),
            other => return Err(format!("Unsupported audio provider: {}", other)),
        };

        info!(provider, model, filename = %filename, size = audio_bytes.len(), "Sending audio for transcription");

        let file_part = reqwest::multipart::Part::bytes(audio_bytes)
            .file_name(filename)
            .mime_str(&attachment.mime_type)
            .map_err(|e| format!("Failed to set MIME type: {}", e))?;

        let form = reqwest::multipart::Form::new()
            .part("file", file_part)
            .text("model", model.to_string())
            .text("response_format", "text");

        let client = reqwest::Client::new();
        let resp = client
            .post(api_url)
            .bearer_auth(&api_key)
            .multipart(form)
            .timeout(std::time::Duration::from_secs(60))
            .send()
            .await
            .map_err(|e| format!("Transcription request failed: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Transcription API error ({}): {}", status, body));
        }

        let transcription = resp
            .text()
            .await
            .map_err(|e| format!("Failed to read transcription response: {}", e))?;

        let transcription = transcription.trim().to_string();
        if transcription.is_empty() {
            return Err("Transcription returned empty text".into());
        }

        info!(
            provider,
            model,
            chars = transcription.len(),
            "Audio transcription complete"
        );

        Ok(MediaUnderstanding {
            media_type: MediaType::Audio,
            description: transcription,
            provider: provider.to_string(),
            model: model.to_string(),
        })
    }

    /// Describe video by **sampling frames** with ffmpeg, then running the same vision stack as images.
    ///
    /// Works with **Qwen-VL** (DashScope), **GLM-4V** (Zhipu), **豆包** (火山 Ark), or **OpenAI** vision models.
    /// Requires `ffmpeg` / `ffprobe` on `PATH`. Enable `MediaConfig.video_description` in kernel config.
    pub async fn describe_video(
        &self,
        attachment: &MediaAttachment,
        prompt: Option<&str>,
    ) -> Result<MediaUnderstanding, String> {
        attachment.validate()?;
        if attachment.media_type != MediaType::Video {
            return Err("Expected video attachment".into());
        }

        if !self.config.video_description {
            return Err("Video description is disabled in configuration (set media.video_description = true)".into());
        }

        let _permit = self.semaphore.acquire().await.map_err(|e| e.to_string())?;

        let provider = self
            .config
            .image_provider
            .as_deref()
            .or_else(|| detect_vision_provider())
            .ok_or(
                "No vision provider for video frames. Set DASHSCOPE_API_KEY, ZHIPU_API_KEY, VOLCENGINE_API_KEY, or OPENAI_API_KEY.",
            )?;
        if !matches!(
            provider,
            "qwen" | "zhipu" | "openai" | "volcengine" | "doubao"
        ) {
            return Err(format!(
                "Video frame analysis requires provider qwen, zhipu, volcengine, or openai (got '{provider}'). \
                 Set DASHSCOPE_API_KEY, ZHIPU_API_KEY, VOLCENGINE_API_KEY, or OPENAI_API_KEY."
            ));
        }

        let (video_path, is_temp_video) = materialize_video_path(attachment).await?;
        let duration = tokio::task::spawn_blocking({
            let p = video_path.clone();
            move || ffprobe_duration_sec(&p)
        })
        .await
        .map_err(|e| format!("ffprobe task join: {e}"))??;

        if duration <= 0.0 || !duration.is_finite() {
            if is_temp_video {
                let _ = tokio::fs::remove_file(&video_path).await;
            }
            return Err("ffprobe returned invalid duration".into());
        }

        let offsets: Vec<f64> = [0.08, 0.28, 0.48, 0.68, 0.88]
            .iter()
            .map(|f| (duration * f).max(0.0))
            .collect();

        let default_prompt = "Describe what is visible in this frame: people, actions, setting, \
             text on screen, and anything useful for picking viral short-video clips.";
        let frame_prompt = prompt.unwrap_or(default_prompt);

        let mut parts: Vec<String> = Vec::new();
        let model = match provider {
            "volcengine" | "doubao" => volcengine_vision_model(),
            _ => default_vision_model(provider).to_string(),
        };

        for (i, t) in offsets.iter().enumerate() {
            let frame_path = tokio::task::spawn_blocking({
                let vid = video_path.clone();
                let at = *t;
                let idx = i;
                move || extract_frame_jpeg(&vid, at, idx)
            })
            .await
            .map_err(|e| format!("ffmpeg task join: {e}"))??;

            let jpeg = tokio::fs::read(&frame_path)
                .await
                .map_err(|e| format!("read frame: {e}"))?;
            let _ = tokio::fs::remove_file(&frame_path).await;

            let text = match provider {
                "qwen" => {
                    let key = std::env::var("DASHSCOPE_API_KEY").map_err(|_| "DASHSCOPE_API_KEY not set")?;
                    openai_style_vision(
                        QWEN_CHAT_COMPLETIONS_URL,
                        &key,
                        &model,
                        frame_prompt,
                        "image/jpeg",
                        &jpeg,
                    )
                    .await?
                }
                "zhipu" => {
                    let key = std::env::var("ZHIPU_API_KEY").map_err(|_| "ZHIPU_API_KEY not set")?;
                    openai_style_vision(
                        ZHIPU_CHAT_COMPLETIONS_URL,
                        &key,
                        &model,
                        frame_prompt,
                        "image/jpeg",
                        &jpeg,
                    )
                    .await?
                }
                "openai" => {
                    let key = std::env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY not set")?;
                    openai_style_vision(
                        OPENAI_CHAT_COMPLETIONS_URL,
                        &key,
                        &model,
                        frame_prompt,
                        "image/jpeg",
                        &jpeg,
                    )
                    .await?
                }
                "volcengine" | "doubao" => {
                    let key =
                        std::env::var("VOLCENGINE_API_KEY").map_err(|_| "VOLCENGINE_API_KEY not set")?;
                    let url = volcengine_chat_completions_url();
                    openai_style_vision(
                        &url,
                        &key,
                        &model,
                        frame_prompt,
                        "image/jpeg",
                        &jpeg,
                    )
                    .await?
                }
                _ => unreachable!(),
            };

            parts.push(format!(
                "### ~{:.1}s (sample {}/{})\n{}",
                t,
                i + 1,
                offsets.len(),
                text.trim()
            ));
        }

        if is_temp_video {
            let _ = tokio::fs::remove_file(&video_path).await;
        }

        let description = format!(
            "Video understanding ({} frames, duration {:.1}s):\n\n{}",
            offsets.len(),
            duration,
            parts.join("\n\n")
        );

        info!(provider, model = %model, chars = description.len(), "Video frame description complete");

        Ok(MediaUnderstanding {
            media_type: MediaType::Video,
            description,
            provider: provider.to_string(),
            model,
        })
    }

    /// Process multiple attachments concurrently (bounded by max_concurrency).
    pub async fn process_attachments(
        &self,
        attachments: Vec<MediaAttachment>,
    ) -> Vec<Result<MediaUnderstanding, String>> {
        let mut handles = Vec::new();

        for attachment in attachments {
            let sem = self.semaphore.clone();
            let config = self.config.clone();
            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.map_err(|e| e.to_string())?;
                let engine = MediaEngine {
                    config,
                    semaphore: Arc::new(Semaphore::new(1)), // inner engine, no extra semaphore
                };
                match attachment.media_type {
                    MediaType::Image => engine.describe_image(&attachment).await,
                    MediaType::Audio => engine.transcribe_audio(&attachment).await,
                    MediaType::Video => engine.describe_video(&attachment, None).await,
                }
            });
            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => results.push(Err(format!("Task failed: {e}"))),
            }
        }
        results
    }
}

/// Read raw image bytes and MIME type from an attachment.
async fn read_image_bytes(attachment: &MediaAttachment) -> Result<(Vec<u8>, String), String> {
    match &attachment.source {
        MediaSource::FilePath { path } => {
            let data = tokio::fs::read(path)
                .await
                .map_err(|e| format!("Failed to read image '{path}': {e}"))?;
            Ok((data, attachment.mime_type.clone()))
        }
        MediaSource::Base64 { data, mime_type } => {
            use base64::Engine;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|e| format!("Invalid base64 image: {e}"))?;
            Ok((bytes, mime_type.clone()))
        }
        MediaSource::Url { url } => Err(format!("URL image source not supported: {url}")),
    }
}

/// Returns `(path, is_temp_file)`.
async fn materialize_video_path(attachment: &MediaAttachment) -> Result<(PathBuf, bool), String> {
    match &attachment.source {
        MediaSource::FilePath { path } => Ok((PathBuf::from(path), false)),
        MediaSource::Base64 { data, .. } => {
            use base64::Engine;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|e| format!("Invalid base64 video: {e}"))?;
            let p = std::env::temp_dir().join(format!("openfang_vid_{}.mp4", uuid::Uuid::new_v4()));
            tokio::fs::write(&p, bytes)
                .await
                .map_err(|e| format!("temp video write: {e}"))?;
            Ok((p, true))
        }
        MediaSource::Url { url } => Err(format!("URL video source not supported: {url}")),
    }
}

fn ffprobe_duration_sec(path: &Path) -> Result<f64, String> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path)
        .output()
        .map_err(|e| format!("ffprobe failed to start: {e}. Is ffmpeg installed?"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!("ffprobe error: {}", err.trim()));
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let v: f64 = s
        .trim()
        .parse()
        .map_err(|_| format!("ffprobe returned non-numeric duration: {}", s.trim()))?;
    Ok(v)
}

fn extract_frame_jpeg(video: &Path, at_sec: f64, idx: usize) -> Result<PathBuf, String> {
    let out = std::env::temp_dir().join(format!(
        "openfang_frame_{}_{}.jpg",
        uuid::Uuid::new_v4(),
        idx
    ));
    let ss = format!("{:.3}", at_sec);
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            &ss,
            "-i",
        ])
        .arg(video)
        .args(["-frames:v", "1", "-q:v", "2"])
        .arg(&out)
        .status()
        .map_err(|e| format!("ffmpeg failed to start: {e}. Is ffmpeg installed?"))?;
    if !status.success() {
        return Err("ffmpeg frame extract failed".into());
    }
    Ok(out)
}

fn parse_chat_completion_content(body: &serde_json::Value) -> Result<String, String> {
    let content = body
        .pointer("/choices/0/message/content")
        .ok_or("Missing choices[0].message.content in API response")?;
    if let Some(s) = content.as_str() {
        let t = s.trim();
        if t.is_empty() {
            return Err("Vision model returned empty content".into());
        }
        return Ok(t.to_string());
    }
    if let Some(arr) = content.as_array() {
        let mut out = String::new();
        for part in arr {
            if let Some(t) = part.get("text").and_then(|x| x.as_str()) {
                out.push_str(t);
            }
        }
        let t = out.trim();
        if !t.is_empty() {
            return Ok(t.to_string());
        }
    }
    Err("Unexpected vision API response shape".into())
}

/// OpenAI-compatible `/chat/completions` with a single `image_url` data URI.
async fn openai_style_vision(
    endpoint: &str,
    api_key: &str,
    model: &str,
    user_text: &str,
    image_mime: &str,
    image_bytes: &[u8],
) -> Result<String, String> {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);
    let data_uri = format!("data:{image_mime};base64,{b64}");

    let body = json!({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                { "type": "text", "text": user_text },
                { "type": "image_url", "image_url": { "url": data_uri } }
            ]
        }],
        "max_tokens": 2048
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(endpoint)
        .bearer_auth(api_key)
        .json(&body)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await
        .map_err(|e| format!("Vision HTTP error: {e}"))?;

    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| format!("Vision read body: {e}"))?;

    if !status.is_success() {
        return Err(format!("Vision API {}: {}", status, text));
    }

    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("Vision JSON parse: {e}: {text}"))?;
    parse_chat_completion_content(&v)
}

/// `{VOLCENGINE_BASE_URL}/chat/completions` — 火山引擎方舟 OpenAPI（豆包等）.
fn volcengine_chat_completions_url() -> String {
    format!(
        "{}/chat/completions",
        VOLCENGINE_BASE_URL.trim_end_matches('/')
    )
}

/// 豆包 / 多模态推理接入点：默认 **Seed 2.0 Pro**；可改为 `doubao-seed-2-0-lite-260215` / `mini` 或方舟 `ep-xxxx`。
fn volcengine_vision_model() -> String {
    std::env::var("OPENFANG_VOLCENGINE_VISION_MODEL").unwrap_or_else(|_| {
        "doubao-seed-2-0-pro-260215".to_string()
    })
}

/// Detect which vision provider to use (`OPENFANG_VISION_PROVIDER` overrides).
fn detect_vision_provider() -> Option<&'static str> {
    if let Ok(p) = std::env::var("OPENFANG_VISION_PROVIDER") {
        let low = p.to_lowercase();
        match low.as_str() {
            "qwen" | "dashscope" => return Some("qwen"),
            "zhipu" | "glm" | "bigmodel" => return Some("zhipu"),
            "volcengine" | "doubao" | "ark" => return Some("volcengine"),
            "openai" => return Some("openai"),
            "gemini" | "google" => return Some("gemini"),
            "anthropic" | "claude" => return Some("anthropic"),
            _ => {}
        }
    }
    if std::env::var("DASHSCOPE_API_KEY").is_ok() {
        return Some("qwen");
    }
    if std::env::var("ZHIPU_API_KEY").is_ok() {
        return Some("zhipu");
    }
    if std::env::var("VOLCENGINE_API_KEY").is_ok() {
        return Some("volcengine");
    }
    if std::env::var("OPENAI_API_KEY").is_ok() {
        return Some("openai");
    }
    if std::env::var("GEMINI_API_KEY").is_ok() || std::env::var("GOOGLE_API_KEY").is_ok() {
        return Some("gemini");
    }
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        return Some("anthropic");
    }
    None
}

/// Transcribe audio using Parakeet MLX (local, via uv + Python).
async fn transcribe_with_parakeet_mlx(
    attachment: &MediaAttachment,
) -> Result<MediaUnderstanding, String> {
    use tokio::time::{timeout, Duration};

    // Materialize audio to a temp file if needed
    let (audio_path, is_temp) = match &attachment.source {
        MediaSource::FilePath { path } => (std::path::PathBuf::from(path), false),
        MediaSource::Base64 { data, mime_type } => {
            use base64::Engine;
            let decoded = base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|e| format!("Failed to decode base64 audio: {e}"))?;
            let ext = match mime_type.as_str() {
                "audio/wav" | "audio/x-wav" => "wav",
                "audio/mpeg" | "audio/mp3" => "mp3",
                "audio/ogg" => "ogg",
                "audio/webm" => "webm",
                "audio/mp4" | "audio/m4a" => "m4a",
                "audio/flac" => "flac",
                _ => "wav",
            };
            let path = std::env::temp_dir().join(format!(
                "openfang_parakeet_{}.{}",
                uuid::Uuid::new_v4(),
                ext
            ));
            tokio::fs::write(&path, decoded)
                .await
                .map_err(|e| format!("Failed to write temp audio: {e}"))?;
            (path, true)
        }
        MediaSource::Url { url } => {
            return Err(format!("URL audio not supported for parakeet-mlx: {url}"));
        }
    };

    let script = r#"
import json, sys
from parakeet_mlx import from_pretrained
model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
result = model.transcribe(sys.argv[1])
print(json.dumps({"text": result.text, "model": "mlx-community/parakeet-tdt-0.6b-v3"}))
"#;

    let mut cmd = tokio::process::Command::new("uv");
    cmd.args([
        "run",
        "--with",
        "parakeet-mlx",
        "python3",
        "-c",
        script,
        &audio_path.to_string_lossy(),
    ]);
    cmd.env("PYTHONUNBUFFERED", "1");
    cmd.kill_on_drop(true);

    let output = timeout(Duration::from_secs(900), cmd.output())
        .await
        .map_err(|_| "parakeet-mlx timed out after 15 minutes".to_string())?
        .map_err(|e| format!("Failed to launch parakeet-mlx: {e}"))?;

    if is_temp {
        let _ = tokio::fs::remove_file(&audio_path).await;
    }

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("parakeet-mlx failed: {}", stderr.trim()));
    }

    let stdout =
        String::from_utf8(output.stdout).map_err(|e| format!("parakeet-mlx non-UTF8: {e}"))?;
    let parsed: serde_json::Value = serde_json::from_str(stdout.trim())
        .map_err(|e| format!("parakeet-mlx parse failed: {e}"))?;

    let text = parsed["text"]
        .as_str()
        .ok_or("missing text field")?
        .trim()
        .to_string();
    if text.is_empty() {
        return Err("parakeet-mlx returned empty transcription".into());
    }

    Ok(MediaUnderstanding {
        media_type: MediaType::Audio,
        description: text,
        provider: "parakeet-mlx".to_string(),
        model: parsed["model"]
            .as_str()
            .unwrap_or("parakeet-tdt-0.6b-v3")
            .to_string(),
    })
}

/// Detect which audio transcription provider is available.
fn detect_audio_provider() -> Option<&'static str> {
    // Explicit opt-in for local Parakeet MLX transcription
    if std::env::var("OPENFANG_ENABLE_PARAKEET_MLX").is_ok() {
        return Some("parakeet-mlx");
    }
    if std::env::var("GROQ_API_KEY").is_ok() {
        return Some("groq");
    }
    if std::env::var("OPENAI_API_KEY").is_ok() {
        return Some("openai");
    }
    None
}

/// Get the default vision model for a provider.
fn default_vision_model(provider: &str) -> &str {
    match provider {
        "qwen" => "qwen-vl-plus",
        "zhipu" => "glm-4v-plus",
        "volcengine" | "doubao" => "doubao-seed-2-0-pro-260215",
        "openai" => "gpt-4o",
        "anthropic" => "claude-sonnet-4-20250514",
        "gemini" => "gemini-2.5-flash",
        _ => "unknown",
    }
}

/// Get the default audio model for a provider.
fn default_audio_model(provider: &str) -> &str {
    match provider {
        "parakeet-mlx" => "mlx-community/parakeet-tdt-0.6b-v3",
        "groq" => "whisper-large-v3-turbo",
        "openai" => "whisper-1",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openfang_types::media::{MediaSource, MAX_IMAGE_BYTES};

    #[test]
    fn test_engine_creation() {
        let config = MediaConfig::default();
        let engine = MediaEngine::new(config);
        assert_eq!(engine.config.max_concurrency, 2);
    }

    #[test]
    fn test_engine_max_concurrency_clamped() {
        let config = MediaConfig {
            max_concurrency: 100,
            ..Default::default()
        };
        let engine = MediaEngine::new(config);
        // Semaphore was clamped to 8
        assert!(engine.semaphore.available_permits() <= 8);
    }

    #[tokio::test]
    async fn test_describe_image_wrong_type() {
        let engine = MediaEngine::new(MediaConfig::default());
        let attachment = MediaAttachment {
            media_type: MediaType::Audio,
            mime_type: "audio/mpeg".into(),
            source: MediaSource::FilePath {
                path: "test.mp3".into(),
            },
            size_bytes: 1024,
        };
        let result = engine.describe_image(&attachment).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Expected image"));
    }

    #[tokio::test]
    async fn test_describe_image_invalid_mime() {
        let engine = MediaEngine::new(MediaConfig::default());
        let attachment = MediaAttachment {
            media_type: MediaType::Image,
            mime_type: "application/pdf".into(),
            source: MediaSource::FilePath {
                path: "test.pdf".into(),
            },
            size_bytes: 1024,
        };
        let result = engine.describe_image(&attachment).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_describe_image_too_large() {
        let engine = MediaEngine::new(MediaConfig::default());
        let attachment = MediaAttachment {
            media_type: MediaType::Image,
            mime_type: "image/png".into(),
            source: MediaSource::FilePath {
                path: "big.png".into(),
            },
            size_bytes: MAX_IMAGE_BYTES + 1,
        };
        let result = engine.describe_image(&attachment).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_transcribe_audio_wrong_type() {
        let engine = MediaEngine::new(MediaConfig::default());
        let attachment = MediaAttachment {
            media_type: MediaType::Image,
            mime_type: "image/png".into(),
            source: MediaSource::FilePath {
                path: "test.png".into(),
            },
            size_bytes: 1024,
        };
        let result = engine.transcribe_audio(&attachment).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_video_disabled() {
        let config = MediaConfig {
            video_description: false,
            ..Default::default()
        };
        let engine = MediaEngine::new(config);
        let attachment = MediaAttachment {
            media_type: MediaType::Video,
            mime_type: "video/mp4".into(),
            source: MediaSource::FilePath {
                path: "test.mp4".into(),
            },
            size_bytes: 1024,
        };
        let result = engine.describe_video(&attachment, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("disabled"));
    }

    #[test]
    fn test_detect_vision_provider_none() {
        // In test env, likely no API keys set — should return None.
        // (This test is environment-dependent, but safe.)
        let _ = detect_vision_provider(); // Just verify it doesn't panic
    }

    #[test]
    fn test_default_vision_models() {
        assert_eq!(default_vision_model("qwen"), "qwen-vl-plus");
        assert_eq!(default_vision_model("zhipu"), "glm-4v-plus");
        assert_eq!(
            default_vision_model("volcengine"),
            "doubao-seed-2-0-pro-260215"
        );
        assert_eq!(
            default_vision_model("doubao"),
            "doubao-seed-2-0-pro-260215"
        );
        assert_eq!(
            default_vision_model("anthropic"),
            "claude-sonnet-4-20250514"
        );
        assert_eq!(default_vision_model("openai"), "gpt-4o");
        assert_eq!(default_vision_model("gemini"), "gemini-2.5-flash");
        assert_eq!(default_vision_model("unknown"), "unknown");
    }

    #[test]
    fn test_default_audio_models() {
        assert_eq!(default_audio_model("groq"), "whisper-large-v3-turbo");
        assert_eq!(default_audio_model("openai"), "whisper-1");
    }

    #[tokio::test]
    async fn test_transcribe_audio_rejects_image_type() {
        let engine = MediaEngine::new(MediaConfig::default());
        let attachment = MediaAttachment {
            media_type: MediaType::Image,
            mime_type: "image/png".into(),
            source: MediaSource::FilePath {
                path: "test.png".into(),
            },
            size_bytes: 1024,
        };
        let result = engine.transcribe_audio(&attachment).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Expected audio"));
    }

    #[tokio::test]
    async fn test_transcribe_audio_no_provider() {
        // With no API keys set, should fail with provider error
        let engine = MediaEngine::new(MediaConfig::default());
        let attachment = MediaAttachment {
            media_type: MediaType::Audio,
            mime_type: "audio/webm".into(),
            source: MediaSource::FilePath {
                path: "test.webm".into(),
            },
            size_bytes: 1024,
        };
        let result = engine.transcribe_audio(&attachment).await;
        // Either fails with "No audio transcription provider" or file read error
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_transcribe_audio_url_source_rejected() {
        // URL source should be rejected
        let config = MediaConfig {
            audio_provider: Some("groq".to_string()),
            ..Default::default()
        };
        let engine = MediaEngine::new(config);
        let attachment = MediaAttachment {
            media_type: MediaType::Audio,
            mime_type: "audio/mpeg".into(),
            source: MediaSource::Url {
                url: "https://example.com/audio.mp3".into(),
            },
            size_bytes: 1024,
        };
        let result = engine.transcribe_audio(&attachment).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("URL-based audio source not supported"));
    }

    #[tokio::test]
    async fn test_transcribe_audio_file_not_found() {
        let config = MediaConfig {
            audio_provider: Some("groq".to_string()),
            ..Default::default()
        };
        let engine = MediaEngine::new(config);
        let attachment = MediaAttachment {
            media_type: MediaType::Audio,
            mime_type: "audio/webm".into(),
            source: MediaSource::FilePath {
                path: "/nonexistent/path/audio.webm".into(),
            },
            size_bytes: 1024,
        };
        let result = engine.transcribe_audio(&attachment).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to read audio file"));
    }
}
