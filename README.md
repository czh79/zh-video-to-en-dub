# 🎬 中文视频一键转英文配音流水线

> 丢进一条视频链接，出来一条英文口播成片：语音识别 → 语义切片 → LLM 改写 → AI 配音 → 画面重构 → 合成。

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-required-orange?logo=ffmpeg&logoColor=white)](https://ffmpeg.org/)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/czh79/zh-video-to-en-dub/pulls)

**中文** | [English](#-english)

---

## 它解决什么问题

把中文教程视频做成英文版本，传统做法要：听写 → 翻译 → 录音 → 对轨 → 重做字幕 → 遮掉原字幕，一条 10 分钟的视频能耗掉一整天。

这个脚本把整条链路串成了一个命令。它不只是"翻译 + 配音"，而是**重构画面**：裁掉原视频的中文硬字幕、给新英文配音重新配速、生成独立字幕舱和 HUD 命令弹窗，让成品看起来是原生英文内容，而不是机翻配音的搬运。

## 核心能力

| 能力 | 实现 |
|---|---|
| **语义级切片** | faster-whisper 逐词时间戳 + 停顿检测，切成 6–12 秒可独立重排的片段，不会把一句话切两半 |
| **AI 改写** | Gemini / Ollama 双后端，分批改写为地道英文口播稿；LLM 不可用时自动降级到真实翻译保底引擎 |
| **AI 配音** | edge-tts 逐段合成，默认 `en-US-GuyNeural`，语速 +5%，自动检测并重建残损缓存 |
| **画面重构** | 裁剪去除中文硬字幕 → 0.8x–1.3x 动态变速匹配配音时长 → 底部独立字幕舱 → HUD 终端弹窗 → 声波挂件 |
| **断点续跑** | 每个片段的渲染产物独立缓存，中断后重跑不重复计算 |
| **完全本地可控** | 语音识别走本地模型，只有改写环节需要联网（也可切 Ollama 全离线） |

## 工作流程

```
源视频 (URL / 本地文件)
   │
   ├─ 1. yt-dlp 下载，自动复用已下载的 source_raw.mp4
   ├─ 2. faster-whisper 识别中文 → 按语义边界切成片段
   ├─ 3. LLM 批量改写英文口播稿 + 抽取 HUD 命令行
   ├─ 4. edge-tts 逐片段合成英文音频
   ├─ 5. FFmpeg 逐片段渲染：去硬字幕 + 变速 + 字幕舱 + HUD
   └─ 6. concat 拼接 + 可选 BGM 智能闪避混音
                                   ↓
                        output_youtube/yt_geek_tutorial_*.mp4
```

## 快速开始

```bash
git clone https://github.com/czh79/zh-video-to-en-dub.git
cd zh-video-to-en-dub
pip install -r requirements.txt
```

**前置条件**

- FFmpeg 与 ffprobe 已加入 `PATH`，且 FFmpeg 编译时带 `libass`（渲染 ASS 字幕必需）
  ```bash
  ffmpeg -version | grep libass    # 有输出即 OK
  ```
- 首次运行会自动从 `hf-mirror.com` 拉取 Whisper 模型（已内置国内镜像加速）

**配置密钥**（不再写死在代码里）

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="你的密钥"
# macOS / Linux
export GEMINI_API_KEY="你的密钥"
```

**运行**

```bash
python zh2en.py
```

按提示粘贴链接或本地路径，回车即开始。想复用上次下载的视频，直接连按两次回车。

## 配置项

| 变量 | 位置 | 默认值 | 说明 |
|---|---|---|---|
| `GEMINI_API_KEY` | 环境变量 | 空 | 留空则自动降级为翻译保底引擎 |
| `WHISPER_MODEL_ZH` | 环境变量 | `faster-whisper-small` | 中文识别模型，可换 `medium` 提精度 |
| `GEMINI_MODEL` | 代码常量 | `gemini-2.5-flash` | 改写用模型 |
| `OLLAMA_URL` | 代码常量 | `http://localhost:11434/api/generate` | 本地大模型地址 |
| `EDGE_VOICE` | 代码常量 | `en-US-GuyNeural` | 可选 `ChristopherNeural` / `AndrewMultilingualNeural` |
| `OUT_WIDTH / OUT_HEIGHT` | 代码常量 | `1920 × 1080` | 输出分辨率 |
| `BGM_FILE` | 文件 | `assets/cyber_bgm.mp3` | 放进去就自动启用背景音乐闪避混音 |

## 常见问题

**Q：字幕舱盖住画面怎么办？**
改 `build_segment_ass_overlay()` 里的 `MarginV`，数值越小越贴底。

**Q：配音比画面长/短？**
脚本已按配音时长做 0.8x–1.3x 动态变速。超出区间会在日志里提示，可放宽 `EDGE_RATE` 语速。

**Q：能只做翻译不重构画面吗？**
可以，跳过第 5 步的 `render_remastered_segment` 即可，但多数情况下画面重构才是让成品"看起来原生"的关键。

## ⚠️ 使用边界

本项目是**内容本地化工具**，适用于你自己拥有版权、或已获授权的素材。请勿用于搬运他人作品规避平台原创检测——那既违反平台规则，也可能构成侵权。作者不对使用者的二次分发行为负责。

---

## 🇬🇧 English

> A one-command pipeline that turns a Chinese video into an English-dubbed cut: ASR → semantic segmentation → LLM rewrite → TTS dubbing → frame re-authoring → mux.

Most "video translation" tools just overlay a synthetic voice. This one **re-authors the picture**: it crops away the burned-in Chinese subtitles, retimes each segment to match the new English audio, and renders a dedicated subtitle bay plus HUD overlays — so the result reads as native English content rather than a re-upload.

### Features

- **Semantic segmentation** — faster-whisper word timestamps + pause detection produce 6–12s standalone segments instead of cutting mid-sentence.
- **Dual LLM backends** — Gemini or local Ollama, with a rule-based translation fallback when no LLM is reachable.
- **Per-segment TTS** — edge-tts, configurable voice and rate, with automatic rebuild of corrupted cache entries.
- **Frame re-authoring** — hard-subtitle crop, 0.8x–1.3x dynamic retiming, ASS subtitle bay, HUD command popup, waveform widget.
- **Resumable** — every segment is cached independently; a crash costs you one segment, not the whole run.

### Quick start

```bash
git clone https://github.com/czh79/zh-video-to-en-dub.git
cd zh-video-to-en-dub
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"   # optional
python zh2en.py                    # paste a URL or local path
```

Requirements: Python 3.9+, FFmpeg with `libass` on `PATH`.

### Notes

RTX-class GPU is not required — ASR runs on CPU, just slower. A 10-minute video typically takes ~8 minutes end-to-end on a modern laptop.

**Please only use this on content you own or are licensed to localize.** Re-uploading other people's work to dodge originality checks violates platform rules and may infringe copyright.

---

<div align="center">

如果这个项目帮你省下了一天的手工剪辑，**点个 Star** ⭐ 就是最好的支持。

</div>
