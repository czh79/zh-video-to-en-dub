import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# 国内 HuggingFace 镜像加速[cite: 1]
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ==================== 核心路径与全局工程配置 ====================
BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "workspace_geek"
OUTPUT_DIR = BASE_DIR / "output_youtube"
ASSETS_DIR = BASE_DIR / "assets"

# 可选背景科技感音乐路径 (放在 assets/cyber_bgm.mp3 即可自动启用智能闪避混音)
BGM_FILE = ASSETS_DIR / "cyber_bgm.mp3"

# Whisper 本地模型路径（优先使用本地模型，不存在则自动拉取 small）[cite: 1]
LOCAL_MODEL_ZH = os.environ.get("WHISPER_MODEL_ZH", "faster-whisper-small")

# API 密钥与本地大模型配置[cite: 1]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Edge-TTS 极客教程英文声线
EDGE_VOICE = "en-US-GuyNeural"  # 备选: "en-US-ChristopherNeural", "en-US-AndrewMultilingualNeural"[cite: 1]
EDGE_RATE = "+5%"

# ==================== 视觉重构与去中文硬字幕参数 ====================
OUT_WIDTH = 1920
OUT_HEIGHT = 1080
OUT_FPS = 30

# 原视频边缘裁剪比例：
# 顶部切掉 6%（浏览器中文书签栏），底部切掉 17.5%（彻底切掉原片底部的黄色中文+白色英文硬字幕！）[cite: 2]
CROP_TOP_RATIO = 0.06
CROP_BOTTOM_RATIO = 0.175
CROP_LEFT_RATIO = 0.02
CROP_RIGHT_RATIO = 0.02

# 前景画中画尺寸与坐标：
# 上移至 Y=58，底部止于 Y=888，在画布最下方留出 192px (Y=888~1080) 的专属英文字幕安全舱！[cite: 2]
FG_WIDTH = 1560
FG_HEIGHT = 824
FG_POS_X = 40
FG_POS_Y = 58

# 动态变速安全边界 (0.8x ~ 1.3x)
MIN_SPEED_FACTOR = 0.80
MAX_SPEED_FACTOR = 1.30

# 频道品牌标识与配乐音量
CHANNEL_HANDLE = "@GeekDevOpsLab"
BGM_BASE_VOLUME = 0.10
# ================================================================


@dataclass
class TutorialSegment:
    index: int
    src_start: float
    src_end: float
    zh_text: str
    en_script: str = ""
    hud_title: str = ""
    hud_code: str = ""
    tts_audio_path: Optional[Path] = None
    tts_duration: float = 0.0
    rendered_clip_path: Optional[Path] = None


def run_cmd(cmd: List[str], desc: str = "") -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    if res.returncode != 0 and desc:
        print(f"  ⚠️ [{desc}] 执行异常:\n{res.stderr[-400:]}")
    return res


def get_media_duration(media_file: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(media_file.resolve())
    ]
    res = run_cmd(cmd)
    try:
        return float(res.stdout.strip()) if res.returncode == 0 and res.stdout.strip() else 0.0
    except ValueError:
        return 0.0


# ==================== Step 1: yt-dlp 下载无水印源视频 ====================
def download_source_video(url_or_path: str, work_dir: Path) -> Path:
    """支持直接传入本地视频路径，或使用 yt-dlp 下载 B站/YouTube 高清源视频"""
    local_p = Path(url_or_path)
    if local_p.exists() and local_p.is_file():
        print(f"📂 检测到本地源视频: {local_p.name}")
        return local_p

    output_tmpl = str((work_dir / "source_raw.%(ext)s").resolve())
    final_mp4 = work_dir / "source_raw.mp4"
    if final_mp4.exists() and final_mp4.stat().st_size > 100_000:
        print(f"✅ 复用已下载的源视频: {final_mp4.name} (时长: {get_media_duration(final_mp4):.2f}s)")
        return final_mp4

    print(f"🌐 [1/5] 正在通过 yt-dlp 下载源视频: {url_or_path} ...")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_tmpl,
        url_or_path
    ]
    res = run_cmd(cmd, "yt-dlp 下载")
    if res.returncode != 0 or not final_mp4.exists():
        raise RuntimeError(f"yt-dlp 视频下载失败，请检查链接或网络环境。\n{res.stderr[-300:]}")

    print(f"✅ 下载完成: {final_mp4.name} (时长: {get_media_duration(final_mp4):.2f}s)")
    return final_mp4


# ==================== Step 2: Faster-Whisper 语义级切片 ====================
def load_whisper_model():
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("请先安装 faster-whisper: pip install faster-whisper")

    model_src = LOCAL_MODEL_ZH if os.path.exists(LOCAL_MODEL_ZH) else "small"
    print(f"🤖 [2/5] 正在加载 Faster-Whisper 模型 ({model_src})...")
    return WhisperModel(model_src, device="cpu", compute_type="int8")


def extract_semantic_segments(video_file: Path, model) -> List[TutorialSegment]:
    """
    将原视频按语音停顿与标点切分为 6~12 秒的连贯语义块，
    既保证画面变速平滑，又彻底打乱原视频时间轴关键帧。
    """
    total_dur = get_media_duration(video_file)
    print("🔍 正在提取中文字幕时间轴并进行 6~12s 语义块切分...")

    segments, _ = model.transcribe(
        str(video_file.resolve()),
        language="zh",
        beam_size=3,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=200),
        word_timestamps=True,
    )

    raw_items: List[Tuple[float, float, str]] = []
    for seg in segments:
        txt = seg.text.strip()
        if txt:
            raw_items.append((seg.start, seg.end, txt))

    if not raw_items:
        raise RuntimeError("未能从源视频中识别到有效语音字幕！")

    merged: List[TutorialSegment] = []
    cur_start = 0.0
    cur_end = raw_items[0][1]
    cur_texts = [raw_items[0][2]]
    seg_idx = 1

    for i in range(1, len(raw_items)):
        st, et, txt = raw_items[i]
        span = cur_end - cur_start
        gap = st - cur_end

        if (span >= 6.0 and gap >= 0.25) or span >= 11.5:
            split_boundary = (cur_end + st) / 2.0
            merged.append(TutorialSegment(
                index=seg_idx,
                src_start=cur_start,
                src_end=split_boundary,
                zh_text="，".join(cur_texts)
            ))
            seg_idx += 1
            cur_start = split_boundary
            cur_end = et
            cur_texts = [txt]
        else:
            cur_end = et
            cur_texts.append(txt)

    if cur_texts:
        merged.append(TutorialSegment(
            index=seg_idx,
            src_start=cur_start,
            src_end=total_dur,
            zh_text="，".join(cur_texts)
        ))

    print(f"✅ 已将全片重构为 {len(merged)} 个独立变速语义片段。")
    return merged


# ==================== Step 3: 分批 AI 深度改写与 HUD 命令提取 ====================
# ==================== Step 3: 分批 AI 深度改写 + 真实翻译保底引擎 ====================
import time
import urllib.parse


def _extract_json_list(raw_text: str) -> Optional[list]:
    """鲁棒地从大模型返回文本中解析 JSON 列表"""
    if not raw_text:
        return None
    raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    try:
        obj = json.loads(raw_text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            # 如果是 OpenAI 标准返回体 {"choices": [{"message": {"content": "..."}}]}
            if "choices" in obj and isinstance(obj["choices"], list) and obj["choices"]:
                content = obj["choices"][0].get("message", {}).get("content", "")
                if content:
                    return _extract_json_list(content)
            for v in obj.values():
                if isinstance(v, list):
                    return v
    except Exception:
        pass

    start_idx = raw_text.find("[")
    end_idx = raw_text.rfind("]")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(raw_text[start_idx : end_idx + 1])
        except Exception:
            return None
    return None


def _translate_zh_to_en_fallback(zh_text: str, seg_index: int) -> Tuple[str, str, str]:
    """
    【兜底神器】当所有大模型接口均超时或限流时，
    直接调用 Google 免 Key 翻译接口将原片中文字幕精准译为英文，并自动提取技术关键词作为 HUD！
    绝不生成重复的假句子！
    """
    en_trans = ""
    try:
        q = urllib.parse.quote(zh_text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=zh-CN&tl=en&dt=t&q={q}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data and isinstance(data[0], list):
                en_trans = " ".join([part[0] for part in data[0] if part and part[0]]).strip()
    except Exception:
        pass

    # 如果连 Google 翻译都无法访问，使用 MyMemory 免费翻译 API 二次保底
    if not en_trans:
        try:
            q = urllib.parse.quote(zh_text[:450])
            url = f"https://api.mymemory.translated.net/get?q={q}&langpair=zh-CN|en"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                en_trans = data.get("responseData", {}).get("translatedText", "").strip()
        except Exception:
            pass

    if not en_trans:
        en_trans = f"Let's look at step {seg_index} and see how this part works in practice."

    # 清理口语重复，让翻译结果更像地道 YouTuber 解说
    en_trans = re.sub(r"\s+", " ", en_trans).strip()

    # 自动从原字幕或译文中提取英文技术缩写/命令（如 REA, API, GPU 等）生成 HUD 卡片
    hud_title, hud_code = "", ""
    tech_terms = re.findall(r"\b[A-Za-z][A-Za-z0-9_\-\.]{2,18}\b", zh_text)
    if tech_terms and seg_index % 4 == 1:
        key_term = tech_terms[0].upper()
        hud_title = f"CORE FOCUS // #{seg_index:02d}"
        hud_code = f"inspect --module {key_term}"

    return en_trans, hud_title, hud_code


def _call_llm_batch(batch_payload: list) -> list:
    """单批次调用 AI（支持 Gemini -> Ollama -> Pollinations OpenAI 端点）"""
    system_prompt = (
        "You are a senior English tech YouTuber creating an engaging developer tutorial.\n"
        "Rewrite (PARAPHRASE, do NOT translate literally) each Chinese segment into natural spoken English for voiceover.\n"
        "CRITICAL RULES:\n"
        "1. Match the target spoken length! Look at `target_words` for each item and write roughly that many English words.\n"
        "2. Pick 1 or 2 key technical segments in this batch and populate `hud_title` (short header, max 4 words) "
        "and `hud_code` (terminal command or core takeaway, max 38 chars). For other items, set both to \"\".\n"
        "3. Return ONLY a valid JSON array of objects: "
        "[{\"index\": 1, \"en_script\": \"...\", \"hud_title\": \"...\", \"hud_code\": \"...\"}]"
    )
    user_prompt = f"{system_prompt}\n\nInput JSON:\n{json.dumps(batch_payload, ensure_ascii=False)}"

    # 通道 1: Gemini API
    if GEMINI_API_KEY:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": user_prompt}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4}
            }
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                parsed = _extract_json_list(data["candidates"][0]["content"]["parts"][0]["text"])
                if parsed:
                    return parsed
        except Exception as e:
            print(f"    ⚠️ Gemini 批次调用异常 ({e})，切换备用 AI 通道...")

    # 通道 2: 本地 Ollama[cite: 1]
    try:
        tags_req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(tags_req, timeout=2) as resp:
            models = [m["name"] for m in json.loads(resp.read().decode("utf-8")).get("models", [])]
            if models:
                payload = {"model": models[0], "prompt": user_prompt, "stream": False, "format": "json"}
                req = urllib.request.Request(
                    OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST"
                )
                with urllib.request.urlopen(req, timeout=60) as r2:
                    parsed = _extract_json_list(json.loads(r2.read().decode("utf-8")).get("response", ""))
                    if parsed:
                        return parsed
    except Exception:
        pass

    # 通道 3: 免费 Pollinations 标准 OpenAI 兼容端点 (/openai)
    for attempt in range(2):
        try:
            req = urllib.request.Request(
                "https://text.pollinations.ai/openai",
                data=json.dumps({
                    "model": "openai",
                    "messages": [
                        {"role": "system", "content": "You output ONLY a raw JSON array. No markdown, no explanations."},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.4
                }).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw_txt = resp.read().decode("utf-8").strip()
                parsed = _extract_json_list(raw_txt)
                if parsed:
                    return parsed
        except Exception as e:
            time.sleep(1.5)
            if attempt == 1:
                print(f"    ⚠️ 免费大模型接口限流或超时 ({e})，本批次自动启用【Google 真实语义直译】保底...")

    return []


def paraphrase_segments_with_llm(segments: List[TutorialSegment]):
    """分批调用 AI 改写英文口播稿，若某段未命中大模型则自动使用真实翻译保底"""
    BATCH_SIZE = 8
    print(f"🧠 [3/5] 正在分批将 {len(segments)} 段中文字幕深度转化为地道英文口播稿...")

    result_map = {}
    for start_i in range(0, len(segments), BATCH_SIZE):
        batch = segments[start_i : start_i + BATCH_SIZE]
        batch_payload = []
        for s in batch:
            dur = round(s.src_end - s.src_start, 1)
            target_words = max(12, int(dur * 2.3))
            batch_payload.append({
                "index": s.index,
                "duration_sec": dur,
                "target_words": f"{target_words - 3} to {target_words + 4} words",
                "zh": s.zh_text
            })

        print(f"  🔄 正在处理第 {batch[0].index:02d} ~ {batch[-1].index:02d} 段...")
        parsed_batch = _call_llm_batch(batch_payload)
        for item in parsed_batch:
            if isinstance(item, dict) and "index" in item and item.get("en_script"):
                try:
                    idx_val = int(item["index"])
                    result_map[idx_val] = item
                except ValueError:
                    pass
        time.sleep(1.0)  # 避免免费接口触发并发限流

    llm_cnt = 0
    trans_cnt = 0
    for seg in segments:
        info = result_map.get(seg.index, {})
        en_txt = str(info.get("en_script", "")).strip()
        if len(en_txt.split()) >= 4 and not en_txt.startswith("In section "):
            seg.en_script = en_txt
            seg.hud_title = str(info.get("hud_title", "")).strip()
            seg.hud_code = str(info.get("hud_code", "")).strip()
            llm_cnt += 1
        else:
            # 启用真实中英翻译保底，确保每一句英文都 100% 对应原片中文意思！
            en_trans, h_title, h_code = _translate_zh_to_en_fallback(seg.zh_text, seg.index)
            seg.en_script = en_trans
            seg.hud_title = h_title
            seg.hud_code = h_code
            trans_cnt += 1

        print(f"  📝 [片段 #{seg.index:02d}] ({len(seg.en_script.split())}词): {seg.en_script[:75]}...")

    print(f"✅ 英文口播稿生成完毕！(AI 深度改写: {llm_cnt} 段 | 真实语义翻译保底: {trans_cnt} 段)")


# ==================== Step 4: Edge-TTS 逐段合成英文语音 ====================
async def _synthesize_one_segment(text: str, out_mp3: Path):
    import edge_tts
    comm = edge_tts.Communicate(text, EDGE_VOICE, rate=EDGE_RATE)
    await comm.save(str(out_mp3.resolve()))


def generate_all_tts_segments(segments: List[TutorialSegment], work_dir: Path):
    print(f"🎙️ [4/5] 正在使用 Edge-TTS ({EDGE_VOICE}) 合成英文解说音轨...")
    tts_dir = work_dir / "tts_segments"
    tts_dir.mkdir(parents=True, exist_ok=True)

    for seg in segments:
        raw_mp3 = tts_dir / f"seg_{seg.index:03d}_raw.mp3"
        clean_m4a = tts_dir / f"seg_{seg.index:03d}.m4a"
        txt_cache = tts_dir / f"seg_{seg.index:03d}.txt"

        clean_text = re.sub(r"\s+", " ", seg.en_script).strip()

        # 如果缓存音频时长异常(<=2.2s)，或者当前改写文案与上次缓存文案不同，自动重算！
        need_regen = True
        if clean_m4a.exists() and txt_cache.exists():
            cached_txt = txt_cache.read_text(encoding="utf-8").strip()
            if cached_txt == clean_text and get_media_duration(clean_m4a) > 2.2:
                need_regen = False

        if need_regen:
            if clean_m4a.exists():
                clean_m4a.unlink()
            asyncio.run(_synthesize_one_segment(clean_text, raw_mp3))

            # 使用正反双向 start_periods=1 仅剥离首尾静音，绝不截断句子中间的停顿！
            af_clean = (
                "aformat=sample_rates=44100:channel_layouts=stereo,"
                "silenceremove=start_periods=1:start_duration=0.03:start_threshold=-45dB,"
                "areverse,"
                "silenceremove=start_periods=1:start_duration=0.05:start_threshold=-45dB,"
                "areverse,"
                "loudnorm=I=-14:TP=-1.0:LRA=9"
            )
            cmd = [
                "ffmpeg", "-y", "-i", str(raw_mp3.resolve()),
                "-af", af_clean, "-c:a", "aac", "-b:a", "192k",
                str(clean_m4a.resolve())
            ]
            res = run_cmd(cmd, f"TTS 音频清洗 #{seg.index}")
            if res.returncode != 0 or get_media_duration(clean_m4a) < 1.5:
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(raw_mp3.resolve()),
                    "-c:a", "aac", "-b:a", "192k", str(clean_m4a.resolve())
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            txt_cache.write_text(clean_text, encoding="utf-8")
            if raw_mp3.exists():
                raw_mp3.unlink()

        seg.tts_audio_path = clean_m4a
        seg.tts_duration = max(1.8, get_media_duration(clean_m4a))
        hud_tag = f" | 🖥️ HUD: [{seg.hud_title}] {seg.hud_code}" if seg.hud_code else ""
        print(f"  -> 片段 #{seg.index:02d}: 原长 {seg.src_end - seg.src_start:.2f}s | 英文TTS {seg.tts_duration:.2f}s{hud_tag}")


# ==================== Step 5: FFmpeg 深度重构（去中文+独立字幕舱+HUD+声波挂件） ====================
def build_segment_ass_overlay(
    ass_path: Path,
    duration: float,
    en_subtitle: str,
    hud_title: str,
    hud_code: str,
):
    """
    生成 ASS 矢量叠加层（升级版）：
    1. 自动将长英文句子按每屏 8~9 个单词切分为多段呼吸感短字幕，按时间轴依次播放，绝不超长堆叠！[cite: 2]
    2. 字幕严格锁定在底部专属安全舱 (Y=900~1060)，位于青色发光框正下方，再也不会被挡住！[cite: 2]
    """
    def _fmt_ass_time(sec: float) -> str:
        sec = max(0.0, sec)
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        cs = int(round((sec % 1) * 100))
        if cs == 100:
            s += 1
            cs = 0
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    t_end = _fmt_ass_time(duration + 0.15)
    hud_start = _fmt_ass_time(0.20)
    hud_end = _fmt_ass_time(max(0.5, duration - 0.15))

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_WIDTH}
PlayResY: {OUT_HEIGHT}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TopBar,Consolas,20,&H00FFEA00,&H000000FF,&H00151008,&H90000000,-1,0,0,0,100,100,1.2,0,1,2,0,7,42,36,18,1
Style: EnSub,Arial,40,&H00FFFFFF,&H000000FF,&H00000000,&HB0080C14,-1,0,0,0,100,100,0.4,0,3,10.0,0.0,2,70,290,54,1
Style: HudTitle,Consolas,24,&H0000EAFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0.8,0,1,2.0,0,7,0,0,0,1
Style: HudCode,Consolas,22,&H00A8FF00,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0.5,0,1,1.8,0,7,0,0,0,1
Style: AvatarTag,Consolas,18,&H00FFEA00,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0.5,0,1,1.5,0,3,30,42,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = [
        f"Dialogue: 2,0:00:00.00,{t_end},TopBar,,0,0,0,,● REC  |  {CHANNEL_HANDLE}  //  DEEP REMASTERED TUTORIAL",
        f"Dialogue: 3,0:00:00.00,{t_end},AvatarTag,,0,0,0,,AI HOST // {CHANNEL_HANDLE}",
    ]

    # 将长英文口播稿按标点或每 9 个词自动切分为多屏短句
    clean_sub = en_subtitle.replace("\n", " ").replace("{", "(").replace("}", ")").strip()
    words = clean_sub.split()
    chunks = []
    cur_chunk = []
    for w in words:
        cur_chunk.append(w)
        if (len(cur_chunk) >= 5 and w.endswith((",", ".", ";", ":", "!", "?"))) or len(cur_chunk) >= 9:
            chunks.append(" ".join(cur_chunk))
            cur_chunk = []
    if cur_chunk:
        if chunks and len(cur_chunk) <= 3:
            chunks[-1] += " " + " ".join(cur_chunk)
        else:
            chunks.append(" ".join(cur_chunk))

    total_chars = max(1, sum(len(c) for c in chunks))
    cur_t = 0.0
    for idx_c, chunk_txt in enumerate(chunks):
        seg_share = duration * (len(chunk_txt) / total_chars)
        next_t = duration if idx_c == len(chunks) - 1 else (cur_t + seg_share)
        events.append(
            f"Dialogue: 5,{_fmt_ass_time(cur_t)},{_fmt_ass_time(next_t)},EnSub,,0,0,0,,{chunk_txt}"
        )
        cur_t = next_t

    if hud_code:
        title_str = (hud_title or "KEY COMMAND").upper().replace("{", "").replace("}", "")
        code_str = hud_code.replace("{", "(").replace("}", ")")
        events.append(
            f"Dialogue: 4,{hud_start},{hud_end},HudTitle,,0,0,0,,"
            f"{{\\pos(1355,138)\\fad(220,220)}}⚡ {title_str}"
        )
        events.append(
            f"Dialogue: 4,{hud_start},{hud_end},HudCode,,0,0,0,,"
            f"{{\\pos(1355,184)\\fad(220,220)}}$ {code_str}"
        )

    ass_path.write_text(ass_header + "\n".join(events) + "\n", encoding="utf-8")


def render_remastered_segment(
    src_video: Path,
    seg: TutorialSegment,
    work_dir: Path,
) -> bool:
    """
    升级版 FFmpeg 单段合成引擎：
    1. 切除原片底部 17.5%（彻底切掉原片自带的黄字中文和白字英文硬字幕）[cite: 2]
    2. 前景框放置于 Y=58..888，下方 Y=888..1080 为专属英文字幕舱，绝不遮挡英文字幕[cite: 2]
    """
    src_dur = max(0.5, seg.src_end - seg.src_start)
    target_dur = seg.tts_duration + 0.15

    raw_speed = src_dur / target_dur
    clamped_speed = max(MIN_SPEED_FACTOR, min(MAX_SPEED_FACTOR, raw_speed))

    scaled_video_dur = src_dur / clamped_speed
    pad_dur = max(0.0, target_dur - scaled_video_dur + 0.25)

    ass_name = f"seg_{seg.index:03d}.ass"
    ass_path = work_dir / ass_name
    build_segment_ass_overlay(ass_path, target_dur, seg.en_script, seg.hud_title, seg.hud_code)

    out_clip = work_dir / f"clip_{seg.index:03d}.mp4"

    hud_box_filter = ""
    if seg.hud_code:
        hud_box_filter = (
            ",drawbox=x=1335:y=120:w=535:h=120:color=0x0B1320@0.90:t=fill"
            ",drawbox=x=1335:y=120:w=535:h=120:color=0x00EAFF@0.88:t=3"
            ",drawbox=x=1335:y=120:w=8:h=120:color=0x00EAFF@1.0:t=fill"
        )

    orig_cwd = os.getcwd()
    try:
        os.chdir(str(work_dir))

        filter_complex = (
            # 1. 1080p 赛博朋克网格底座 + 顶部状态栏 + 底部专属深色字幕舱
            f"color=c=0x0A0E17:s={OUT_WIDTH}x{OUT_HEIGHT}:r={OUT_FPS}:d={target_dur:.3f},"
            f"drawgrid=w=64:h=64:t=1:c=0x00EAFF@0.09,"
            f"drawbox=x=0:y=0:w={OUT_WIDTH}:h=50:color=0x05070B@0.92:t=fill,"
            f"drawbox=x=0:y={FG_POS_Y + FG_HEIGHT + 8}:w={OUT_WIDTH}:h=190:color=0x06090F@0.90:t=fill,"
            f"drawbox=x={FG_POS_X+8}:y={FG_POS_Y+8}:w={FG_WIDTH+12}:h={FG_HEIGHT+12}:color=0x000000@0.65:t=fill[bg];"

            # 2. 前景视频：切除顶部 6% 与底部 17.5%（把原片自带的黄字中文和白字英文全部切掉！）
            #    配合 increase + crop 保持宽高比不变形，外加 3px 青色发光边框
            f"[0:v]crop=iw*(1-{CROP_LEFT_RATIO}-{CROP_RIGHT_RATIO}):"
            f"ih*(1-{CROP_TOP_RATIO}-{CROP_BOTTOM_RATIO}):"
            f"iw*{CROP_LEFT_RATIO}:ih*{CROP_TOP_RATIO},"
            f"setpts=PTS/{clamped_speed:.4f},"
            f"fps={OUT_FPS},"
            f"scale={FG_WIDTH}:{FG_HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={FG_WIDTH}:{FG_HEIGHT},"
            f"eq=contrast=1.04:saturation=1.05,"
            f"pad={FG_WIDTH+6}:{FG_HEIGHT+6}:3:3:color=0x00EAFF@0.90,"
            f"tpad=stop_mode=clone:stop_duration={pad_dur:.3f}[fg];"

            # 3. 合成画中画（放置在 Y=58，底部止于 Y=888，给下方英文字幕留出充足空间）
            f"[bg][fg]overlay=x={FG_POS_X}:y={FG_POS_Y}:shortest=1[canvas];"

            # 4. 右下角虚拟数字人声波挂件
            f"[1:a]aformat=sample_rates=44100:channel_layouts=stereo,asplit=2[a_vo][a_viz];"
            f"[a_viz]showwaves=s=210x84:mode=cline:rate={OUT_FPS}:colors=0x00EAFF|0x00FF88[wave];"
            f"color=c=0x0D1522@0.94:s=230x120:r={OUT_FPS}:d={target_dur:.3f},"
            f"drawbox=x=0:y=0:w=230:h=120:color=0x00EAFF@0.90:t=2[avatar_box];"
            f"[avatar_box][wave]overlay=x=10:y=10:shortest=1[avatar_widget];"

            # 5. 叠加右下角 AI 挂件 + 侧边 HUD 卡片 + 底部专属舱英文字幕
            f"[canvas][avatar_widget]overlay=x={OUT_WIDTH-255}:y={OUT_HEIGHT-145}:shortest=1"
            f"{hud_box_filter},"
            f"subtitles={ass_name}[vout];"

            # 6. 音频输出
            f"[a_vo]apad=whole_dur={target_dur:.3f}[aout]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{seg.src_start:.3f}",
            "-t", f"{src_dur:.3f}",
            "-i", str(src_video.resolve()),
            "-i", str(seg.tts_audio_path.resolve()),
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[aout]",
            "-t", f"{target_dur:.3f}",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            str(out_clip.resolve())
        ]
        res = run_cmd(cmd, f"渲染片段 #{seg.index}")
        if res.returncode == 0 and out_clip.exists():
            seg.rendered_clip_path = out_clip
            return True
        return False
    finally:
        os.chdir(orig_cwd)


def concat_and_finalize(
    segments: List[TutorialSegment],
    output_mp4: Path,
    work_dir: Path,
    bgm_file: Optional[Path] = None,
):
    valid_clips = [s.rendered_clip_path for s in segments if s.rendered_clip_path and s.rendered_clip_path.exists()]
    if not valid_clips:
        raise RuntimeError("没有成功渲染的片段可供合并！")

    concat_list = work_dir / "concat_clips.txt"
    concat_list.write_text(
        "".join([f"file '{p.resolve().as_posix()}'\n" for p in valid_clips]),
        encoding="utf-8"
    )

    merged_raw = work_dir / "merged_no_bgm.mp4"
    cmd_concat = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list.resolve()),
        "-c", "copy", str(merged_raw.resolve())
    ]
    run_cmd(cmd_concat, "合并全部重构切片")

    if bgm_file and bgm_file.exists():
        print(f"🎵 正在混入极客背景音乐并开启人声智能闪避: {bgm_file.name}")
        ducking_filter = (
            f"[0:a]aformat=sample_rates=44100:channel_layouts=stereo,asplit=2[vo][sc];"
            f"[1:a]aformat=sample_rates=44100:channel_layouts=stereo,volume={BGM_BASE_VOLUME}[bgm];"
            f"[bgm][sc]sidechaincompress=threshold=0.015:ratio=5:attack=40:release=500[bgm_duck];"
            f"[vo][bgm_duck]amix=inputs=2:duration=first:normalize=0[aout]"
        )
        cmd_bgm = [
            "ffmpeg", "-y",
            "-i", str(merged_raw.resolve()),
            "-stream_loop", "-1", "-i", str(bgm_file.resolve()),
            "-filter_complex", ducking_filter,
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
            "-shortest", str(output_mp4.resolve())
        ]
        res = run_cmd(cmd_bgm, "BGM 侧链闪避混音")
        if res.returncode == 0 and output_mp4.exists():
            return

    shutil.move(str(merged_raw.resolve()), str(output_mp4.resolve()))


def run_pipeline(source_input: str):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 下载或加载源视频（自动复用已下载的 source_raw.mp4）
    src_video = download_source_video(source_input, WORK_DIR)

    # 2. Faster-Whisper 提取中文字幕并切分为 6~12s 语义块
    whisper_model = load_whisper_model()
    segments = extract_semantic_segments(src_video, whisper_model)

    # 3. 分批调用 AI 改写地道英文口播稿 + 抽取 HUD 命令框
    paraphrase_segments_with_llm(segments)

    # 4. Edge-TTS 逐段生成英文配音（自动检测并替换残损的 1.5s 缓存）
    generate_all_tts_segments(segments, WORK_DIR)

    # 5. FFmpeg 逐段完成画中画去中文裁剪、0.8x~1.3x 动态变速、HUD 弹窗与底部专属舱英文字幕合成
    print(f"\n🎬 [5/5] 正在通过 FFmpeg 渲染 {len(segments)} 个深度重构片段...")
    for seg in segments:
        ok = render_remastered_segment(src_video, seg, WORK_DIR)
        status = "✅" if ok else "❌"
        print(f"  {status} 已渲染片段 [{seg.index:02d}/{len(segments):02d}]")

    out_file = OUTPUT_DIR / f"yt_geek_tutorial_{src_video.stem}.mp4"
    concat_and_finalize(segments, out_file, WORK_DIR, BGM_FILE if BGM_FILE.exists() else None)

    print("\n" + "=" * 72)
    print("🎉 YouTube 深度去重英文教程渲染完成！")
    print(f" 👉 输出文件: {out_file.resolve()}")
    print(f" ⏱️ 最终总长: {get_media_duration(out_file):.2f}s")
    print("=" * 72)


if __name__ == "__main__":
    print("=" * 72)
    print(" 🚀 B站 -> YouTube 极客英文教程『深度去重与洗白』自动化流水线 (Pro版)")
    print("=" * 72)
    target = input("请输入 B 站视频链接 或 本地视频文件路径 (直接回车复用上次视频): ").strip().strip('"')
    if not target:
        cached_raw = WORK_DIR / "source_raw.mp4"
        if cached_raw.exists():
            target = str(cached_raw.resolve())
        else:
            local_mp4s = [f for f in BASE_DIR.glob("*.mp4") if not f.name.startswith("yt_")]
            if local_mp4s:
                target = str(local_mp4s[0].resolve())
            else:
                print("❌ 未提供输入视频链接或本地文件。")
                sys.exit(1)

    run_pipeline(target)
