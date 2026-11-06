#!/usr/bin/env python3
"""video-translator: 为本地视频自动生成双语(原声->中文)SRT字幕.

流程: 视频 -> PyAV提取音频 -> faster-whisper识别 -> argos-translate翻译 -> 双语SRT
所有模型与翻译包均从国内镜像下载, 无需科学上网.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from study_ai import enrich_chapters
from study_pack import make_chapters, write_study_pack

SCRIPT_DIR = Path(__file__).resolve().parent


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_env_file(SCRIPT_DIR / ".env")

# ---- 镜像配置(必须在导入 huggingface_hub 之前设置) ----
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

GH_PROXY = os.environ.get("VT_GH_PROXY", "https://gh-proxy.com/")
ARGOS_INDEX_URL = (
    "https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json"
)

from argostranslate import package, translate  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402


def _patch_stanza_offline() -> None:
    """禁止 stanza 联网检查资源(模型已随翻译包本地提供), 否则弱网环境会无限挂起."""
    try:
        from stanza.pipeline.core import Pipeline as _SP

        _orig = _SP.__init__

        def _init(self, *a, **kw):
            kw.setdefault("download_method", None)
            try:
                return _orig(self, *a, **kw)
            except KeyError:
                # 某些翻译包的本地 resources.json 不完整, 降级为预分词模式
                kw["tokenize_pretokenized"] = True
                return _orig(self, *a, **kw)

        _SP.__init__ = _init
    except Exception:
        pass


_patch_stanza_offline()


def _patch_argos_sentencizer() -> None:
    """字幕片段本就是短句, 用轻量规则分句替代 stanza,
    规避部分翻译包本地 stanza 资源不完整导致的崩溃."""
    try:
        from argostranslate.sbd import StanzaSentencizer

        def _split(self, text):
            text = (text or "").strip()
            if not text:
                return []
            if len(text) <= 120:
                return [text]
            parts = [p.strip() for p in re.split(r"(?<=[.!?。！？…])\s*", text)]
            return [p for p in parts if p] or [text]

        StanzaSentencizer.split_sentences = _split
    except Exception:
        pass


_patch_argos_sentencizer()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def proxied(url: str) -> str:
    """把 GitHub 系 URL 改写为走代理."""
    if any(
        host in url
        for host in ("github.com", "githubusercontent.com", "githubassets.com")
    ):
        return GH_PROXY + url
    return url


# ---------------------------------------------------------------- whisper ---

_MODEL_SIZES = {
    "tiny": "~75MB, 最快, 准确率低",
    "base": "~145MB, 快",
    "small": "~480MB, 推荐, 平衡",
    "medium": "~1.5GB, 慢, 更准",
    "large-v3": "~3GB, 最准, 最慢",
}


def load_whisper(model_size: str, device: str) -> WhisperModel:
    log(f"加载 Whisper 模型 [{model_size}] (首次运行会自动下载)")
    return WhisperModel(model_size, device=device, compute_type="int8")


def decode_clip(media_path: str, start: float, duration: float):
    """从视频中截取 [start, start+duration) 的音频, 返回16k单声float32数组."""
    import av
    import numpy as np

    c = av.open(media_path)
    st = c.streams.audio[0]
    res = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    chunks = []
    c.seek(int(start / st.time_base), stream=st)
    end = start + duration
    for frame in c.decode(audio=0):
        t = float(frame.pts * st.time_base) if frame.pts is not None else 0.0
        if t < start - 1.0:
            continue
        if t >= end:
            break
        for rf in res.resample(frame):
            chunks.append(rf.to_ndarray().reshape(-1))
    c.close()
    if not chunks:
        sys.exit("错误: 该时间段没有音频")
    return np.concatenate(chunks).astype(np.float32)


def transcribe(model: WhisperModel, media_path: str, cache: Path | None = None):
    if cache is not None and cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            log(f"命中识别缓存 {cache.name} ({len(data['segs'])} 段), 跳过重新识别")
            return data["lang"], [tuple(x) for x in data["segs"]]
        except Exception as e:
            log(f"缓存读取失败, 重新识别: {e}")
    log("提取音频并开始语音识别 ...")
    segments, info = model.transcribe(media_path, vad_filter=True, beam_size=5)
    lang = info.language
    log(f"检测到源语言: {lang}  (概率 {info.language_probability:.0%})")
    segs = []
    with tempfile.TemporaryDirectory() as td:
        # segments 是生成器, 先物化出来
        for s in segments:
            segs.append((s.start, s.end, s.text.strip()))
            print(
                f"\r  已识别 {len(segs)} 段 / {s.end:.0f}s",
                end="",
                flush=True,
            )
    print()
    if not segs:
        sys.exit("错误: 没有识别到任何语音内容")
    log(f"识别完成, 共 {len(segs)} 段")
    if cache is not None:
        try:
            cache.write_text(
                json.dumps({"lang": lang, "segs": segs}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            log(f"缓存写入失败(不影响结果): {e}")
    return lang, segs


# ------------------------------------------------------------------ argos ---


def _http_get(url: str, timeout: int = 60):
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "vt"})
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_argos_index() -> list:
    for url in (ARGOS_INDEX_URL, proxied(ARGOS_INDEX_URL)):
        try:
            with _http_get(url) as r:
                return json.load(r)
        except Exception as e:
            log(f"索引获取失败({url.split('/')[2]}): {e}")
    sys.exit("错误: 无法下载翻译包索引")


def download_argos_model(name: str, url: str) -> Path:
    dest = Path(tempfile.gettempdir()) / f"vt_{name}.argosmodel"
    log(f"下载翻译模型 {name} ...")
    for attempt, target in enumerate((url, proxied(url))):
        try:
            with _http_get(target, timeout=300) as r, open(dest, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(
                            f"\r  {done / 1e6:.0f}/{total / 1e6:.0f} MB",
                            end="",
                            flush=True,
                        )
            print()
            return dest
        except Exception as e:
            print()
            log(f"下载失败(第{attempt + 1}次尝试): {e}")
    sys.exit(f"错误: 翻译模型 {name} 下载失败")


def installed_pair_codes() -> set[tuple[str, str]]:
    return {(p.from_code, p.to_code) for p in package.get_installed_packages()}


def _vkey(v: str):
    try:
        return tuple(int(x) for x in v.split("."))
    except ValueError:
        return (0,)


def ensure_translation_chain(src: str) -> None:
    """确保存在 src->zh 的可用链路; 必要时经英语中转并自动下载模型."""

    def install_pair(a: str, b: str) -> bool:
        if (a, b) in installed_pair_codes():
            return True
        index = fetch_argos_index()
        cand = [p for p in index if p["from_code"] == a and p["to_code"] == b]
        if not cand:
            log(f"索引中没有 {a}->{b} 的翻译模型")
            return False
        pkg = max(cand, key=lambda p: _vkey(p.get("package_version", "0")))
        path = download_argos_model(pkg["code"], pkg["links"][0])
        package.install_from_path(path)
        path.unlink(missing_ok=True)
        log(f"已安装翻译模型 {a}->{b}")
        return True

    if src == "zh":
        return
    if (src, "zh") in installed_pair_codes():
        return
    if not install_pair(src, "zh"):
        # 没有直达包(如小语种), 经英语中转
        if not install_pair(src, "en") or not install_pair("en", "zh"):
            sys.exit(f"错误: 无法建立 {src}->中文 的翻译链路")


def translate_segments(src: str, segs: list) -> list:
    if src == "zh":
        log("源语言已是中文, 跳过翻译")
        return [(t, "") for _, _, t in segs]

    ensure_translation_chain(src)
    pairs = {(p.from_code, p.to_code): p for p in package.get_installed_packages()}
    direct = pairs.get((src, "zh"))

    def to_zh(text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if direct is not None:
            return translate.translate(text, src, "zh")
        en = translate.translate(text, src, "en")
        return translate.translate(en, "en", "zh")

    out = []
    t0 = time.time()
    for i, (_, _, text) in enumerate(segs, 1):
        zh = to_zh(text)
        out.append((text, zh))
        print(f"\r  翻译进度 {i}/{len(segs)}", end="", flush=True)
    dt = time.time() - t0
    print()
    log(f"翻译完成, 用时 {dt:.0f}s")
    return out


# ------------------------------------------------------------- api 引擎 ----

API_BASE = os.environ.get("VT_API_BASE", "https://api.deepseek.com")
API_KEY = os.environ.get("VT_API_KEY", "")
API_MODEL = os.environ.get("VT_API_MODEL", "deepseek-chat")
BATCH_SIZE = int(os.environ.get("VT_BATCH_SIZE", "20"))
BATCH_DELAY = float(os.environ.get("VT_BATCH_DELAY", "15"))

_LANG_NAMES = {
    "en": "英语", "ja": "日语", "ko": "韩语", "fr": "法语", "de": "德语",
    "ru": "俄语", "es": "西班牙语", "it": "意大利语", "pt": "葡萄牙语",
    "th": "泰语", "vi": "越南语", "ar": "阿拉伯语", "zh": "中文",
}


def _api_url() -> str:
    base = API_BASE.rstrip("/")
    return base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")


def _chat(messages: list, timeout: int = 180) -> str:
    body = json.dumps({
        "model": API_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }).encode("utf-8")
    req = urllib.request.Request(
        _api_url(),
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"]


def _parse_json_array(text: str) -> list | None:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
        return arr
    return None


def translate_batch_api(texts: list, src_name: str) -> list | None:
    sys_prompt = (
        f"你是影视字幕翻译专家。把用户给出的JSON数组里的{src_name}台词翻译成简体中文。"
        "要求：口语化、符合中文表达习惯、简洁(适合字幕)、人名音译、保留原文语气。"
        f"仅返回一个与输入等长的JSON字符串数组，不要输出任何其他内容。"
    )
    user_prompt = json.dumps(texts, ensure_ascii=False)
    for attempt in range(3):
        try:
            content = _chat([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ])
            arr = _parse_json_array(content)
            if arr is not None and len(arr) == len(texts):
                return [x.strip() for x in arr]
            log(f"  API返回格式异常, 重试({attempt + 1}/3)")
        except Exception as e:
            if "403" in str(e):
                wait = 45 * (attempt + 1)
                log(f"  被限流(403), 等待{wait}s后重试({attempt + 1}/3)")
                time.sleep(wait)
            else:
                log(f"  API请求失败: {e} 重试({attempt + 1}/3)")
                time.sleep(2 * (attempt + 1))
    return None


def translate_segments_api(lang: str, segs: list, allow_fallback: bool = True) -> list:
    src_name = _LANG_NAMES.get(lang, lang)
    log(f"使用 API 引擎翻译 ({API_MODEL}), 批大小 {BATCH_SIZE}")
    result = []
    batches = [segs[i:i + BATCH_SIZE] for i in range(0, len(segs), BATCH_SIZE)]
    t0 = time.time()
    done = 0
    for bi, batch in enumerate(batches, 1):
        if bi > 1:
            time.sleep(BATCH_DELAY)
        texts = [t for _, _, t in batch]
        zh_list = translate_batch_api(texts, src_name)
        if zh_list is None:
            if allow_fallback:
                log(f"  第{bi}批 API 翻译失败, 本地引擎兜底")
                ensure_translation_chain(lang)
                direct = lang != "zh"
                zh_list = []
                for t in texts:
                    if not t.strip():
                        zh_list.append("")
                        continue
                    if direct:
                        zh_list.append(translate.translate(t, lang, "zh"))
                    else:
                        en = translate.translate(t, lang, "en")
                        zh_list.append(translate.translate(en, "en", "zh"))
            else:
                sys.exit(f"错误: API 翻译失败(第{bi}批)")
        result.extend(zip(texts, zh_list))
        done += len(batch)
        print(f"\r  进度 {done}/{len(segs)}", end="", flush=True)
    print()
    log(f"API 翻译完成, 用时 {time.time() - t0:.0f}s")
    return result


def resolve_engine(choice: str) -> str:
    if choice == "local":
        return "local"
    if choice == "api":
        if not API_KEY:
            sys.exit("错误: 未配置 VT_API_KEY, 无法使用 api 引擎")
        return "api"
    if API_KEY:
        return "api"
    log("未配置 VT_API_KEY, 自动使用本地离线引擎")
    return "local"


# ------------------------------------------------------------------- srt ----


def fmt_ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_bilingual_srt(pairs, segs, out_path: Path, no_original: bool) -> None:
    lines = []
    for idx, ((start, end, _), (orig, zh)) in enumerate(zip(segs, pairs), 1):
        body = orig
        if not no_original and zh:
            body = f"{orig}\n{zh}"
        elif zh:
            body = zh
        lines.append(f"{idx}\n{fmt_ts(start)} --> {fmt_ts(end)}\n{body}\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ main ----


def main() -> None:
    ap = argparse.ArgumentParser(
        description="为本地视频/音频自动生成 原文+中文 双语SRT字幕",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("media", nargs="+", help="视频或音频文件路径(支持多个)")
    ap.add_argument("-m", "--model", default="small", choices=_MODEL_SIZES,
                    help="Whisper 模型大小: " + "; ".join(f"{k}={v}" for k, v in _MODEL_SIZES.items()))
    ap.add_argument("--device", default="cpu", choices=["cpu", "auto"], help="推理设备")
    ap.add_argument("--zh-only", action="store_true", help="只输出中文字幕(不含原文)")
    ap.add_argument("--engine", default="auto", choices=["auto", "api", "local"],
                    help="翻译引擎: auto=有API Key用api否则local; api=大模型; local=离线")
    ap.add_argument("--fresh", action="store_true", help="忽略识别缓存强制重新识别")
    ap.add_argument("--study", action="store_true",
                    help="同时生成本地学习助手HTML（章节、字幕检索、时间戳笔记）")
    ap.add_argument("--ai-outline", default="auto", choices=["auto", "on", "off"],
                    help="AI提炼章节: auto=有API Key时启用; on=尝试启用; off=只用离线提炼")
    ap.add_argument("--start-sec", type=int, default=None,
                    help="只处理片段: 从第几秒开始(试看效果用)")
    ap.add_argument("--dur-sec", type=int, default=180,
                    help="片段时长(秒), 默认180, 配合 --start-sec 使用")
    args = ap.parse_args()

    files = []
    for p in args.media:
        pp = Path(p).expanduser().resolve()
        if not pp.exists():
            sys.exit(f"错误: 文件不存在 {pp}")
        files.append(pp)

    device = "auto" if args.device == "auto" else "cpu"
    model = load_whisper(args.model, device)
    engine = resolve_engine(args.engine)

    for f in files:
        print("\n" + "=" * 60)
        log(f"处理: {f.name}")
        if args.start_sec is not None:
            offset = float(args.start_sec)
            out = f.with_name(f"{f.stem}.片段{offset:.0f}-{offset + args.dur_sec:.0f}.srt")
            cache = f.with_name(f"{f.stem}.片段{offset:.0f}.segs.json")
            log(f"片段模式: 从 {offset:.0f}s 开始, 处理 {args.dur_sec}s")
            audio = decode_clip(str(f), offset, args.dur_sec)
            segments, info = model.transcribe(audio, vad_filter=True, beam_size=5)
            lang = info.language
            log(f"检测到源语言: {lang}  (概率 {info.language_probability:.0%})")
            segs = []
            for s in segments:
                segs.append((s.start, s.end, s.text.strip()))
                print(f"\r  已识别 {len(segs)} 段 / {s.end:.0f}s", end="", flush=True)
            print()
            if not segs:
                sys.exit("错误: 该时间段没有识别到语音")
            log(f"识别完成, 共 {len(segs)} 段")
            try:
                cache.write_text(json.dumps({"lang": lang, "segs": segs}, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                log(f"缓存写入失败(不影响结果): {e}")
        else:
            offset = 0.0
            out = f.with_suffix(".srt")
            lang, segs = transcribe(model, str(f), None if args.fresh else f.with_suffix(".segs.json"))
        segs = [(a + offset, b + offset, t) for a, b, t in segs]
        if engine == "api":
            pairs = translate_segments_api(lang, segs)
        else:
            pairs = translate_segments(lang, segs)
        write_bilingual_srt(pairs, segs, out, args.zh_only)
        log(f"完成 -> {out}")
        if args.study:
            chapters = make_chapters(segs, pairs)
            use_ai = args.ai_outline == "on" or (
                args.ai_outline == "auto" and bool(API_KEY)
            )
            if use_ai and not API_KEY:
                log("未配置 VT_API_KEY，AI章节提炼跳过，使用离线章节")
            elif use_ai:
                try:
                    log(f"使用 {API_MODEL} 提炼章节标题、知识点和复习题 ...")
                    chapters = enrich_chapters(chapters, segs, pairs, _chat)
                    log("AI章节提炼完成")
                except Exception as e:
                    log(f"AI章节提炼失败，已自动使用离线章节: {e}")
            study = write_study_pack(f, segs, pairs, chapters=chapters)
            log(f"学习助手 -> {study}（浏览器打开即可使用）")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已取消")
        sys.exit(130)
