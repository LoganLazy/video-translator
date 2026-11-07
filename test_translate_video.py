"""Legacy translate_video.py 回归测试. 只用标准库 + av; 重型依赖用桩模块注入."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import types
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# --- 先注入重型依赖桩, 再导入被测模块 ---
_faster_whisper = types.ModuleType("faster_whisper")
_faster_whisper.WhisperModel = object
sys.modules.setdefault("faster_whisper", _faster_whisper)
_argos = types.ModuleType("argostranslate")
_argos.package = SimpleNamespace(
    get_installed_packages=lambda: [], install_from_path=lambda p: None)
_argos.translate = SimpleNamespace(translate=lambda *a: "")
sys.modules.setdefault("argostranslate", _argos)
sys.modules.setdefault("argostranslate.package", _argos.package)
sys.modules.setdefault("argostranslate.translate", _argos.translate)

import translate_video as old


class FakeModel:
    """不调真实 Whisper, 只返回固定识别结果."""

    def __init__(self):
        self.calls = 0

    def transcribe(self, *args, **kwargs):
        self.calls += 1
        segments = [SimpleNamespace(start=0.0, end=1.0, text="Hello"),
                    SimpleNamespace(start=1.0, end=2.0, text="World")]
        info = SimpleNamespace(language="en", language_probability=0.99)
        return segments, info


class RaisingModel:
    """一旦被调用就失败, 用于验证缓存命中时不加载模型."""

    def transcribe(self, *args, **kwargs):
        raise AssertionError("模型不应在缓存命中时被调用")


class LegacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _record(self):
                cls.calls.append((self.path, self.headers.get("Authorization")))

            def _json(self, content):
                body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._record()
                if self.path.startswith("/redirect"):
                    self.send_response(302)
                    self.send_header("Location",
                                     "http://127.0.0.1:%d/target" % cls.server_b.server_port)
                    self.end_headers()
                    return
                if self.path.startswith("/unauthorized"):
                    self.send_response(401)
                    self.end_headers()
                    return
                if self.path.startswith("/oversized-header"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(3 * 1024 * 1024))
                    self.end_headers()
                    self.wfile.write(b"{}")
                    return
                if self.path.startswith("/oversized"):
                    body = json.dumps({"choices": [{"message": {"content": "x" * 3_000_000}}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                if self.path.startswith("/truncated"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", "100000")
                    self.end_headers()
                    self.wfile.write(b'{"choices": []}')
                    return
                if self.path.startswith("/model-big"):
                    self.send_response(200)
                    self.send_header("Content-Length", "100")
                    self.end_headers()
                    self.wfile.write(b"x" * 100)
                    return
                if self.path.startswith("/model"):
                    self.send_response(200)
                    self.send_header("Content-Length", "16")
                    self.end_headers()
                    self.wfile.write(b"fake-model-bytes")
                    return
                if self.path.startswith("/fail"):
                    self.send_response(500)
                    self.end_headers()
                    return
                self._json('["你好"]')

            def do_POST(self):
                self._record()
                if self.path.startswith("/redirect"):
                    self.send_response(302)
                    self.send_header("Location",
                                     "http://127.0.0.1:%d/target" % cls.server_b.server_port)
                    self.end_headers()
                    return
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                if self.path.startswith("/unauthorized"):
                    self.send_response(401)
                    self.end_headers()
                    return
                if self.path.startswith("/oversized-header"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(3 * 1024 * 1024))
                    self.end_headers()
                    self.wfile.write(b"{}")
                    return
                if self.path.startswith("/oversized"):
                    body = json.dumps({"choices": [{"message": {"content": "x" * 3_000_000}}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                if self.path.startswith("/truncated"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", "100000")
                    self.end_headers()
                    self.wfile.write(b'{"choices": []}')
                    return
                self._json('["你好"]')

        cls.server_a = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server_b = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread_a = threading.Thread(target=cls.server_a.serve_forever, daemon=True)
        cls.thread_b = threading.Thread(target=cls.server_b.serve_forever, daemon=True)
        cls.thread_a.start()
        cls.thread_b.start()
        cls.base_a = "http://127.0.0.1:%d" % cls.server_a.server_port
        cls.base_b = "http://127.0.0.1:%d" % cls.server_b.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server_a.shutdown()
        cls.server_b.shutdown()
        cls.server_a.server_close()
        cls.server_b.server_close()
        cls.thread_a.join(timeout=3)
        cls.thread_b.join(timeout=3)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="vt-legacy-test-")
        self.root = Path(self.directory.name)
        self.calls.clear()

    def tearDown(self):
        old.BATCH_SIZE = 20
        old.BATCH_DELAY = 15.0
        self.directory.cleanup()

    def _chat(self, path):
        with patch.object(old, "API_BASE", self.base_a + path), \
                patch.object(old, "API_KEY", "test-key"):
            return old._chat([{"role": "user", "content": "hi"}])

    def _wave(self, name="sample.wav"):
        import numpy as np
        path = self.root / name
        samples = np.concatenate([np.full(16000, value, dtype=np.int16) for value in [1000, 4000, 8000]])
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(samples.tobytes())
        return path

    def _video_only_file(self, name="noaudio.mp4"):
        import av
        path = self.root / name
        with av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate=30)
            stream.width, stream.height = 64, 48
            for _ in range(30):
                frame = av.VideoFrame(64, 48, "yuv420p")
                for plane in frame.planes:
                    plane.update(bytes([128]) * plane.buffer_size)
                container.mux(stream.encode(frame))
            container.mux(stream.encode(None))
        return path

    # ------------------------------------------------------------- API ---
    def test_chat_ok(self):
        self.assertEqual(self._chat("/ok"), '["你好"]')
        self.assertEqual(self.calls, [("/ok/v1/chat/completions", "Bearer test-key")])

    def test_redirect_never_forwards_authorization(self):
        with self.assertRaises(old.VideoError) as caught:
            self._chat("/redirect")
        self.assertIn("重定向", str(caught.exception))
        # 重定向在第一跳就被拒绝, 目标服务器根本没有收到请求, 密钥无从泄露.
        # (发往原 API 地址的第一跳本来就带密钥, 关键是它没有被转发.)
        self.assertEqual([path for path, _ in self.calls], ["/redirect/v1/chat/completions"])

    def test_oversized_response_rejected(self):
        with self.assertRaises(old.VideoError) as caught:
            self._chat("/oversized")
        self.assertIn("大小限制", str(caught.exception))

    def test_oversized_declared_length_rejected(self):
        with self.assertRaises(old.VideoError) as caught:
            self._chat("/oversized-header")
        self.assertIn("大小限制", str(caught.exception))

    def test_truncated_response_rejected(self):
        with self.assertRaises(old.VideoError):
            self._chat("/truncated")

    def test_auth_failure_reports_status(self):
        with self.assertRaises(old.VideoError) as caught:
            self._chat("/unauthorized")
        self.assertIn("401", str(caught.exception))

    def test_api_url_forms(self):
        for base, expected in [
                ("https://api.deepseek.com", "https://api.deepseek.com/v1/chat/completions"),
                ("https://example.test/v1/", "https://example.test/v1/chat/completions"),
                ("http://localhost:11434/v1", "http://localhost:11434/v1/chat/completions")]:
            with patch.object(old, "API_BASE", base):
                self.assertEqual(old._api_url(), expected)

    def test_api_url_rejects_credentials_and_queries(self):
        for base in ["file:///tmp/a", "https://user:secret@example.test",
                     "https://example.test?key=x", "https://example.test/#a",
                     "https://example.test:bad", "ftp://example.test/v1"]:
            with self.subTest(base=base), patch.object(old, "API_BASE", base), \
                    self.assertRaises(old.VideoError):
                old._api_url()

    # ---------------------------------------------------------- 环境变量 ---
    def test_bad_batch_env_gives_clear_error(self):
        for name, value, message in [
                ("VT_BATCH_SIZE", "abc", "VT_BATCH_SIZE 格式无效"),
                ("VT_BATCH_SIZE", "0", "VT_BATCH_SIZE 须在 1～100 之间"),
                ("VT_BATCH_DELAY", "-1", "VT_BATCH_DELAY 须在 0～600 之间")]:
            with self.subTest(name=name, value=value), \
                    patch.dict(os.environ, {name: value}), \
                    self.assertRaises(old.VideoError) as caught:
                old._load_batch_settings()
            self.assertIn(message, str(caught.exception))

    def test_import_survives_bad_batch_env(self):
        root = Path(old.__file__).resolve().parent
        code = (
            "import sys, types\n"
            "sys.path.insert(0, %r)\n" % str(root) +
            "fw = types.ModuleType('faster_whisper'); fw.WhisperModel = object\n"
            "sys.modules['faster_whisper'] = fw\n"
            "at = types.ModuleType('argostranslate')\n"
            "from types import SimpleNamespace\n"
            "at.package = SimpleNamespace(get_installed_packages=lambda: [], install_from_path=lambda p: None)\n"
            "at.translate = SimpleNamespace(translate=lambda *a: '')\n"
            "sys.modules['argostranslate'] = at\n"
            "sys.modules['argostranslate.package'] = at.package\n"
            "sys.modules['argostranslate.translate'] = at.translate\n"
            "import translate_video\n"
            "print('import ok')\n"
        )
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "VT_BATCH_SIZE": "abc"}
        result = subprocess.run([sys.executable, "-c", code], env=env,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("import ok", result.stdout)

    def test_main_with_bad_batch_env_fails_cleanly(self):
        media = self._wave()
        with patch.object(sys, "argv", ["translate_video.py", str(media), "--engine", "local"]), \
                patch.dict(os.environ, {"VT_BATCH_SIZE": "abc"}), \
                patch.object(old, "load_whisper") as loader, \
                self.assertRaises(old.VideoError) as caught:
            old.main()
        self.assertIn("VT_BATCH_SIZE 格式无效", str(caught.exception))
        loader.assert_not_called()

    # ------------------------------------------------------------- 代理 ---
    def test_proxied_uses_hostname(self):
        with patch.object(old, "GH_PROXY", "https://proxy.test/"):
            self.assertEqual(old.proxied("https://raw.githubusercontent.com/a/b"),
                             "https://proxy.test/https://raw.githubusercontent.com/a/b")
            self.assertEqual(old.proxied("https://github.com/a/b"),
                             "https://proxy.test/https://github.com/a/b")
            self.assertEqual(old.proxied("https://evilgithub.com/x.zip"),
                             "https://evilgithub.com/x.zip")
            self.assertEqual(old.proxied("https://example.com/dl?from=github.com"),
                             "https://example.com/dl?from=github.com")
            self.assertEqual(old.proxied("https://example.com/x"), "https://example.com/x")

    def test_proxied_empty_proxy_changes_nothing(self):
        with patch.object(old, "GH_PROXY", ""):
            self.assertEqual(old.proxied("https://github.com/a/b"), "https://github.com/a/b")

    # ------------------------------------------------------------- 缓存 ---
    def test_garbage_cache_is_rejected_and_retranscribed(self):
        media = self._wave()
        cache = self.root / "x.segs.json"
        stale_profile = {"path": str(media), "size": 999999, "mtime_ns": 0}
        for bad in [
                {"lang": "xx", "segs": [[5.0, 2.0, "结束早于开始"]]},
                {"lang": "english", "segs": [[0.0, 1.0, "Hello"]]},
                {"lang": "en", "segs": [[0.0, 1.0, ""]]},
                {"lang": "en", "segs": "not-a-list"},
                {"lang": "en", "segs": [[-1.0, 1.0, "Hello"]]},
                {"lang": "en", "segs": [[1.0, 2.0, "后一段"], [0.0, 1.0, "时间倒流"]]},
                # 结构上没问题, 但没有绑定源文件身份: 可能是别的文件留下的旧缓存.
                {"lang": "xx", "segs": [[0, 1, "STALE"], [999, 1000, "bogus"]]},
                # 结构没问题, 但绑定的源文件身份对不上: 文件已被替换.
                {"lang": "en", "segs": [[0.0, 1.0, "Hello"]], "profile": stale_profile}]:
            with self.subTest(bad=bad):
                cache.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
                model = FakeModel()
                with contextlib.redirect_stdout(io.StringIO()):
                    lang, segs = old.transcribe(model, str(media), cache)
                self.assertEqual(model.calls, 1)
                self.assertEqual(lang, "en")
                self.assertEqual(segs, [(0.0, 1.0, "Hello"), (1.0, 2.0, "World")])

    def test_malformed_cache_is_rejected_and_retranscribed(self):
        media = self._wave()
        cache = self.root / "x.segs.json"
        cache.write_text("{not json", encoding="utf-8")
        model = FakeModel()
        with contextlib.redirect_stdout(io.StringIO()):
            lang, segs = old.transcribe(model, str(media), cache)
        self.assertEqual(model.calls, 1)
        self.assertEqual(lang, "en")

    def test_valid_cache_skips_model(self):
        media = self._wave()
        cache = self.root / "x.segs.json"
        cache.write_text(json.dumps(
            {"lang": "en", "segs": [[0.0, 1.0, "Hello"], [1.0, 2.0, "World"]],
             "profile": old._cache_profile(str(media))},
            ensure_ascii=False), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            lang, segs = old.transcribe(RaisingModel(), str(media), cache)
        self.assertEqual(lang, "en")
        self.assertEqual(segs, [(0.0, 1.0, "Hello"), (1.0, 2.0, "World")])

    def test_retranscription_writes_profiled_cache(self):
        media = self._wave()
        cache = self.root / "x.segs.json"
        with contextlib.redirect_stdout(io.StringIO()):
            old.transcribe(FakeModel(), str(media), cache)
        saved = json.loads(cache.read_text(encoding="utf-8"))
        self.assertEqual(saved["profile"], old._cache_profile(str(media)))
        with contextlib.redirect_stdout(io.StringIO()):
            lang, segs = old.transcribe(RaisingModel(), str(media), cache)
        self.assertEqual(segs, [(0.0, 1.0, "Hello"), (1.0, 2.0, "World")])

    # --------------------------------------------------------- 音频截取 ---
    def test_no_audio_gives_clean_error(self):
        video = self._video_only_file()
        with self.assertRaises(old.VideoError) as caught:
            old.decode_clip(str(video), 0, 1)
        self.assertIn("没有音轨", str(caught.exception))

    def test_decode_clip_returns_audio(self):
        audio = old.decode_clip(str(self._wave()), 0, 1)
        self.assertEqual(audio.dtype.name, "float32")
        self.assertGreater(len(audio), 8000)

    # -------------------------------------------------------------- SRT ---
    def test_srt_rejects_end_before_start(self):
        out = self.root / "bad.srt"
        with self.assertRaises(old.VideoError):
            old.write_bilingual_srt([("orig", "译")], [(5.0, 2.0, "orig")], out, False)

    def test_srt_rejects_mismatched_counts(self):
        out = self.root / "bad.srt"
        with self.assertRaises(old.VideoError):
            old.write_bilingual_srt([("a", "甲"), ("b", "乙")], [(0.0, 1.0, "a")], out, False)

    def test_failed_srt_write_keeps_old_file(self):
        out = self.root / "keep.srt"
        out.write_bytes(b"ORIGINAL")
        with self.assertRaises(old.VideoError):
            old.write_bilingual_srt([("orig", "译")], [(5.0, 2.0, "orig")], out, False)
        self.assertEqual(out.read_bytes(), b"ORIGINAL")
        self.assertFalse(list(self.root.glob(".vt-*")))

    def test_srt_symlink_target_refused(self):
        target = self.root / "target.srt"
        target.write_bytes(b"ORIGINAL")
        link = self.root / "link.srt"
        link.symlink_to(target)
        with self.assertRaises(old.VideoError):
            old.write_bilingual_srt([("Hello", "你好")], [(0.0, 1.0, "Hello")], link, False)
        self.assertEqual(target.read_bytes(), b"ORIGINAL")

    def test_srt_valid_output(self):
        out = self.root / "ok.srt"
        old.write_bilingual_srt([("Hello", "你好"), ("World", "世界")],
                                [(0.0, 1.0, "Hello"), (1.0, 2.0, "World")], out, False)
        self.assertEqual(out.read_text(encoding="utf-8"),
                         "1\n00:00:00,000 --> 00:00:01,000\nHello\n你好\n\n"
                         "2\n00:00:01,000 --> 00:00:02,000\nWorld\n世界\n")
        old.write_bilingual_srt([("Hello", "你好")], [(0.0, 1.0, "Hello")], out, True)
        self.assertIn("\n你好\n", out.read_text(encoding="utf-8"))

    # --------------------------------------------------------- 模型下载 ---
    def test_model_download_uses_temp_file(self):
        path = old.download_argos_model("en_zh", self.base_a + "/model")
        try:
            self.assertTrue(path.name.startswith("vt-model-"))
            self.assertTrue(path.name.endswith(".argosmodel"))
            self.assertEqual(path.read_bytes(), b"fake-model-bytes")
        finally:
            path.unlink(missing_ok=True)

    def test_model_download_size_cap(self):
        before = set(Path(tempfile.gettempdir()).glob("vt-model-*.argosmodel"))
        with patch.object(old, "MAX_MODEL_SIZE", 10), \
                self.assertRaises(old.VideoError) as caught:
            old.download_argos_model("en_zh", self.base_a + "/model-big")
        self.assertIn("大小限制", str(caught.exception))
        after = set(Path(tempfile.gettempdir()).glob("vt-model-*.argosmodel"))
        self.assertEqual(after, before)

    def test_model_download_failure_leaves_no_temp_file(self):
        before = set(Path(tempfile.gettempdir()).glob("vt-model-*.argosmodel"))
        with patch.object(old, "GH_PROXY", ""), \
                self.assertRaises(old.VideoError):
            old.download_argos_model("en_zh", self.base_a + "/fail")
        after = set(Path(tempfile.gettempdir()).glob("vt-model-*.argosmodel"))
        self.assertEqual(after, before)

    # ------------------------------------------------------- 空译文提示 ---
    def test_empty_local_translation_is_warned_not_silent(self):
        with patch.object(old.package, "get_installed_packages",
                          return_value=[SimpleNamespace(from_code="en", to_code="zh")]), \
                patch.object(old.translate, "translate", return_value=""):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                pairs = old.translate_segments("en", [(0.0, 1.0, "Hello"), (1.0, 2.0, "World")])
        self.assertEqual(pairs, [("Hello", ""), ("World", "")])
        self.assertIn("注意：2 条台词翻译为空，已保留原文", output.getvalue())

    def test_empty_api_translation_is_warned_not_silent(self):
        with patch.object(old, "translate_batch_api", return_value=["", "译文"]):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                pairs = old.translate_segments_api("en", [(0.0, 1.0, "Hello"), (1.0, 2.0, "World")])
        self.assertEqual(pairs, [("Hello", ""), ("World", "译文")])
        self.assertIn("注意：1 条台词翻译为空，已保留原文", output.getvalue())

    # --------------------------------------------------------- 批量容错 ---
    def test_batch_continues_after_failing_file(self):
        bad = self.root / "bad.mp4"
        good = self.root / "good.mp4"
        bad.write_bytes(b"")
        good.write_bytes(b"")

        def fake_transcribe(model, path, cache=None):
            if "bad" in str(path):
                raise old.VideoError("模拟识别失败")
            return "en", [(0.0, 1.0, "Hello")]

        output = io.StringIO()
        with patch.object(sys, "argv", ["translate_video.py", str(bad), str(good), "--engine", "local"]), \
                patch.object(old, "load_whisper", return_value=object()), \
                patch.object(old, "transcribe", side_effect=fake_transcribe), \
                patch.object(old, "translate_segments", return_value=[("Hello", "你好")]), \
                contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as caught:
                old.main()
        self.assertEqual(caught.exception.code, 1)
        self.assertTrue((self.root / "good.srt").is_file())
        self.assertIn("你好", (self.root / "good.srt").read_text(encoding="utf-8"))
        self.assertFalse((self.root / "bad.srt").exists())
        self.assertIn("失败: bad.mp4: 模拟识别失败", output.getvalue())
        self.assertIn("批量结果: 成功 1，失败 1", output.getvalue())

    # ------------------------------------------- 复核跟进: 索引、片段、早校验 ---
    def test_fetch_argos_index_rejects_oversized(self):
        class BigIndex(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                body = json.dumps([{"from_code": "en", "to_code": "zh"}] * 300000).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), BigIndex)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = "http://127.0.0.1:%d/index.json" % server.server_port
            with patch.object(old, "ARGOS_INDEX_URL", url), \
                    patch.object(old, "GH_PROXY", ""), \
                    self.assertRaises(old.VideoError) as caught:
                old.fetch_argos_index()
            self.assertIn("无法下载翻译包索引", str(caught.exception))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_clip_mode_drops_empty_segments(self):
        media = self._wave()

        class ClipModel:
            def transcribe(self, *args, **kwargs):
                segments = [SimpleNamespace(start=0.0, end=1.0, text="   "),
                            SimpleNamespace(start=1.0, end=2.0, text="Hello")]
                return segments, SimpleNamespace(language="en", language_probability=0.99)

        output = io.StringIO()
        with patch.object(sys, "argv", ["translate_video.py", str(media), "--engine", "local",
                                        "--start-sec", "0", "--dur-sec", "180"]), \
                patch.object(old, "load_whisper", return_value=ClipModel()), \
                patch.object(old, "translate_segments", return_value=[("Hello", "你好")]), \
                contextlib.redirect_stdout(output):
            old.main()
        srt = (self.root / "sample.片段0-180.srt").read_text(encoding="utf-8")
        self.assertEqual(srt.count("-->"), 1)
        self.assertIn("00:00:01,000 --> 00:00:02,000", srt)
        self.assertIn("Hello", srt)
        self.assertIn("你好", srt)

    def test_bad_api_url_fails_before_model_load(self):
        media = self._wave()
        with patch.object(sys, "argv", ["translate_video.py", str(media), "--engine", "api"]), \
                patch.object(old, "API_BASE", "http://user:pass@example.com"), \
                patch.object(old, "API_KEY", "fake-key"), \
                patch.object(old, "load_whisper") as loader, \
                self.assertRaises(old.VideoError) as caught:
            old.main()
        self.assertIn("VT_API_BASE", str(caught.exception))
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
