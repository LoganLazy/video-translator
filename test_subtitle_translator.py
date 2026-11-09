import contextlib
import io
import json
import math
import os
import tempfile
import unittest
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import subtitle_translator as vt


class TranslatorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="vt-v11-test-")
        self.root = Path(self.directory.name)
        self.state = self.root / "state.json"
        self.segments = [[0.0, 1.0, "Hello"], [1.0, 2.0, "World"], [2.0, 3.0, "Again"]]
        self.settings = vt.Settings(key="test-only", batch_size=1, batch_delay=0)
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.envfile = patch.object(vt, "load_env")
        self.envfile.start()

    def tearDown(self):
        self.envfile.stop()
        self.environment.stop()
        self.directory.cleanup()

    def translate(self, **kwargs):
        arguments = dict(segments=self.segments, language="en", engine="api", settings=self.settings,
                         state_path=self.state, profile={"fixture": 1}, api=lambda texts, *_: ["译" + text for text in texts])
        arguments.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return vt.translate_segments(**arguments)

    def wave(self, name="sample.wav"):
        import numpy as np
        path = self.root / name
        samples = np.concatenate([np.full(16000, value, dtype=np.int16) for value in [1000, 4000, 8000]])
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(samples.tobytes())
        return path

    def test_api_url_forms(self):
        for base in ["https://example.test", "https://example.test/v1/", "https://example.test/v1/chat/completions"]:
            self.assertEqual(vt.api_url(base), "https://example.test/v1/chat/completions")
        self.assertEqual(vt.api_url("https://example.test/compatible-mode/v1"), "https://example.test/compatible-mode/v1/chat/completions")

    def test_api_url_rejects_credentials_and_queries(self):
        for base in ["file:///tmp/a", "https://user:secret@example.test", "https://example.test?key=x", "https://example.test/#a", "https://example.test:bad"]:
            with self.subTest(base=base), self.assertRaises(vt.TranslationError):
                vt.api_url(base)

    def test_local_does_not_require_valid_online_settings(self):
        os.environ.update(VT_API_BASE="invalid", VT_BATCH_SIZE="wrong", VT_API_KEY="configured")
        self.assertEqual(vt.Settings.from_env("local").key, "")

    def test_forced_api_requires_key(self):
        with self.assertRaises(vt.TranslationError):
            vt.Settings.from_env("api")

    def test_env_values_have_bounds(self):
        for key, value in [("VT_BATCH_SIZE", "0"), ("VT_BATCH_SIZE", "101"), ("VT_API_TIMEOUT", "nan"), ("VT_BATCH_DELAY", "-1")]:
            with patch.dict(os.environ, {"VT_API_KEY": "test", key: value}), self.assertRaises(vt.TranslationError):
                vt.Settings.from_env("api")

    def test_env_does_not_override_process(self):
        path = self.root / ".env"
        path.write_text('VT_TEST="from file"\nVT_EXIST=changed\n# ignored\n', encoding="utf-8")
        os.environ["VT_EXIST"] = "original"
        self.envfile.stop()
        try:
            vt.load_env(path)
        finally:
            self.envfile.start()
        self.assertEqual(os.environ["VT_TEST"], "from file")
        self.assertEqual(os.environ["VT_EXIST"], "original")

    def test_parse_fenced_response(self):
        self.assertEqual(vt.parse_translations('```json\n["你好"]\n```', 1, ["Hello"]), ["你好"])

    def test_empty_or_misaligned_translations_rejected(self):
        for text in ['[""]', '[" "]', '[]', '[1]', '{"x":1}', 'explanation ["你好"]']:
            with self.subTest(text=text):
                self.assertIsNone(vt.parse_translations(text, 1, ["Hello"]))

    def test_auth_error_not_retried(self):
        chat = Mock(side_effect=vt.ApiError("HTTP 401"))
        sleep = Mock()
        with self.assertRaises(vt.ApiError):
            vt.api_batch(["hello"], "en", self.settings, request_chat=chat, sleep=sleep)
        self.assertEqual(chat.call_count, 1)
        sleep.assert_not_called()

    def test_transient_failure_retried_and_recovers(self):
        chat = Mock(side_effect=[vt.ApiError("HTTP 429", True, 3), '["你好"]'])
        sleep = Mock()
        self.assertEqual(vt.api_batch(["hello"], "en", self.settings, request_chat=chat, sleep=sleep), ["你好"])
        sleep.assert_called_once_with(3)

    def test_malformed_response_retries_are_bounded(self):
        chat = Mock(return_value="[]")
        with self.assertRaises(vt.ApiError):
            vt.api_batch(["hello"], "en", self.settings, request_chat=chat, sleep=Mock())
        self.assertEqual(chat.call_count, 3)

    def test_completed_state_does_not_repeat_requests(self):
        first, _ = self.translate()
        request = Mock(side_effect=AssertionError("must not call"))
        second, counts = self.translate(api=request)
        self.assertEqual(first, second)
        self.assertEqual(counts["api"], 3)
        request.assert_not_called()

    def test_interrupt_resume_only_pending_batch(self):
        request = Mock(side_effect=[["你好"], KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            self.translate(api=request)
        self.assertEqual(vt.read_json(self.state)["translations"][0]["text"], "你好")
        resumed = Mock(side_effect=lambda texts, *_: ["恢复" + text for text in texts])
        pairs, _ = self.translate(api=resumed)
        self.assertEqual(pairs[0][1], "你好")
        self.assertEqual([call.args[0] for call in resumed.call_args_list], [["World"], ["Again"]])

    def test_changed_provider_invalidates_translation_cache(self):
        self.translate()
        request = Mock(side_effect=lambda texts, *_: ["新" + text for text in texts])
        self.translate(settings=replace(self.settings, url="https://another.test/v1/chat/completions"), api=request)
        self.assertEqual(request.call_count, 3)

    def test_changed_source_or_transcript_invalidates_cache(self):
        self.translate()
        for options in [{"profile": {"fixture": 2}}, {"segments": [[0, 1, "new"]]}]:
            request = Mock(side_effect=lambda texts, *_: ["新" + text for text in texts])
            self.translate(api=request, **options)
            self.assertGreater(request.call_count, 0)

    def test_fresh_ignores_saved_translations(self):
        self.translate()
        request = Mock(side_effect=lambda texts, *_: ["新" + text for text in texts])
        self.translate(fresh=True, api=request)
        self.assertEqual(request.call_count, 3)

    def test_forced_api_never_falls_back(self):
        local = Mock()
        with self.assertRaises(vt.ApiError):
            self.translate(api=Mock(side_effect=vt.ApiError("HTTP 403")), local=local)
        local.assert_not_called()

    def test_auto_fallback_stops_retrying_failed_provider(self):
        request = Mock(side_effect=vt.ApiError("HTTP 403"))
        pairs, counts = self.translate(engine="auto", api=request, local=lambda texts, _: ["本地" + t for t in texts])
        self.assertEqual(request.call_count, 1)
        self.assertEqual(counts["local"], 3)
        self.assertEqual(len(pairs), 3)

    def test_chinese_source_uses_no_online_or_local_model(self):
        fail = Mock(side_effect=AssertionError("unused"))
        pairs, counts = self.translate(language="zh", api=fail, local=fail)
        self.assertEqual(counts["original"], 3)
        self.assertEqual(pairs[0], ("Hello", "Hello"))

    def test_strict_api_does_not_reuse_auto_fallback(self):
        self.translate(engine="auto", api=Mock(side_effect=vt.ApiError("HTTP 403")), local=lambda texts, _: texts)
        request = Mock(side_effect=lambda texts, *_: ["API" + t for t in texts])
        _, counts = self.translate(api=request)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(counts["local"], 0)

    def test_bad_batch_does_not_commit_partial_results(self):
        with self.assertRaises(vt.TranslationError):
            self.translate(api=lambda *_: [""])
        self.assertIsNone(vt.read_json(self.state))

    def test_batch_character_and_count_limits(self):
        segments = [[i, i + 1, "word"] for i in range(5)]
        self.assertEqual(list(vt.make_batches(range(5), segments, 3, 9)), [[0, 1], [2, 3], [4]])
        with self.assertRaises(vt.TranslationError):
            list(vt.make_batches([0], segments, 3, 2))

    def test_srt_alignment_and_zh_only(self):
        pairs = [("Hello", "你好"), ("World", "世界"), ("Again", "再次")]
        text = vt.render_srt(pairs, self.segments).decode()
        self.assertIn("00:00:00,000 --> 00:00:01,000\nHello\n你好", text)
        self.assertNotIn("Hello", vt.render_srt(pairs, self.segments, True).decode())
        with self.assertRaises(vt.TranslationError):
            vt.render_srt(pairs[:1], self.segments)

    def test_chinese_is_not_duplicated(self):
        text = vt.render_srt([("你好", "你好")], [[0, 1, "你好"]]).decode()
        self.assertEqual(text.count("你好"), 1)

    def test_srt_removes_blank_cue_separators_and_controls(self):
        text = vt.render_srt([("Hi\x00\n\nthere", "你好\n\n世界")], [[0, 1, "Hi"]]).decode()
        self.assertIn("Hi\nthere\n你好\n世界", text)

    def test_timestamp_rounding(self):
        self.assertEqual(vt.fmt_ts(59.9996), "00:01:00,000")
        for number in [-1, math.nan, math.inf]:
            with self.assertRaises(vt.TranslationError):
                vt.fmt_ts(number)

    def test_existing_srt_preserved_by_default(self):
        output = self.root / "movie.srt"
        output.write_bytes(b"original")
        with self.assertRaises(vt.TranslationError):
            vt.publish_srt(output, b"new")
        self.assertEqual(output.read_bytes(), b"original")
        self.assertFalse(list(self.root.glob(".vt-*")))

    def test_overwrite_makes_private_exact_backup(self):
        output = self.root / "movie.srt"
        output.write_bytes(b"original\r\n")
        backup = vt.publish_srt(output, b"new", True)
        self.assertEqual(backup.read_bytes(), b"original\r\n")
        self.assertEqual(output.read_bytes(), b"new")
        self.assertEqual(backup.stat().st_mode & 0o777, 0o600)

    def test_atomic_failure_keeps_old_data(self):
        path = self.root / "state.json"
        path.write_bytes(b"original")
        with patch.object(vt.os, "replace", side_effect=OSError("simulated")), self.assertRaises(OSError):
            vt.atomic_bytes(path, b"new")
        self.assertEqual(path.read_bytes(), b"original")
        self.assertFalse(list(self.root.glob(".vt-*")))

    def test_symlink_output_and_state_rejected(self):
        target, link = self.root / "original", self.root / "link"
        target.write_bytes(b"original")
        link.symlink_to(target)
        with self.assertRaises(vt.TranslationError):
            vt.publish_srt(link, b"new", True)
        with self.assertRaises(vt.TranslationError):
            vt.save_json(link, {"test": 1})
        self.assertEqual(target.read_bytes(), b"original")

    def test_clip_exactly_trims_preroll_and_tail(self):
        import numpy as np
        path = self.wave()
        audio = vt.decode_clip(path, 1.25, 0.5)
        self.assertEqual(len(audio), 8000)
        self.assertTrue(np.allclose(audio, 4000 / 32768, atol=0.0001))
        audio = vt.decode_clip(path, 0.9, 0.2)
        self.assertEqual(len(audio), 3200)
        self.assertTrue(np.allclose(audio[:1600], 1000 / 32768, atol=0.0001))
        self.assertTrue(np.allclose(audio[1600:], 4000 / 32768, atol=0.0001))

    def test_validated_transcript_skips_model_load(self):
        path = self.wave()
        profile = vt.source_profile(path, "tiny", "auto", None, 180)
        cache = path.with_name(path.name + ".vt-transcript.json")
        vt.save_json(cache, {"schema": 1, "profile": profile, "language": "en", "segments": self.segments})
        getter = Mock(side_effect=AssertionError("must not load model"))
        language, segments = vt.transcribe(path, profile, getter)
        self.assertEqual(language, "en")
        self.assertEqual(segments, self.segments)
        getter.assert_not_called()

    def test_changed_video_invalidates_transcript(self):
        path = self.wave()
        before = vt.source_profile(path, "tiny", "auto", None, 180)
        vt.save_json(path.with_name(path.name + ".vt-transcript.json"), {"schema": 1, "profile": before, "language": "en", "segments": self.segments})
        path.write_bytes(path.read_bytes() + b"changed")
        after = vt.source_profile(path, "tiny", "auto", None, 180)
        getter = Mock(side_effect=RuntimeError("model required"))
        with self.assertRaises(RuntimeError):
            vt.transcribe(path, after, getter)
        getter.assert_called_once()

    def test_invalid_cache_is_not_accepted(self):
        path = self.wave()
        profile = vt.source_profile(path, "tiny", "auto", None, 180)
        cache = path.with_name(path.name + ".vt-transcript.json")
        cache.write_text('{"broken":')
        getter = Mock(side_effect=RuntimeError("model required"))
        with self.assertRaises(RuntimeError):
            vt.transcribe(path, profile, getter)
        getter.assert_called_once()

    def test_segment_validation(self):
        for value in [[], [[1, 0, "bad"]], [[0, 1, ""]], [[math.nan, 1, "bad"]], [[True, 1, "bad"]], [[1, 2, "a"], [0, 1, "b"]]]:
            self.assertFalse(vt.valid_segments(value))

    def test_clip_transcript_uses_source_timeline_and_caps_end(self):
        path = self.wave()
        profile = vt.source_profile(path, "tiny", "en", 1.25, 0.5)
        model = Mock()
        model.transcribe.return_value = iter([SimpleNamespace(start=0, end=0.8, text="Hello")]), SimpleNamespace(language="en")
        language, segments = vt.transcribe(path, profile, lambda: model)
        self.assertEqual(segments, [[1.25, 1.75, "Hello"]])

    def test_existing_output_skipped_before_model_load(self):
        path = self.wave()
        path.with_suffix(".srt").write_bytes(b"original")
        factory = Mock(side_effect=AssertionError("unused"))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(vt.main([str(path), "--engine", "local"], model_factory=factory), 0)
        factory.assert_not_called()
        self.assertEqual(path.with_suffix(".srt").read_bytes(), b"original")

    def test_batch_failure_does_not_block_next_file(self):
        good = self.wave()
        bad = self.root / "missing.mp4"
        segments = [SimpleNamespace(start=0, end=1, text="Hello")]
        model = Mock()
        model.transcribe.return_value = iter(segments), SimpleNamespace(language="en")
        with patch.object(vt, "local_translate", return_value=["你好"]), contextlib.redirect_stdout(io.StringIO()):
            status = vt.main([str(bad), str(good), "--engine", "local"], model_factory=lambda *_a, **_k: model)
        self.assertEqual(status, 1)
        self.assertIn("你好", good.with_suffix(".srt").read_text())

    def test_nonfinite_and_negative_clip_args_rejected(self):
        for options in [["--start-sec", "-1"], ["--start-sec", "nan"], ["--dur-sec", "0"], ["--dur-sec", "inf"]]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                vt.main(["dummy.mp4", *options])
            self.assertEqual(result.exception.code, 2)

    def test_clip_names_include_fractional_start_and_duration(self):
        self.assertEqual(vt.output_path(Path("movie.mp4"), 1.25, 0.5).name, "movie.片段1.25-1.75.srt")

    def test_long_movie_fractional_clip_names_do_not_collide(self):
        first = vt.output_path(Path("movie.mp4"), 11882.16, 180)
        second = vt.output_path(Path("movie.mp4"), 11882.17, 180)
        self.assertNotEqual(first, second)
        self.assertEqual(first.name, "movie.片段11882.16-12062.16.srt")

    def test_output_directory_keeps_source_directory_untouched(self):
        source = self.root / "read-only"
        source.mkdir()
        path = self.wave("read-only/片段.wav")
        before = path.read_bytes(), path.stat().st_mtime_ns
        source.chmod(0o555)
        model = Mock()
        model.transcribe.return_value = iter([SimpleNamespace(start=0, end=1, text="Hello")]), SimpleNamespace(language="en")
        destination = self.root / "output with spaces"
        try:
            with patch.object(vt, "local_translate", return_value=["你好"]), contextlib.redirect_stdout(io.StringIO()):
                status = vt.main([str(path), "--engine", "local", "--output-dir", str(destination)],
                                 model_factory=lambda *_a, **_k: model)
            self.assertEqual(status, 0)
            self.assertEqual(list(source.iterdir()), [path])
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
            subtitles = list(destination.glob("*/*.srt"))
            self.assertEqual(len(subtitles), 1)
            self.assertIn("你好", subtitles[0].read_text())
            self.assertEqual(len(list(destination.rglob("*.progress.json"))), 1)
            self.assertEqual(len(list(destination.rglob("*.transcript.json"))), 1)
        finally:
            source.chmod(0o700)

    def test_destination_distinguishes_same_names_and_different_extensions(self):
        paths = [self.root / "a/movie.mp4", self.root / "b/movie.mp4", self.root / "a/movie.mkv"]
        outputs = [vt.result_path(path, None, 180, self.root / "out") for path in paths]
        self.assertEqual(len(set(outputs)), 3)
        self.assertEqual(outputs[0], vt.result_path(paths[0], None, 180, self.root / "out"))

    def test_alternating_clips_reuses_each_completed_profile(self):
        path = self.wave()
        model = Mock()
        model.transcribe.side_effect = lambda *_a, **_k: (iter([SimpleNamespace(start=0, end=0.5, text="Hello")]), SimpleNamespace(language="en"))
        local = Mock(return_value=["你好"])
        logs = io.StringIO()
        with patch.object(vt, "local_translate", local), contextlib.redirect_stdout(logs):
            for start in [0, 1, 0, 1]:
                status = vt.main([str(path), "--engine", "local", "--start-sec", str(start), "--dur-sec", "1", "--overwrite"],
                                 model_factory=lambda *_a, **_k: model)
                self.assertEqual(status, 0)
        self.assertEqual(model.transcribe.call_count, 2)
        self.assertEqual(local.call_count, 2)
        self.assertEqual(len(list(self.root.rglob("*.transcript.json"))), 2)
        self.assertEqual(len(list(self.root.rglob("*.progress.json"))), 2)

    def test_cache_paths_separate_model_engine_provider_but_not_key_rotation(self):
        path = self.wave()
        profile = vt.source_profile(path, "tiny", "en", None, 180)
        def paths(settings=self.settings, engine="api", current=profile):
            return vt.state_paths(path, current, engine, settings, path.with_suffix(".srt"))
        transcript, state = paths()
        for settings in [replace(self.settings, model="other"), replace(self.settings, url="https://other.test/v1/chat/completions")]:
            new_transcript, new_state = paths(settings)
            self.assertEqual(transcript, new_transcript)
            self.assertNotEqual(state, new_state)
        self.assertNotEqual(state, paths(engine="local")[1])
        self.assertNotEqual(transcript, paths(current={**profile, "model": "small"})[0])
        self.assertEqual((transcript, state), paths(replace(self.settings, key="rotated-key")))
        self.assertNotIn(self.settings.key, str(state))

    def test_v111_state_migrates_without_model_or_translation_calls(self):
        path = self.wave()
        profile = vt.source_profile(path, "small", "auto", None, 180)
        legacy = path.with_name(path.name + ".vt-transcript.json")
        progress = path.with_name(path.name + ".vt-progress.json")
        vt.save_json(legacy, {"schema": 1, "profile": profile, "language": "en", "segments": self.segments})
        self.translate(profile=profile, engine="local", settings=vt.Settings(), state_path=progress,
                       local=lambda texts, _: ["译" + t for t in texts])
        before = legacy.read_bytes(), progress.read_bytes()
        fail = Mock(side_effect=AssertionError("cache should be reused"))
        with patch.object(vt, "local_translate", fail), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(vt.main([str(path), "--engine", "local", "--output-dir", str(self.root / "out")], model_factory=fail), 0)
        fail.assert_not_called()
        self.assertEqual((legacy.read_bytes(), progress.read_bytes()), before)
        self.assertEqual(len(list((self.root / "out").rglob("*.progress.json"))), 1)

    def test_invalid_legacy_identity_is_not_migrated(self):
        self.translate(profile={"other": 2})
        destination = self.root / "new-progress.json"
        request = Mock(side_effect=lambda texts, *_: ["新" + t for t in texts])
        self.translate(state_path=destination, legacy_state=self.state, api=request)
        self.assertEqual(request.call_count, 3)

    def test_cache_symlink_does_not_write_outside_result_directory(self):
        path = self.wave()
        outside = self.root / "outside"
        outside.mkdir()
        (self.root / ".vt-cache").symlink_to(outside, target_is_directory=True)
        fail = Mock(side_effect=AssertionError("must reject before loading model"))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(vt.main([str(path), "--engine", "local"], model_factory=fail), 1)
        fail.assert_not_called()
        self.assertEqual(list(outside.iterdir()), [])

    def test_result_subdirectory_symlink_is_rejected(self):
        path = self.wave()
        output = vt.result_path(path, None, 180, self.root / "out")
        output.parent.rmdir()
        output.parent.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(vt.TranslationError):
            vt.result_path(path, None, 180, self.root / "out")

    def test_output_cannot_replace_source_itself(self):
        with self.assertRaises(vt.TranslationError):
            vt.result_path(self.root / "input.srt", None, 180)

    def test_invalid_output_directory_fails_before_model_load(self):
        path = self.wave()
        fail = Mock(side_effect=AssertionError("must reject before loading model"))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(vt.main([str(path), "--engine", "local", "--output-dir", str(path)], model_factory=fail), 1)
        fail.assert_not_called()

    def test_credentials_are_not_saved_in_progress(self):
        self.translate()
        self.assertNotIn("test-only", self.state.read_text())
        self.assertNotIn("test-only", repr(self.settings))

    def test_empty_local_result_is_preserved_as_review_item(self):
        _, counts = self.translate(engine="local", local=lambda texts, _: [vt.Untranslated(texts[0])])
        self.assertEqual(counts["untranslated"], 3)
        self.assertEqual(counts["local"], 0)
        records = vt.read_json(self.state)["translations"]
        self.assertEqual(records[0], {"text": "Hello", "engine": "untranslated", "reason": "empty_local_output"})

    def test_untranslated_srt_marks_original_once_even_in_zh_only(self):
        pairs, _ = self.translate(engine="local", local=lambda texts, _: [vt.Untranslated(texts[0])])
        for zh_only in [False, True]:
            text = vt.render_srt(pairs, self.segments, zh_only).decode()
            self.assertEqual(text.count("Hello"), 1)
            self.assertEqual(text.count("【未译，保留原文】"), 3)

    def test_saved_review_items_are_not_silently_retranslated(self):
        self.translate(engine="local", local=lambda texts, _: [vt.Untranslated(texts[0])])
        local = Mock(side_effect=AssertionError("must not request again"))
        _, counts = self.translate(engine="local", local=local)
        self.assertEqual(counts["untranslated"], 3)
        local.assert_not_called()

    def test_retry_untranslated_only_processes_review_rows(self):
        responses = [["完成一"], [vt.Untranslated("World")], ["完成三"]]
        self.translate(engine="local", local=Mock(side_effect=responses))
        local = Mock(return_value=["世界"])
        pairs, counts = self.translate(engine="local", local=local, retry_untranslated=True)
        local.assert_called_once_with(["World"], "en")
        self.assertEqual(counts["untranslated"], 0)
        self.assertEqual([p[1] for p in pairs], ["完成一", "世界", "完成三"])

    def test_wrong_original_in_review_item_rejected(self):
        with self.assertRaises(vt.TranslationError):
            self.translate(engine="local", local=lambda *_: [vt.Untranslated("different source")])
        self.assertIsNone(vt.read_json(self.state))

    def test_api_cannot_commit_a_local_review_object(self):
        with self.assertRaises(vt.TranslationError):
            self.translate(api=lambda texts, *_: [vt.Untranslated(texts[0])])

    def test_empty_intermediate_does_not_feed_empty_text_to_next_model(self):
        package, translator = Mock(), Mock()
        package.get_installed_packages.return_value = [SimpleNamespace(from_code="ja", to_code="en"), SimpleNamespace(from_code="en", to_code="zh")]
        translator.translate.return_value = ""
        with patch.object(vt, "ensure_chain"), patch.object(vt, "_argos", (package, translator)):
            result = vt.local_translate(["test"], "ja")
        translator.translate.assert_called_once_with("test", "ja", "en")
        self.assertEqual(result, [vt.Untranslated("test")])

    def test_local_model_exception_still_reports_a_failure(self):
        package, translator = Mock(), Mock()
        package.get_installed_packages.return_value = [SimpleNamespace(from_code="en", to_code="zh")]
        translator.translate.side_effect = RuntimeError("model failure")
        with patch.object(vt, "ensure_chain"), patch.object(vt, "_argos", (package, translator)), self.assertRaises(RuntimeError):
            vt.local_translate(["test"], "en")

    def test_model_downloads_send_browser_user_agent(self):
        # Cloudflare-fronted model hosts (argos-net.com) reject the default
        # "Python-urllib/x.y" User-Agent with HTTP 403, so both the model
        # index fetch and the model file download must present a browser UA.
        # (Reproduced live: bare urlopen -> 403, curl -> 200, same URL.)
        def fake_response(body):
            response = MagicMock()
            response.__enter__.return_value = io.BytesIO(body)
            return response

        index = [{"from_code": "en", "to_code": "zh", "package_version": "1.9",
                  "links": ["https://argos-net.com/v1/translate-en_zh-1_9.argosmodel"]}]
        with patch("urllib.request.urlopen", return_value=fake_response(json.dumps(index).encode())) as urlopen:
            self.assertEqual(vt.get_model_index(), index)
        request = urlopen.call_args[0][0]
        user_agent = request.get_header("User-agent")
        self.assertTrue(user_agent.startswith("Mozilla/5.0"), user_agent)
        self.assertNotIn("Python-urllib", user_agent)

        package, translator = Mock(), Mock()
        package.get_installed_packages.return_value = []
        with patch.object(vt, "argos_modules", return_value=(package, translator)), \
                patch.object(vt, "get_model_index", return_value=index), \
                patch("urllib.request.urlopen", return_value=fake_response(b"fake-model-bytes")) as urlopen:
            vt.ensure_chain("en")
        request = urlopen.call_args[0][0]
        user_agent = request.get_header("User-agent")
        self.assertTrue(user_agent.startswith("Mozilla/5.0"), user_agent)
        self.assertNotIn("Python-urllib", user_agent)
        package.install_from_path.assert_called_once()
        installed = package.install_from_path.call_args[0][0]
        self.assertEqual(Path(installed).suffix, ".argosmodel")
        self.assertFalse(Path(installed).exists())

    def test_api_chat_sends_browser_user_agent(self):
        # Cloudflare-fronted API hosts reject the default
        # "Python-urllib/x.y" User-Agent with HTTP 403 (error 1010), so
        # the chat request must present a browser UA, like the model
        # download already does.
        # (Reproduced live: Python-urllib UA -> 403, curl UA -> 200.)
        body = json.dumps({"choices": [{"message": {"content": "你好"}}]}).encode()
        response = MagicMock()
        response.__enter__.return_value = response
        response.headers.get.return_value = None
        response.read.return_value = body
        opener = MagicMock()
        opener.open.return_value = response
        settings = vt.Settings(key="test-key", url="https://example.invalid/v1/chat/completions",
                               model="test-model")
        with patch("urllib.request.build_opener", return_value=opener):
            self.assertEqual(vt.chat(settings, [{"role": "user", "content": "Hello"}]), "你好")
        request = opener.open.call_args[0][0]
        user_agent = request.get_header("User-agent")
        self.assertTrue(user_agent.startswith("Mozilla/5.0"), user_agent)
        self.assertNotIn("Python-urllib", user_agent)

    def test_cli_publishes_review_markers_and_reports_incomplete_translation(self):
        path = self.wave()
        model = Mock()
        model.transcribe.return_value = iter([SimpleNamespace(start=0, end=1, text="Hello")]), SimpleNamespace(language="en")
        output = io.StringIO()
        with patch.object(vt, "local_translate", return_value=[vt.Untranslated("Hello")]), contextlib.redirect_stdout(output):
            status = vt.main([str(path), "--engine", "local"], model_factory=lambda *_a, **_k: model)
        self.assertEqual(status, 0)
        self.assertIn("需校对原文 1 条", output.getvalue())
        self.assertIn("本地 0", output.getvalue())
        self.assertIn("【未译，保留原文】Hello", path.with_suffix('.srt').read_text())


if __name__ == "__main__":
    unittest.main()
