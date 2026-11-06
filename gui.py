#!/usr/bin/env python3
"""video-translator 图形界面"""

import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

DIR = Path(__file__).resolve().parent
VT = DIR / ".venv" / "bin" / "python"
MAIN = DIR / "translate_video.py"

MODELS = ["small", "base", "tiny", "medium", "large-v3"]
ENGINES = [("auto", "自动(API优先)"), ("api", "API大模型"), ("local", "本地离线")]


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.proc: subprocess.Popen | None = None
        root.title("视频学习助手 · 双语字幕")
        root.geometry("720x520")
        root.minsize(620, 460)
        self._build()
        root.after(100, self._poll)

    def _build(self) -> None:
        pad = {"padx": 8, "pady": 4}
        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)

        ttk.Button(top, text="选择影片(可多选)", command=self.pick).pack(side="left")
        self.count_var = tk.StringVar(value="未选择文件")
        ttk.Label(top, textvariable=self.count_var).pack(side="left", padx=8)

        row = ttk.Frame(self.root)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="识别模型:").pack(side="left")
        self.model = ttk.Combobox(row, values=MODELS, state="readonly", width=9)
        self.model.set("small")
        self.model.pack(side="left", padx=(2, 12))
        ttk.Label(row, text="翻译引擎:").pack(side="left")
        self.engine = ttk.Combobox(
            row, values=[n for _, n in ENGINES], state="readonly", width=14
        )
        self.engine.set(ENGINES[0][1])
        self.engine.pack(side="left", padx=(2, 12))
        self.zh_only = tk.BooleanVar()
        ttk.Checkbutton(row, text="只要中文(不带原文)", variable=self.zh_only).pack(
            side="left"
        )
        self.study = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="生成学习助手", variable=self.study).pack(side="left", padx=(12, 0))

        self.bar = ttk.Progressbar(self.root, mode="determinate")
        self.bar.pack(fill="x", **pad)

        btns = ttk.Frame(self.root)
        btns.pack(fill="x", **pad)
        self.start_btn = ttk.Button(btns, text="开始处理", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(
            btns, text="停止", command=self.stop, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=6)
        self.open_btn = ttk.Button(
            btns, text="打开字幕文件夹", command=self.open_dir, state="disabled"
        )
        self.open_btn.pack(side="left")
        self.study_btn = ttk.Button(
            btns, text="打开学习助手", command=self.open_study, state="disabled"
        )
        self.study_btn.pack(side="left", padx=6)

        self.log = scrolledtext.ScrolledText(self.root, height=18, font=("monospace", 10))
        self.log.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def pick(self) -> None:
        names = filedialog.askopenfilenames(
            title="选择影片或音频",
            filetypes=[
                ("音视频文件", "*.mp4 *.mkv *.avi *.mov *.flv *.wmv *.ts *.webm *.m4a *.mp3 *.wav *.flac *.aac"),
                ("所有文件", "*.*"),
            ],
        )
        if names:
            self.files = [Path(n) for n in names]
            self.count_var.set(f"已选 {len(self.files)} 个文件")
            self.open_btn.config(state="normal")

    def _engine_code(self) -> str:
        return next(c for c, n in ENGINES if n == self.engine.get())

    def start(self) -> None:
        if not getattr(self, "files", None):
            messagebox.showinfo("提示", "请先选择影片文件")
            return
        cmd = [
            str(VT), str(MAIN),
            *[str(f) for f in self.files],
            "-m", self.model.get(),
            "--engine", self._engine_code(),
        ]
        if self.zh_only.get():
            cmd.append("--zh-only")
        if self.study.get():
            cmd.append("--study")
        self.log.delete("1.0", "end")
        self.bar.config(mode="indeterminate")
        self.bar.start(12)
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.study_btn.config(state="disabled")
        threading.Thread(target=self._run, args=(cmd,), daemon=True).start()

    def _run(self, cmd: list) -> None:
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=str(DIR),
            )
            for line in self.proc.stdout:
                self.q.put(("line", line.rstrip()))
            code = self.proc.wait()
            self.q.put(("done", code))
        except Exception as e:
            self.q.put(("line", f"运行出错: {e}"))
            self.q.put(("done", 1))
        finally:
            self.proc = None

    def stop(self) -> None:
        if self.proc:
            self.proc.terminate()

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "line":
                    self._append(payload)
                elif kind == "done":
                    self.bar.stop()
                    self.bar.config(mode="determinate", value=0)
                    self.start_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    if payload == 0:
                        if self.study.get():
                            self._append("全部完成 ✓ 字幕和学习助手已生成在影片同目录")
                            self.study_btn.config(state="normal")
                        else:
                            self._append("全部完成 ✓ 字幕已生成在影片同目录")
                    else:
                        self._append("处理中断或失败, 见上方日志")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _append(self, text: str) -> None:
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def open_dir(self) -> None:
        if getattr(self, "files", None):
            subprocess.Popen(["xdg-open", str(self.files[0].parent)])

    def open_study(self) -> None:
        if getattr(self, "files", None):
            page = self.files[0].with_name(f"{self.files[0].stem}.学习助手.html")
            if page.exists():
                subprocess.Popen(["xdg-open", str(page)])
            else:
                messagebox.showinfo("提示", "尚未找到学习助手页面，请先完成处理")


if __name__ == "__main__":
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()
