#!/usr/bin/env python3
"""GUI for the independent subtitle translator; no study-assistant integration."""
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, scrolledtext, ttk

from subtitle_translator import VERSION

ROOT = Path(__file__).resolve().parent
ENGINES = [("auto", "自动（API 失败可本地兜底）"), ("api", "仅 API（失败保留进度）"), ("local", "仅本地翻译")]


def _opener_failed(proc):
    """True when a folder opener exited with a non-zero status.

    Only a real non-zero exit code counts: a mocked proc returns a
    non-int from poll(), which must not trigger the fallback dialog.
    """
    code = proc.poll()
    return isinstance(code, int) and code != 0


class App:
    def __init__(self, root):
        self.root, self.proc, self.cancelled = root, None, False
        self.files, self.q, self.controls = [], queue.Queue(), []
        self.needs_review = False
        root.title(f"视频翻译工具 {VERSION} · 双语字幕")
        root.geometry("800x580")
        root.minsize(720, 540)
        root.protocol("WM_DELETE_WINDOW", self.close)
        families = set(tkfont.families(root))
        for family in ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "Microsoft YaHei", "SimSun"]:
            if family in families:
                for name in ["TkDefaultFont", "TkTextFont", "TkMenuFont", "TkFixedFont"]:
                    tkfont.nametofont(name).configure(family=family, size=10)
                break
        self.build()
        self.timer = root.after(100, self.poll)

    def build(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")
        self.pick_button = ttk.Button(top, text="选择视频 / 音频", command=self.pick)
        self.pick_button.pack(side="left")
        self.count = tk.StringVar(value="未选择文件")
        ttk.Label(top, textvariable=self.count).pack(side="left", padx=12)
        row = ttk.Frame(self.root, padding=(10, 0))
        row.pack(fill="x")
        ttk.Label(row, text="识别模型").pack(side="left")
        self.model = ttk.Combobox(row, values=["tiny", "base", "small", "medium", "large-v3"], state="readonly", width=10)
        self.model.set("small")
        self.model.pack(side="left", padx=(5, 15))
        self.engine = ttk.Combobox(row, values=[item[1] for item in ENGINES], state="readonly", width=30)
        self.engine.set(ENGINES[0][1])
        self.engine.pack(side="left")
        ttk.Label(row, text="源语言").pack(side="left", padx=(15, 5))
        self.language = ttk.Combobox(row, values=["auto", "en", "ja", "ko", "zh", "fr", "de", "es"], state="readonly", width=6)
        self.language.set("auto")
        self.language.pack(side="left")
        destination = ttk.Frame(self.root, padding=(10, 8, 10, 0))
        destination.pack(fill="x")
        ttk.Label(destination, text="保存位置").pack(side="left")
        self.output_dir = tk.StringVar(value="")
        self.output_entry = ttk.Entry(destination, textvariable=self.output_dir)
        self.output_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.output_button = ttk.Button(destination, text="选择目录", command=self.pick_output)
        self.output_button.pack(side="left")
        ttk.Label(self.root, text="留空保存在视频旁；指定目录后，每个视频独立子目录，字幕和进度一起保存。",
                  padding=(10, 4, 10, 0)).pack(anchor="w")
        options = ttk.Frame(self.root, padding=10)
        options.pack(fill="x")
        self.zh_only = tk.BooleanVar(value=False)
        self.overwrite = tk.BooleanVar(value=False)
        self.fresh = tk.BooleanVar(value=False)
        checks = []
        for label, var in [("只输出中文", self.zh_only), ("替换已有字幕（先备份）", self.overwrite), ("重新识别和翻译", self.fresh)]:
            widget = ttk.Checkbutton(options, text=label, variable=var)
            widget.pack(side="left", padx=(0, 15))
            checks.append(widget)
        ttk.Label(self.root, text="默认保留已有字幕；中断后用相同文件、模型和引擎继续，已完成译文不会重复请求。",
                  padding=(10, 0)).pack(anchor="w")
        self.bar = ttk.Progressbar(self.root, mode="determinate")
        self.bar.pack(fill="x", padx=10, pady=10)
        actions = ttk.Frame(self.root, padding=(10, 0))
        actions.pack(fill="x")
        self.start_button = ttk.Button(actions, text="开始翻译", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="停止并保留进度", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        self.open_button = ttk.Button(actions, text="打开字幕文件夹", command=self.open_dir, state="disabled")
        self.open_button.pack(side="left")
        self.status = tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.status, padding=10).pack(anchor="w")
        self.log = scrolledtext.ScrolledText(self.root, height=18, font="TkFixedFont", state="disabled")
        self.log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.controls = [(self.pick_button, "normal"), (self.model, "readonly"), (self.engine, "readonly"),
                         (self.language, "readonly"), (self.output_entry, "normal"), (self.output_button, "normal"),
                         *[(widget, "normal") for widget in checks]]

    def pick_output(self):
        if self.proc is not None:
            return
        value = filedialog.askdirectory(title="选择字幕和进度保存目录")
        if value:
            self.output_dir.set(value)

    def pick(self):
        if self.proc is not None:
            return
        values = filedialog.askopenfilenames(title="选择视频或音频", filetypes=[
            ("音视频", "*.mp4 *.mkv *.avi *.mov *.flv *.wmv *.ts *.webm *.m4a *.mp3 *.wav *.flac *.aac"),
            ("全部文件", "*.*")])
        if values:
            self.files = list(dict.fromkeys(Path(value) for value in values))
            self.count.set(f"已选择 {len(self.files)} 个文件")
            self.open_button.config(state="normal")

    def command(self):
        engine = next(code for code, name in ENGINES if name == self.engine.get())
        worker = ROOT / ".venv" / "bin" / "python"
        command = [str(worker) if worker.is_file() else sys.executable, "-u", str(ROOT / "subtitle_translator.py"),
                   *map(str, self.files), "--model", self.model.get(), "--language", self.language.get(), "--engine", engine]
        for var, flag in [(self.zh_only, "--zh-only"), (self.overwrite, "--overwrite"), (self.fresh, "--fresh")]:
            if var.get():
                command.append(flag)
        if self.output_dir.get().strip():
            command.extend(["--output-dir", self.output_dir.get().strip()])
        return command

    def start(self):
        if self.proc is not None:
            return
        if not self.files:
            messagebox.showinfo("提示", "请先选择视频或音频")
            return
        try:
            # Create synchronously: the stop button never races an unassigned child.
            self.proc = subprocess.Popen(self.command(), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, encoding="utf-8", errors="replace", bufsize=1)
        except OSError:
            messagebox.showerror("启动失败", "无法启动翻译程序，请核对 Python 运行环境")
            return
        self.cancelled = False
        self.needs_review = False
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")
        for control, _ in self.controls:
            control.config(state="disabled")
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status.set("处理中；识别或首次下载模型可能需要等待")
        self.bar.config(mode="indeterminate", value=0)
        self.bar.start(12)
        threading.Thread(target=self.read_output, args=(self.proc,), daemon=True).start()

    def read_output(self, process):
        try:
            for line in process.stdout:
                self.q.put((process, "line", line.rstrip()))
        finally:
            if process.stdout:
                process.stdout.close()
            self.q.put((process, "done", process.wait()))

    def stop(self):
        process = self.proc
        if process is None or process.poll() is not None:
            return
        self.cancelled = True
        self.status.set("正在停止，已完成批次的进度会保留")
        self.stop_button.config(state="disabled")
        try:
            process.terminate()
        except ProcessLookupError:
            return
        self.root.after(2500, lambda: self.kill_if_running(process))

    @staticmethod
    def kill_if_running(process):
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass

    def append(self, text):
        self.log.config(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def poll(self):
        try:
            while True:
                process, kind, payload = self.q.get_nowait()
                if process is not self.proc:
                    continue
                if kind == "line":
                    self.append(payload)
                    if "需校对原文" in payload:
                        self.needs_review = True
                    progress = re.search(r"翻译进度 (\d+)/(\d+)", payload)
                    if progress:
                        self.bar.stop()
                        self.bar.config(mode="determinate", maximum=int(progress[2]), value=int(progress[1]))
                    if "处理：" in payload:
                        self.bar.config(mode="indeterminate", value=0)
                        self.bar.start(12)
                else:
                    self.bar.stop()
                    self.bar.config(mode="determinate", value=0)
                    self.proc = None
                    for control, state in self.controls:
                        control.config(state=state)
                    self.start_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.status.set("已停止；下次可继续" if self.cancelled else
                                    ("已生成字幕；含未译原文，请人工校对" if payload == 0 and self.needs_review else
                                     ("处理结束，请核对成功 / 跳过数量" if payload == 0 else "部分或全部失败，详见日志；完成进度已保留")))
        except queue.Empty:
            pass
        self.timer = self.root.after(100, self.poll)

    def close(self):
        if self.proc is not None:
            if not messagebox.askyesno("退出", "翻译仍在运行，是否停止并退出？已完成批次会保留。"):
                return
            process = self.proc
            self.stop()
            self.root.after(2600, lambda: self.finish_close(process))
            return
        self.root.destroy()

    def finish_close(self, process):
        self.kill_if_running(process)
        self.root.destroy()

    def open_dir(self):
        if self.files:
            folder = Path(self.output_dir.get().strip()).expanduser().resolve() if self.output_dir.get().strip() else self.files[0].parent
            try:
                proc = subprocess.Popen(["xdg-open", str(folder)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                self._show_folder_fallback(folder)
                return
            # xdg-open exits right away when no file manager can take the
            # folder, but Popen alone cannot see that failure, so check the
            # exit code shortly after instead of blocking the UI.
            self.root.after(1500, lambda: self._check_open_dir(proc, folder))

    def _check_open_dir(self, proc, folder):
        if _opener_failed(proc):
            self._show_folder_fallback(folder)

    def _show_folder_fallback(self, folder):
        messagebox.showinfo("无法打开字幕文件夹", f"系统没有可用的文件管理器，字幕保存在：\n{folder}")


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")
    App(root)
    root.mainloop()
