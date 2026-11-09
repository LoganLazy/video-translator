"""Real Tk/subprocess smoke test. Requires DISPLAY and a cached tiny model.

Run with an explicit public speech fixture; never selects personal media.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path
from unittest.mock import Mock, patch

from subtitle_gui import App, VERSION
from subtitle_translator import result_path


def main():
    fixture = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, value=True):
        assert value, name
        checks.append(name)
        print('PASS', name, flush=True)

    os.environ['VT_API_KEY'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    root = tk.Tk()
    errors = []
    root.report_callback_exception = lambda *error: errors.append(str(error))
    app = App(root)

    def wait_done(timeout=40):
        deadline = time.monotonic() + timeout
        while app.proc is not None and time.monotonic() < deadline:
            root.update()
            time.sleep(0.02)
        root.update()
        assert app.proc is None, 'GUI process timeout'

    try:
        with tempfile.TemporaryDirectory(prefix='vt-v11-gui-media-') as directory:
            media = Path(directory) / '公开语音样本.flac'
            shutil.copy2(fixture, media)
            root.update()
            check('independent translation title', f'视频翻译工具 {VERSION}' in root.title() and '学习助手' not in root.title())
            check('safe defaults', not app.overwrite.get() and not app.fresh.get())
            with patch('subtitle_gui.filedialog.askopenfilenames', return_value=[str(media)]):
                app.pick()
            check('file picker selects actual fixture', app.files == [media])
            destination = Path(directory) / '字幕输出'
            with patch('subtitle_gui.filedialog.askdirectory', return_value=str(destination)):
                app.pick_output()
            check('output picker selects destination', app.output_dir.get() == str(destination))
            app.model.set('tiny')
            app.engine.set('仅本地翻译')
            app.language.set('en')
            check('translation command has no study options', '--study' not in app.command() and 'subtitle_translator.py' in app.command()[2])
            check('output directory reaches worker', app.command()[-2:] == ['--output-dir', str(destination)])
            app.start()
            process = app.proc
            check('process exists before stop can be clicked', process is not None)
            check('controls locked during run', str(app.pick_button['state']) == 'disabled' and str(app.model['state']) == 'disabled')
            check('destination cannot change during run', str(app.output_entry['state']) == 'disabled' and str(app.output_button['state']) == 'disabled')
            app.start()
            check('double start cannot spawn another process', app.proc is process)
            app.stop()
            wait_done()
            check('immediate stop is acknowledged', app.cancelled and '已停止' in app.status.get())
            app.start()
            wait_done()
            subtitle = result_path(media, None, 180, destination)
            check('GUI produces real bilingual SRT', subtitle.is_file() and '-->' in subtitle.read_text() and 'country' in subtitle.read_text().lower())
            check('source has no adjacent subtitles or caches', not media.with_suffix('.srt').exists() and not list(media.parent.glob('*.vt-*.json')) and not (media.parent / '.vt-cache').exists())
            with patch('subtitle_gui.subprocess.Popen') as opener:
                app.open_dir()
            check('open folder follows chosen destination', opener.call_args.args[0] == ['xdg-open', str(destination)])
            with patch('subtitle_gui.messagebox.showinfo') as notice:
                app._check_open_dir(Mock(poll=lambda: 1), destination)
            check('failed folder opener shows the folder path', notice.called and str(destination) in notice.call_args.args[1])
            with patch('subtitle_gui.messagebox.showinfo') as notice:
                app._check_open_dir(Mock(poll=lambda: 0), destination)
            check('successful folder opener stays silent', not notice.called)
            with patch('subtitle_gui.subprocess.Popen', side_effect=OSError('no xdg-open')), patch('subtitle_gui.messagebox.showinfo') as notice:
                app.open_dir()
            check('missing folder opener shows the folder path', notice.called and str(destination) in notice.call_args.args[1])
            check('controls restored after completion', str(app.model['state']) == 'readonly' and str(app.start_button['state']) == 'normal')
            before = subtitle.read_bytes()
            app.start()
            wait_done()
            check('GUI default preserves existing subtitles', subtitle.read_bytes() == before and '跳过 1' in app.log.get('1.0', 'end'))
            app.overwrite.set(True)
            app.start()
            wait_done()
            backups = list(destination.rglob('*.bak'))
            check('GUI replacement keeps exact backup', len(backups) == 1 and backups[0].read_bytes() == before)
            check('GUI shows resumed results', '恢复已完成译文' in app.log.get('1.0', 'end'))
            with patch.object(app, 'command', return_value=[sys.executable, '-u', '-c', 'print("需校对原文 2 条：已标记未译")']):
                app.start()
                wait_done()
            check('review-needed output has an explicit GUI warning', '未译原文' in app.status.get() and '校对' in app.status.get())
            check('no Tk callback exceptions', not errors)
            root.geometry('800x580+10+10')
            root.update()
            screenshot_python = os.environ.get('VT_SCREENSHOT_PYTHON')
            if screenshot_python:
                subprocess.run([screenshot_python, '-c',
                                'from PIL import ImageGrab; import os,sys; ImageGrab.grab(xdisplay=os.environ["DISPLAY"]).save(sys.argv[1])',
                                str(output / 'translator-v1.1.png')], check=True)
            (output / 'report.json').write_text(json.dumps({'status': 'passed', 'checks': checks, 'callback_errors': errors}, ensure_ascii=False, indent=2))
    finally:
        if app.proc is not None:
            app.proc.terminate()
            try:
                app.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                app.proc.kill()
                app.proc.wait(timeout=3)
        root.destroy()


if __name__ == '__main__':
    main()
