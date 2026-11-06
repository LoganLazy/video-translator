"""Build a self-contained, local learning page from timestamped subtitles."""

import html
import json
import re
from pathlib import Path
from urllib.parse import quote


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _clock(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def make_chapters(segs: list[tuple], pairs: list[tuple], target_seconds: int = 360) -> list[dict]:
    """Create readable chapters at natural pauses, roughly every six minutes."""
    rows = [
        {"start": round(start, 2), "end": round(end, 2), "original": _clean(orig), "zh": _clean(zh)}
        for (start, end, _), (orig, zh) in zip(segs, pairs)
    ]
    if not rows:
        return []
    groups, current = [], [rows[0]]
    anchor = rows[0]["start"]
    for row in rows[1:]:
        elapsed = row["start"] - anchor
        pause = row["start"] - current[-1]["end"]
        if elapsed >= target_seconds and (pause >= 1.2 or elapsed >= target_seconds * 1.45):
            groups.append(current)
            current, anchor = [row], row["start"]
        else:
            current.append(row)
    groups.append(current)

    chapters = []
    for number, group in enumerate(groups, 1):
        text = " ".join(x["zh"] or x["original"] for x in group)
        sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?])\s*", text) if len(s.strip()) > 8]
        # Spread the selected sentences across the chapter instead of only
        # repeating its opening; this gives a more representative offline summary.
        if len(sentences) > 3:
            picks = [sentences[0], sentences[len(sentences) // 2], sentences[-1]]
        else:
            picks = sentences
        summary = "；".join(picks)[:360] or "本章节包含视频内容与对应字幕。"
        title = (sentences[0] if sentences else text)[:34].rstrip("。；，, ")
        chapters.append({
            "number": number, "start": group[0]["start"], "end": group[-1]["end"],
            "title": title or f"第 {number} 章", "summary": summary,
            "keywords": [], "questions": [],
        })
    return chapters


def write_study_pack(
    media: Path,
    segs: list[tuple],
    pairs: list[tuple],
    output: Path | None = None,
    chapters: list[dict] | None = None,
) -> Path:
    """Write an offline HTML viewer with chapters, subtitle search, and local notes."""
    output = output or media.with_name(f"{media.stem}.学习助手.html")
    rows = [
        {"start": round(start, 2), "end": round(end, 2), "original": _clean(orig), "zh": _clean(zh)}
        for (start, end, _), (orig, zh) in zip(segs, pairs)
    ]
    data = {
        "title": media.stem,
        "media": quote(media.name),
        "rows": rows,
        "chapters": chapters if chapters is not None else make_chapters(segs, pairs),
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page_title = html.escape(f"{media.stem} · 视频学习助手")
    document = f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{page_title}</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#10131a;color:#e9edf7;font:15px system-ui,-apple-system,"Microsoft YaHei",sans-serif}}header{{padding:18px 5%;border-bottom:1px solid #2b3140}}h1{{font-size:20px;margin:0}}main{{max-width:1400px;margin:auto;padding:20px 5%;display:grid;grid-template-columns:minmax(360px,1.25fr) minmax(320px,.75fr);gap:22px}}video{{width:100%;max-height:62vh;background:#000;border-radius:12px}}section{{background:#191e29;border:1px solid #2b3140;border-radius:12px;padding:16px;margin-top:14px}}h2{{font-size:16px;margin:0 0 12px}}button,.jump{{border:0;border-radius:7px;background:#2d69ed;color:#fff;padding:7px 10px;cursor:pointer;font:inherit}}button.secondary{{background:#343c50}}input,textarea,select{{background:#10131a;color:#e9edf7;border:1px solid #394155;border-radius:7px;padding:9px;font:inherit}}input,textarea{{width:100%}}textarea{{min-height:100px;resize:vertical}}.toolbar{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 0}}.chapter,.line{{padding:10px;border-bottom:1px solid #2b3140}}.chapter:last-child,.line:last-child{{border:0}}.time{{color:#8da9f7;font-variant-numeric:tabular-nums;margin-right:8px}}.chapter b{{display:block;margin:5px 0}}.chapter p{{margin:5px 0;color:#b8c0d3;line-height:1.55}}.tags{{display:flex;gap:5px;flex-wrap:wrap;margin:7px 0}}.tag{{background:#293650;color:#bcd0ff;border-radius:999px;padding:3px 8px;font-size:12px}}.question{{color:#d9c38a!important;border-left:2px solid #937b3e;padding-left:8px}}.line{{cursor:pointer;line-height:1.5;border-radius:8px}}.line:hover{{background:#222a3a}}.line.active{{background:#24375d;box-shadow:inset 3px 0 #5d91ff}}.orig{{color:#aeb8cb;font-size:13px}}#results{{max-height:52vh;overflow:auto}}.muted{{color:#aeb8cb;font-size:13px}}@media(max-width:850px){{main{{display:block}}}} 
</style><header><h1>{page_title}</h1><div class="muted">本地学习包：章节 · 字幕检索 · 时间戳笔记（笔记只存于此浏览器）</div></header>
<main><div><video id="player" controls preload="metadata"></video><div class="toolbar"><label>倍速 <select id="speed"><option>.75</option><option selected>1</option><option>1.25</option><option>1.5</option><option>2</option></select></label><button id="resume" class="secondary">继续上次进度</button><span id="progress" class="muted"></span></div><section><h2>我的笔记</h2><textarea id="notes" placeholder="记录想法；点“插入当前时间”可创建能回跳的时间戳…"></textarea><div class="toolbar"><button id="stamp">插入当前时间</button><button id="save">保存笔记</button><button id="export" class="secondary">导出 Markdown</button><span id="saved" class="muted"></span></div></section><section><h2>逐句字幕与检索</h2><input id="query" placeholder="搜索概念、术语或一句话，点击结果跳转"><div id="results"></div></section></div><aside><section><h2>课程章节</h2><div id="chapters"></div></section></aside></main>
<script id="study-data" type="application/json">{payload}</script><script>
const $=s=>document.querySelector(s),D=JSON.parse($('#study-data').textContent),V=$('#player'),Q=$('#query'),R=$('#results'),C=$('#chapters'),N=$('#notes'),base='vt-study:'+location.pathname,noteKey=base+':notes',progressKey=base+':progress';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
V.src=D.media;N.value=localStorage.getItem(noteKey)||'';function time(n){{let s=Math.max(0,Math.floor(n));return String(Math.floor(s/3600)).padStart(2,'0')+':'+String(Math.floor(s%3600/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0')}}function go(s){{V.currentTime=s;V.play()}}
function render(){{let q=Q.value.trim().toLowerCase(),rows=D.rows.filter(x=>!q||(x.zh+' '+x.original).toLowerCase().includes(q));R.innerHTML=rows.length?rows.map(x=>`<div class="line" data-t="${{x.start}}" data-end="${{x.end}}"><span class="time">${{time(x.start)}}</span>${{esc(x.zh||x.original)}}<div class="orig">${{esc(x.original)}}</div></div>`).join(''):'<p class="muted">没有匹配字幕</p>';R.querySelectorAll('.line').forEach(e=>e.onclick=()=>go(+e.dataset.t))}}
C.innerHTML=D.chapters.map(x=>`<div class="chapter"><button class="jump" data-t="${{x.start}}">${{time(x.start)}}</button><b>${{x.number}}. ${{esc(x.title)}}</b><p>${{esc(x.summary)}}</p>${{x.keywords?.length?'<div class="tags">'+x.keywords.map(k=>'<span class="tag">'+esc(k)+'</span>').join('')+'</div>':''}}${{x.questions?.length?x.questions.map(q=>'<p class="question">复习：'+esc(q)+'</p>').join(''):''}}</div>`).join('');C.querySelectorAll('.jump').forEach(e=>e.onclick=()=>go(+e.dataset.t));Q.oninput=render;render();
function save(show=true){{localStorage.setItem(noteKey,N.value);if(show){{$('#saved').textContent='已保存';setTimeout(()=>$('#saved').textContent='',1600)}}}}$('#save').onclick=()=>save();let saveTimer;N.oninput=()=>{{clearTimeout(saveTimer);saveTimer=setTimeout(()=>save(false),700)}};$('#stamp').onclick=()=>{{let stamp=`[${{time(V.currentTime)}}] `,pos=N.selectionStart;N.value=N.value.slice(0,pos)+stamp+N.value.slice(N.selectionEnd);N.focus();N.selectionStart=N.selectionEnd=pos+stamp.length;save(false)}};
$('#speed').onchange=e=>V.playbackRate=+e.target.value;let last=+(localStorage.getItem(progressKey)||0);$('#resume').disabled=!last;$('#resume').onclick=()=>go(last);V.ontimeupdate=()=>{{let t=V.currentTime;$('#progress').textContent='进度 '+time(t);if(Math.floor(t)%5===0)localStorage.setItem(progressKey,t);R.querySelectorAll('.line').forEach(e=>{{let active=t>=+e.dataset.t&&t<+e.dataset.end;e.classList.toggle('active',active);if(active&&!Q.value)e.scrollIntoView({{block:'nearest'}})}})}};
$('#export').onclick=()=>{{save(false);let body=`# ${{D.title}} 学习笔记\n\n${{N.value}}\n`;let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([body],{{type:'text/markdown;charset=utf-8'}}));a.download=D.title+'.学习笔记.md';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}};
</script></html>'''
    output.write_text(document, encoding="utf-8")
    return output
