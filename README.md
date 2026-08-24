# video-translator 影片自动翻译字幕工具

为本地影片自动生成 **原文 + 中文** 双语 SRT 字幕。识别在本地运行；翻译支持 **大模型 API（推荐）** 与 **本地离线** 双引擎。

> 无需 root、无需显卡、无需科学上网——便携版 Python + 国内镜像模型下载，解压即用。

## 原理

```
影片 ──> 提取音频(PyAV) ──> 语音识别(faster-whisper) ──> 翻译引擎 ──> 双语SRT
                                                        ├─ api   大模型API(默认优先)
                                                        └─ local argos离线(兜底)
```

- 支持多语言源（英/日/韩/法/德/俄等），自动检测语言
- API 引擎走 OpenAI 兼容接口，批量翻译、限流退避、异常时自动用本地引擎兜底，字幕永不因 API 故障而失败
- 没有直达中文翻译包的小语种会自动经英语中转
- 所有本地模型从国内镜像下载（hf-mirror / gh-proxy）
- 内置多路补丁：禁用 stanza 联网（防弱网卡死）、轻量分句替代（防部分语言包资源残缺）

## 快速开始

**图形界面（推荐）：** 在应用菜单搜「影片中文字幕生成器」，或运行 `vt-gui`——选片、点开始即可。

**命令行方式：**

```bash
# 1. (可选但推荐) 配置大模型API以获得高质量译文
cp .env.example .env    # 编辑 .env, 填入 VT_API_KEY

# 2. 生成双语字幕
./vt 影片.mp4
```

## 翻译引擎配置

编辑项目目录下的 `.env`（参考 `.env.example`）：

| 变量 | 说明 | 默认 |
|------|------|------|
| `VT_API_KEY` | API Key，留空则自动用本地引擎 | 空 |
| `VT_API_BASE` | OpenAI 兼容接口地址 | DeepSeek |
| `VT_API_MODEL` | 模型名 | deepseek-chat |
| `VT_BATCH_SIZE` | 每次请求翻译的条数 | 20 |
| `VT_BATCH_DELAY` | 批次间隔秒数(免费中转站建议 20~60) | 15 |

已适配的供应商（改 `VT_API_BASE`/`VT_API_MODEL` 即可）：DeepSeek / Kimi / 通义千问 / OpenAI / 各类 OpenAI 兼容中转站 / 本地 ollama。

也可以用命令行参数强制指定：

```bash
./vt --engine api 影片.mp4     # 强制API(未配Key会报错)
./vt --engine local 影片.mp4  # 强制离线
```

费用参考：deepseek-chat 翻译一部两小时电影约 2~5 万字，通常不到 0.5 元。
免费中转站普遍有严格限流(403)，请调大 `VT_BATCH_DELAY`、调小 `VT_BATCH_SIZE`。

## 使用

```bash
# 基本用法：在影片旁边生成 同名.srt
./vt 影片.mp4

# 批量处理
./vt 第1集.mkv 第2集.mkv

# 换更大的识别模型（更准但更慢）
./vt -m medium 影片.mp4

# 只输出中文字幕，不带原文
./vt --zh-only 影片.mp4

# 片段试看效果：从第600秒起只处理180秒
./vt 影片.mp4 --start-sec 600 --dur-sec 180   # 输出 影片.片段600-780.srt

# 长片后台运行(脱离终端, 日志在 last-run.log)
./vtd 电影.mkv

# 强制重新识别(默认有 .segs.json 识别缓存)
./vt --fresh 影片.mp4
```

生成的 `影片.srt` 放在影片同目录，用 PotPlayer / VLC / mpv 打开影片即可自动加载。
mpv 手动加载：`mpv --sub-file=影片.srt 影片.mp4`

支持格式：mp4 / mkv / avi / mov / flv / wmv / ts / webm 等视频，mp3 / m4a / wav / flac 等音频均可。

## Whisper 模型选择

| 模型 | 大小 | 说明 |
|------|------|------|
| tiny | ~75MB | 最快，准确率低 |
| base | ~145MB | 快 |
| **small**（默认） | ~480MB | 推荐平衡之选 |
| medium | ~1.5GB | 更准，CPU 上较慢 |
| large-v3 | ~3GB | 最准最慢 |

无显卡的机器建议 small；追求质量且时间充裕用 medium。

## 常见问题

**Q: 翻译质量不满意？**
识别走本地（Whisper 足够准），翻译走 API（质量关键）。免费离线引擎仅作兜底。

**Q: API 频繁 403？**
中转站限流。调大 `VT_BATCH_DELAY`(如60)、调小 `VT_BATCH_SIZE`(如10)。程序遇 403 会自动长退避重试，全部失败时回退本地引擎。

**Q: 卡在"识别完成"不动？**
旧版本曾因 stanza 组件联网检查更新在弱网下挂死，现已内置离线补丁。请拉取最新代码。

**Q: 没有 sudo/root 能装吗？**
能。仓库自带便携方案：`runtime/` 为免安装 Python（见下方安装说明），所有依赖装在用户目录。

**Q: 处理速度如何？**
16 核 CPU + small 模型约为片长的 0.3~1 倍时间（VAD 会跳过无语音段）；有 NVIDIA 显卡会快数倍。

## 从零安装（新机器）

```bash
git clone https://github.com/<you>/video-translator.git
cd video-translator

# 1. 准备便携Python 3.11 (免root, 已含tkinter)
mkdir -p runtime && curl -L -o /tmp/cp.tar.gz \
  "https://github.com/astral-sh/python-build-standalone/releases/download/20241016/cpython-3.11.10%2B20241016-x86_64-unknown-linux-gnu-install_only.tar.gz" \
  && tar -xzf /tmp/cp.tar.gz -C runtime

# 2. 建虚拟环境装依赖 (国内走清华源; torch 用CPU版避免下载CUDA大包)
runtime/python/bin/python3.11 -m venv .venv
.venv/bin/pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
.venv/bin/pip install "https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.6.0%2Bcpu-cp311-cp311-linux_x86_64.whl"
.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple argostranslate faster-whisper

# 3. 试跑
./vt 影片.mp4
```

海外网络环境可去掉 `-i` 镜像参数直接官方源安装。

## 环境变量（一般不用改）

| 变量 | 默认 | 用途 |
|------|------|------|
| `HF_ENDPOINT` | hf-mirror.com | Whisper 模型下载镜像 |
| `HF_HUB_DISABLE_XET` | 1 | 兼容镜像站, 禁用Xet协议 |
| `VT_GH_PROXY` | gh-proxy.com | GitHub 资源代理 |

## 目录结构

```
video-translator/
├── vt                  # 命令行入口
├── vt-gui              # 图形界面入口
├── vtd                 # 后台守护式入口(长片用)
├── gui.py              # tkinter 图形界面
├── translate_video.py  # 主程序
├── .env.example        # API 配置模板(复制为 .env 使用, .env已被gitignore)
├── requirements.txt
└── runtime/            # 便携Python解释器(gitignore, 见安装说明)
```

## License

MIT
