# dsh-asr-local

给 [`dsh-video-storyboard`](https://github.com/SAOS-3624/dsh-video-storyboard) 用的**本机语音转写服务**：
一个 OpenAI 兼容的 `/v1/audio/transcriptions` 端点，sherpa-onnx + paraformer-zh + ct-transformer 标点 + silero VAD。

不需要 API key，不联网，模型和音频都不出本机。装好之后 `video_storyboard` 出的分镜清单里每一格都会带台词。

> A local, offline, OpenAI-compatible ASR endpoint for `dsh-video-storyboard`.
> sherpa-onnx + paraformer-zh + ct-transformer punctuation + silero VAD. No API key, nothing leaves the machine.

**相关项目**

- 主项目：[SAOS-3624/dsh-video-storyboard](https://github.com/SAOS-3624/dsh-video-storyboard) — 本仓库只是给它补上转写这一块
- 它定的契约：[docs/asr-contract.md](https://github.com/SAOS-3624/dsh-video-storyboard/blob/main/docs/asr-contract.md)
- 它给的三种 ASR 方案：[docs/CONFIG.zh.md §7.1](https://github.com/SAOS-3624/dsh-video-storyboard/blob/main/docs/CONFIG.zh.md)
- 推理引擎：[k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)

---

## 为什么需要它

`dsh-video-storyboard` 用一条很省的办法让模型「看」视频：把画面压成一张宫格图 + 一份带时间码的清单。
**但插件本身不带转写服务**——没配 ASR 时它只能看图说话，能描述「画面上有个女孩在跑」，
说不出她讲了什么。

作者的 [`docs/CONFIG.zh.md`](https://github.com/SAOS-3624/dsh-video-storyboard/blob/main/docs/CONFIG.zh.md)
给了三条路：接现成的 OpenAI 兼容服务、用云端 API、或者自己写一个满足
[`asr-contract.md`](https://github.com/SAOS-3624/dsh-video-storyboard/blob/main/docs/asr-contract.md)
的服务端。**这个仓库是第三条**，按作者的实际布局实现，约 300 行，可以直接跑。

## 30 秒开始

```bash
pip install sherpa-onnx numpy        # 2.x
python3 dl.py                        # 下三个模型，约 500 MB（GitHub 慢就加 ASR_MIRROR，见下）
python3 server-sherpa.py             # 先手动跑一次，确认能起
```

再把它交给插件托管，写 `~/.dsh/asr.json`：

```json
{
  "baseUrl": "http://127.0.0.1:8000/v1",
  "model": "paraformer-zh-2024-03-09",
  "language": "zh",
  "manage": {
    "start": ["/usr/bin/python3", "/absolute/path/to/server-sherpa.py"],
    "idleMs": 5000,
    "waitMs": 90000
  }
}
```

`start` 写**绝对路径**，解释器用你自己 `pip install` 的那个（`which python3` 的输出）。
配好之后不用手动起服务了：调用 `video_storyboard` 时插件会自动拉起，跑完空闲 5 秒自动让服务退出。

验证：

```bash
python3 -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/health').status)"
```

然后随便挑一个有说话的片子：

```js
video_storyboard({ input: "talk.mp4", from: 0, segment: 300 })
```

清单里看到 `transcript="asr"`，格子后面跟着台词，就成了。

## 下载慢 / 连不上 GitHub

模型挂在 GitHub Releases 上，有些网络会超时。用镜像前缀：

```bash
ASR_MIRROR=https://gh-proxy.com/ python3 dl.py
```

`dl.py` 只保留要用的文件——paraformer 官方包里还有一个 786 MB 的 fp32 模型，解开就删。
最后实际占 **约 503 MB**：

| 文件 | 体积 |
|---|---|
| `sherpa-onnx-paraformer-zh-2024-03-09/model.int8.onnx` + `tokens.txt` | 218 MB |
| `sherpa-onnx-punct-ct-transformer-.../model.onnx` + `tokens.json` | 282 MB |
| `silero_vad.onnx` | 1.8 MB |

## 实测（Android 手机，proot Ubuntu 24.04，4 线程）

| 项目 | 数值 |
|---|---|
| 加载模型（asr + 标点） | **3.9 s** |
| 5 分钟音频转写 | **19.8 s**（约 15× 实时） |
| 29 分钟音频整片 | **44 s**（约 40× 实时，按 60 秒切片） |
| 磁盘 | 503 MB |

台式机上会更快（作者 16 核 Windows：45 秒音频 2.1 秒）。

## 它做对的三件事

1. **用 VAD 断句，不按固定秒数切**——硬切会把词切两半，时间戳整体漂移（契约文档里记过一次 13 秒的漂移）。
2. **`start`/`end` 是相对本次上传的音频**——插件自己加窗口偏移。写绝对时间的话每一格都会错位，而且不报错。
3. **标点失败不返回 500**——标点只是可读性，坏了就退回无标点文本照常 200。否则一个装饰性环节能把整条台词链带崩。

另外，`/shutdown` 是给 `manage.idleMs` 用的：回了包就退出，插件不必杀进程。

## 排障

| 现象 | 多半是 |
|---|---|
| `asr-failed · ECONNREFUSED` | 服务没起：`start` 路径不对，或解释器没装 sherpa-onnx |
| `no-key` | `baseUrl` 不是本机地址，插件认为需要 key 但没给 |
| `asr-empty` | 跑了，这段真没人说话（纯画面 / 纯音乐，**正常**） |
| `asr-failed` 且 `/health` 通 | 模型加载失败，看服务端日志 |
| 术语全错 | 加 `hotwords`，**按片子传别写全局**（见下） |

```js
video_storyboard({ input: "lecture.mp4", hotwords: "分镜 切点 悬浮窗 陪看" })
```

热词是双刃剑：拿摄影词去识别材料学讲座只会把结果往错的方向拉。

## 调优

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `ASR_MODEL_DIR` | `./models` | 模型目录 |
| `ASR_PORT` | `8000` | 端口 |
| `ASR_THREADS` | `4` | 每个模型的线程数 |
| `ASR_MODEL_NAME` | `paraformer-zh-2024-03-09` | 回报给客户端的模型名 |
| `ASR_VAD_MODEL` | 自动找 | 显式指定 `silero_vad.onnx` |
| `ASR_MIN_SILENCE` | `0.4` | 多大停顿算一句结束（讲座调 0.6，抢话调 0.2） |
| `ASR_MIN_SPEECH` | `0.25` | 短于这个的气音直接丢 |

`idleMs` 一般保持 5000：sherpa 冷启动几秒，太短会频繁重启；调大到 60000 就是常驻，代价是约 500 MB 内存。

## 换模型

把新的 sherpa-onnx 模型解到 `models/` 下，服务启动时自动找：

- ASR：路径含 `paraformer`、且有 `model.int8.onnx`（没有 int8 才用 `model.onnx`）；
- 标点：路径含 `punct`、且有 `model.onnx`。

换完记得同步改 `asr.json` 的 `model`——manifest 的缓存键里含这个名字，不改可能命中旧缓存。

## 已知取舍

- **中文好，英文术语差**：`storyboard` 会被听成 `styboard`，`see-now` 听成 `c now`。靠 `hotwords` 补。
  （对比：SenseVoiceSmall 中文更容易错成同音字「掐件」，但英文术语反而对得多——两害。）
- **重叠说话 / 唱歌**：VAD 会当噪声丢掉，这是故意的，避免把歌词当台词。
- **标点模型是 ct-transformer**：够用，不如大模型自然。

## License

MIT。模型权重的许可见各自的发布页（sherpa-onnx releases）。

## 致谢

- [`dsh-video-storyboard`](https://github.com/SAOS-3624/dsh-video-storyboard)：契约是它定的，本仓库只是把第三条路填上；
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)：识别、标点、VAD 三件套；
- [silero-vad](https://github.com/snakers4/silero-vad)。
