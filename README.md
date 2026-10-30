# Konamiu 的 skills

用 AI 做短剧的两个 Claude Code skill：**storyloom 把剧本写对，manju 把剧本拍出来。**

Two Claude Code skills for AI short dramas: **storyloom** writes and checks the script, **manju** turns it into a finished episode (storyboard, images, voices, editing, QC). Docs are in Chinese.

| skill | 做什么 | 装上就能用吗 |
|---|---|---|
| [storyloom](storyloom/SKILL.md) | 写剧本：全集大纲、分集、悬疑案子，“故事图谱”管真相、线索和时间线，给导演看之前先派子代理盲测和找茬 | 能 |
| [manju](manju/SKILL.md) | 拍剧：分镜 → 出图 → 配音 → 剪辑合成 → 自检，动态漫和真人短剧两种做法 | 动态漫要先装几个工具；真人模式还要一台租的显卡，见下面 |

## 安装

```bash
npx skills add Konamiu/skills
```

只装其中一个：

```bash
npx skills add Konamiu/skills --skill storyloom
```

也可以手动装：把 `storyloom/`、`manju/` 两个文件夹复制进 `~/.claude/skills/`。

装好后在 Claude Code 里直接说“帮我写个短剧大纲”“把这一集拍出来”，它会自己用上。

## manju 要准备什么

按要做的事从少到多：

| 要做的事 | 需要 |
|---|---|
| 写分镜、检查分镜 | [uv](https://docs.astral.sh/uv/)（Python 依赖它会自己装） |
| 出定妆照、场景图、首帧（免费） | 登录好的 [Codex CLI](https://github.com/openai/codex)，用它的出图 |
| 配音、剪辑合成 | ffmpeg、中文字体（macOS 自带；Linux 装 Noto Sans CJK，或用 `MJ_FONT` 指定） |
| 自检时核对台词 | [whisper.cpp](https://github.com/ggml-org/whisper.cpp) 的 `whisper-cli` 和一个模型（推荐 ggml-large-v3-turbo） |
| 真人模式：一场一场拍真人戏 | 一台跑 ComfyUI 的大显存显卡（作者用 AutoDL 租的 5090），装好 MiniMax-H3，见 [真人模式.md](manju/references/真人模式.md) |
| 真人模式：设计声音、配乐 | 同一台 ComfyUI 上装 Qwen3-TTS 和 MiniMax Music 3 的节点 |
| 换声音、口型不动（`redub`） | 一个装了 demucs 的 Python |

macOS 上一条命令装好前三样：

```bash
brew install uv ffmpeg whisper-cpp
```

### 本机设置

地址、实例编号和模型路径都写在 `~/.config/manju/config.json`，只存在你自己的电脑上。用到哪项写哪项：

```json
{
  "comfy_url": "https://你的-comfyui-地址",
  "autodl_instance": "pro-xxxxxxxxxxxx",
  "whisper_model": "/path/to/ggml-large-v3-turbo.bin",
  "demucs_python": "/path/to/demucs-env/bin/python"
}
```

- 不写 `whisper_model`，就去 `~/.cache/whisper/` 里找。
- 也可以用环境变量 `MJ_COMFY_URL`、`MJ_WHISPER_MODEL`、`MJ_DEMUCS_PY`。
- 用 AutoDL 开关机：在控制台复制开发者 Token，存进 `~/.config/manju/autodl_token`，然后 `chmod 600` 这个文件。

**这些地址和口令别放进仓库，也别发给别人。** ComfyUI 的地址谁拿到都能用你的显卡。

## 作者自己的用法

仓库放在 `~/skills/`。`~/.claude/skills/` 和 `~/.agents/skills/` 里的同名文件夹是指过来的链接，改这里的文件就是改正在用的 skill。

加一个新 skill：

```bash
mkdir ~/skills/新名字    # 里面写 SKILL.md
ln -s ~/skills/新名字 ~/.claude/skills/新名字
```

## 许可

[MIT](LICENSE)
