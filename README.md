# 我的 skills

自己写的 Claude Code skills。每个文件夹一个 skill。

| skill | 做什么 |
|---|---|
| [storyloom](storyloom/SKILL.md) | 写剧本：大纲、分集、悬疑案子、故事图谱，给导演看之前先盲测和找茬 |
| [manju](manju/SKILL.md) | 拍剧：分镜、Codex 出图、远程 5090 拍真人戏、配音、剪辑合成、自检 |

storyloom 把剧本写对，manju 把剧本拍出来。

## 本机怎么用

仓库放在 `~/skills/`，`~/.claude/skills/` 里的同名文件夹是指过来的链接，改这里的文件就等于改正在用的 skill。

加一个新 skill：

```bash
mkdir ~/skills/新名字    # 里面写 SKILL.md
ln -s ~/skills/新名字 ~/.claude/skills/新名字
```

## 密钥不进仓库

AutoDL 口令、ComfyUI 地址这些放在 `~/.config/manju/`，不放在这里。
