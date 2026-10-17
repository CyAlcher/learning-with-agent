# Learning with Agent

> 把学习当工程——列清单、每天固定大块时间、用间隔重复钉进长期记忆、边学边写代码验证「真会了」，而不是读完就算。

## 模仿的锚点

**jwasham/coding-interview-university**（36 万 ⭐ / 8.5 万 fork，全球第一「公开学习」仓库）

作者 John Washam 自学转岗进 Amazon，他的一句话方法论：

- **清单化**：列一张分科清单逐个打勾，避免「感觉学了其实没学」
- **间隔重复（Anki）**：把知识点钉进长期记忆，对抗遗忘
- **动手验证**：`Don't just read about it — code it`，读完不算会，写出来才算
- **聚焦**：他复盘 "I wasted a lot of time on things I didn't need to know" → 别贪多

## 学习记录规则

每个主题一个目录，四条线记清楚：

1. **学什么** —— 目标 + 一句话结论
2. **关键结论** —— 底层机制，不是表面结论
3. **动手验证** —— 写出来的代码 / 跑通的证据
4. **复盘** —— 哪一步卡了、下次怎么更快

## 提交方式（简单）

本意是「对外输出所学」，有进展就提交，不做每日打卡：

1. 更新 `PROJECT_STATUS.md`，记这次做了什么
2. `git add -A && git commit -m "YYYYMMDD 一句话说明"`
3. `git push`

想看提交量时（可选）：
```bash
git log --since="today 00:00" --oneline          # 今日提交
git log --since="7 days ago" --oneline | wc -l   # 本周提交数
```

## 目录

（主题按时间追加）
