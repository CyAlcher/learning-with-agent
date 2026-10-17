# Learning Meta-Agent

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

## 代码提交纪律

### 提交标准
- **每日 ≥1 commit**：哪怕一行，保持「今日所学」有痕
- **每周 ≥5 commit**：四线（学 / 结论 / 验证 / 复盘）各至少覆盖一次

### 提交粒度
小步提交，一个 commit = 一个学习单元，不攒大包。

### commit message 规范
`YYYYMMDD 主题：一句话结论`
例：`20261017 meta-agent 学习：jwasham = 清单化 + 间隔重复 + 动手验证`

### 每日复盘（睡前 / 收尾）
```bash
git log --since="today 00:00" --oneline            # 今日提交了几个 commit
git log --since="today 00:00" --numstat --format= | awk '{add+=$1;del+=$2} END{printf "+%d -%d\n", add, del}'   # 今日增删行
```

### 每周复盘（周日）
```bash
git log --since="7 days ago" --oneline | wc -l     # 本周提交数
git log --since="7 days ago" --numstat --format= | awk '{add+=$1;del+=$2} END{printf "+%d -%d\n", add, del}'   # 本周增删行
```

## 目录

（主题按时间追加）
