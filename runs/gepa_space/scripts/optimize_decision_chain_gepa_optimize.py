#!/usr/bin/env python3
"""gepa.optimize 核心 API 跑通 v2 —— 补上「评估→变异」契约通道（端到端闭环）。

v1 的教训（见 11_GEPA清单 维度 A 核心）：走 optimize 通道时，契约唯一入口是
evaluate 的 Feedback（无 objective/background 兜底），v1 的 Feedback 只写「字段缺失」
没写「缺哪几个字段 + 语义」→ 14B 反思脑补成「任务链」。

v2 三处补契约：
  ① evaluate Feedback 承载完整七字段契约（缺哪些 + 语义 + 错误字段）
  ② evaluate 判分严格版（错误字段 task_id/steps 扣分）
  ③ reflection_prompt_template 锁死七字段（api.py 的 <curr_param>+<side_info> 通道，双保险）

模型走 qwenwm 14B@9090（本地 ollama 仅跑 nomic-embed-text，4B 生成禁用）。

运行:
  cd gepa-main && PYTHONPATH=$PWD envs/agent/bin/python3 <本脚本>
"""
import glob
import json
import os
import urllib.request

os.environ["OLLAMA_API_BASE"] = "http://127.0.0.1:9090"  # qwenwm 隧道 14B

import gepa

print(f"[sanity] gepa 版本: {getattr(gepa, '__version__', '?')}", flush=True)
from gepa import optimize  # noqa: E402
from gepa.adapters.default_adapter.default_adapter import EvaluationResult  # noqa: E402

ROOT = "/Users/max/Documents/08_algo_copilot/daily"
MODEL = "ollama/qwen3:14b-32k"
RUN_DIR = "/Users/max/Documents/08_algo_copilot/daily/week23/rsi_project/src/learning-with-agent/runs/gepa_space/runs/decision_chain_gepa_optimize"

SEED = {"prompt": "从日报中提取决策链，输出 JSON。"}

REQUIRED_FIELDS = ["topic", "state", "decision", "basis", "basis_type", "action", "result"]
BASIS_TYPES = ["铁律", "偏好", "证据", "推断"]
WRONG_FIELDS = ["task_id", "target", "steps", "condition", "dependency", "verification", "step", "objective"]

# 七字段契约（Feedback + 反思模板共用，保证契约一致）
CONTRACT = ("七字段=topic(决策事项)/state(决策前现状)/decision(做什么决定)/basis(为什么)/"
            "basis_type(四选一:铁律/偏好/证据/推断)/action(决策后行动)/result(结果,日报没有留空)")


def load_daily(n=2):
    days = sorted(glob.glob(f"{ROOT}/week23/20*.md"))[:n]
    return [{"input": open(d, encoding="utf-8").read()[:1500], "additional_context": {}, "answer": ""} for d in days]


def evaluate(data, response):
    """结果指标严格版：判产出 JSON 是不是合法七字段决策链，Feedback 承载完整契约。"""
    try:
        obj = json.loads(response)
    except Exception as e:
        return EvaluationResult(0.01, f"输出不是合法 JSON({e})。必须只输出 {{\"chains\":[...]}}，禁止 Markdown。{CONTRACT}", None)

    chains = obj.get("chains", [])
    if not chains:
        return EvaluationResult(0.1, f"chains 为空，没抽出决策链。决策链=一次完整决策。{CONTRACT}", None)

    filled = 0
    wrong = 0
    missing = []
    for c in chains[:3]:
        miss = [f for f in REQUIRED_FIELDS if not c.get(f)]
        if miss:
            missing.append("缺[" + ",".join(miss) + "]")
        for f in REQUIRED_FIELDS:
            if c.get(f):
                filled += 1
        if c.get("basis_type") in BASIS_TYPES:
            filled += 1
        for w in WRONG_FIELDS:
            if w in c:
                wrong += 1

    completeness = filled / (len(chains[:3]) * 8)
    penalty = min(0.3, wrong * 0.05)
    score = max(0.15, 0.2 + 0.8 * completeness - penalty)

    fb = f"抽出 {len(chains)} 条链，七字段完整度 {completeness:.0%}。"
    if missing:
        fb += " " + "; ".join(missing[:2]) + "。"
    if wrong:
        fb += f" 有 {wrong} 个错误字段(如 task_id/steps/target 是错的,不该出现)。"
    fb += f" 正确契约:{CONTRACT}。"
    return EvaluationResult(score, fb, None)


def sanity_check(trainset):
    body = json.dumps({
        "model": "qwen3:14b-32k",
        "messages": [{"role": "user", "content": SEED["prompt"] + "\n\n日报内容:\n---\n" + trainset[0]["input"] + "\n---"}],
        "stream": False, "format": "json", "options": {"temperature": 0},
    }).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:9090/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    msg = json.loads(urllib.request.urlopen(req, timeout=180).read())["message"]["content"]
    ev = evaluate(trainset[0], msg)
    print(f"[sanity] seed 抽第1条日报 → 得分 {ev.score:.2f} | {ev.feedback[:100]}", flush=True)


# 反思模板：锁死七字段契约（<curr_param>=当前prompt, <side_info>=评估反馈）
REFLECTION_TEMPLATE = """You are improving a prompt for extracting DECISION CHAINS (决策链) from daily reports.

A decision chain = one complete decision, with EXACTLY these 7 fields (Chinese, strict JSON):
1. topic: 决策事项(解决什么问题,一句话)
2. state: 状态(决策前的客观现状)
3. decision: 决策(做了什么决定)
4. basis: 依据(为什么这么决策,一句话)
5. basis_type: 依据类型,四选一:铁律(写死规则)/偏好(反复纠正的倾向)/证据(可溯源文件数据)/推断(未验证)
6. action: 行动(决策后做了什么)
7. result: 结果(行动的结果,日报没有就留空 "")

Output JSON shape MUST be: {"chains": [{"topic":...,"state":...,"decision":...,"basis":...,"basis_type":...,"action":...,"result":...}]}
Do NOT invent other fields (like task_id/target/steps/condition/dependency/verification). Those are WRONG.

Current prompt:
```
<curr_param>
```

Evaluation feedback (what went wrong and why):
```
<side_info>
```

Propose an improved prompt that defines all 7 fields above and outputs JSON with ONLY those 7 fields. Provide the new prompt within ``` blocks."""


def main():
    trainset = load_daily(2)
    print(f"[sanity] dataset: {len(trainset)} 条真实日报（每条截前1500字）", flush=True)
    sanity_check(trainset)

    optimize(
        seed_candidate=SEED,
        trainset=trainset,
        valset=None,
        task_lm=MODEL,
        evaluator=evaluate,
        reflection_lm=MODEL,
        reflection_prompt_template=REFLECTION_TEMPLATE,  # ③ 锁死七字段
        max_metric_calls=12,
        run_dir=RUN_DIR,
    )


if __name__ == "__main__":
    main()
