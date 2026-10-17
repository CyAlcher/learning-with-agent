#!/usr/bin/env python3
"""GEPA LangChainAdapter 端到端优化「决策链抽取」—— 用 LangChain + LangGraph 包真实任务，不用 DSPy。

核心（Max 钉死，2026-10-17）：
- 端到端：优化的 prompt 真跑 LangGraph 图抽日报 → 判「下游能消费的成果」→ 分数。
- 分数 = 结果指标（不是「prompt 好不好看」）：结构{"chains":[...]} + 落盘 + 七字段全填 +
  basis_type 四选一 + action 可执行。满分 12。
- 高分 ⟺ 结果符合人预期：下游「正放执行」能 json.loads().get("chains") 解析、能从真实路径查到、能拆 action 执行。
- 不用 DSPy：LangChainAdapter 与 dspy_adapter 并列，零依赖。

模型：qwenwm 14B@9090（task_lm + reflection_lm 都走 langchain init_chat_model('ollama:...')）。

运行：
  cd <gepa-main> && PYTHONPATH=$PWD/src /Users/max/Documents/mini_conda/envs/agent/bin/python3 <本脚本> --sanity   # 先验判分区分度
  cd <gepa-main> && PYTHONPATH=$PWD/src /Users/max/Documents/mini_conda/envs/agent/bin/python3 <本脚本>            # 完整 optimize + 端到端对比
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from gepa import optimize
from gepa.adapters.langchain_adapter import LangChainAdapter, last_message_text, make_reflection_lm

# ── 常量 ──────────────────────────────────────────────────────────────────────
OLLAMA_BASE = "http://127.0.0.1:9090"
MODEL = "ollama:qwen3:14b-32k"
RUN_DIR = "/Users/max/Documents/08_algo_copilot/daily/week23/rsi_project/src/learning-with-agent/runs/gepa_space/runs/decision_chain_langchain"
REGISTRY_DIR = os.path.join(RUN_DIR, "registry")

# 七字段契约（Feedback + 反思模板共用，保证一致，防 14B 脑补成任务链）
REQUIRED_FIELDS = ["topic", "state", "decision", "basis", "basis_type", "action", "result"]
BASIS_TYPES = ["铁律", "偏好", "证据", "推断"]
WRONG_FIELDS = ["task_id", "target", "steps", "condition", "dependency", "verification", "step", "objective"]
CONTRACT = ("七字段=topic(决策事项)/state(决策前现状)/decision(做什么决定)/basis(为什么)/"
            "basis_type(四选一:铁律/偏好/证据/推断)/action(决策后行动)/result(结果,日报没有留空)")
FULL_SCORE = 12.0  # 结构3 + 落盘3 + 字段2 + 类型2 + 可执行2

SEED = {"system_prompt": "从日报中提取决策链，输出 JSON。"}

# 好 prompt（v2 sanity check 用，七字段完整版）
ORIGINAL_PROMPT = """你是决策链抽取器。从日报抽取「人的决策链」，每条链 = 一次完整决策。
每条决策链字段（严格 JSON）：
- topic: 决策事项(解决什么问题,一句话)
- state: 状态(决策前的客观现状)
- decision: 决策(做了什么决定)
- basis: 依据(为什么这么决策,一句话)
- basis_type: 依据类型,四选一:铁律(写死规则)/偏好(反复纠正的倾向)/证据(可溯源文件/数据)/推断(未验证)
- action: 行动(决策后做了什么)
- result: 结果(行动的结果,日报里没有就留空 "")
只输出 JSON,不要其他文字:
{"chains": [{"topic":"...","state":"...","decision":"...","basis":"...","basis_type":"偏好","action":"...","result":"..."}]}"""

# 人工可读验证素材：真实日报「追加方案」段（决策密集）
TEST_CONTENT = """今天的任务编排：
- 中断槽：任务4（不确定性运维支持）单独保留，随时抢占，不和深度任务混排。
- 主业务串行链：2 -> 1/8 并行 -> 3。先把任务2（销量预测大模型2期）做成统一结论和数据口径，再并行产出任务1（badcase报告）、任务8（sku评估），最后做任务3（产品化排期）。
- 自动化资产链：6 -> 7 -> 5，9 并行挂边。先把任务6做成可复制工作流，再把任务7沉淀成模板/RAG资产，再用任务5做实验验证；任务9作为外部benchmark不阻塞主线。
- 结论：今天真正的P0是 4+2+6，其它都围绕这3个任务做并行支持。

人与AI分工：
- 你只做3件事：定优先级、做关键判断、做最终验收。
- codex cli 负责高歧义任务：拆解、追问、改写、review、定下一步。
- codex exec 负责低歧义任务：批量解析、生成模板、跑build/test/demo、产出日报草稿。
- 标准是 1人主控 + 多exec工人 + 最后review。"""

# 反思模板：锁死七字段契约（<curr_param>=当前prompt, <side_info>=Inputs/Generated Outputs/Feedback 三段）
REFLECTION_TEMPLATE = """You are improving a PROMPT that extracts DECISION CHAINS (决策链) from a daily report.

A decision chain = one complete decision, with EXACTLY these 7 fields (strict JSON):
1. topic: 决策事项(解决什么问题,一句话)
2. state: 状态(决策前的客观现状)
3. decision: 决策(做了什么决定)
4. basis: 依据(为什么这么决策)
5. basis_type: 依据类型,四选一:铁律/偏好/证据/推断
6. action: 行动(决策后做了什么)
7. result: 结果(行动结果,日报没有就留空 "")

Output JSON shape MUST be: {"chains": [{"topic":...,"state":...,"decision":...,"basis":...,"basis_type":...,"action":...,"result":...}]}
Do NOT invent other fields (task_id/target/steps/condition/dependency/verification are WRONG).
Do NOT output Markdown or any text around the JSON. Only raw JSON.

Current prompt:
```
<curr_param>
```

Examples of task inputs, the assistant's outputs, and feedback:
```
<side_info>
```

Propose an improved prompt that defines all 7 fields above and forces ONLY-JSON output with ONLY those 7 fields. Provide the new prompt within ``` blocks."""


# ── LangGraph 图：extract 节点注入 system prompt 跑 LLM 抽取 ──────────────────
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(system_prompt: str, task_llm):
    """最小 LangGraph：单节点 extract，注入 candidate 的 system_prompt 当抽取指令。"""

    def extract_node(state):
        msgs = [SystemMessage(content=system_prompt)] + list(state["messages"])
        resp = task_llm.invoke(msgs)
        if not isinstance(resp, AIMessage):
            resp = AIMessage(content=str(getattr(resp, "content", resp)))
        return {"messages": [resp]}

    g = StateGraph(GraphState)
    g.add_node("extract", extract_node)
    g.add_edge(START, "extract")
    g.add_edge("extract", END)
    return g.compile()


# ── 落盘（结果必须真实写到路径，下游才能查到）──────────────────────────────
def write_registry_md(chains, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# 决策链登记表", "",
        "| 决策链ID | 决策事项 | 决策链(state→decision→basis→action→result) | 依据类型 |",
        "|---|---|---|---|",
    ]
    for i, c in enumerate(chains, 1):
        chain = " → ".join(str(c.get(k, "")) for k in ["state", "decision", "basis", "action", "result"])
        chain = chain.replace("|", "\\|")
        lines.append(f"| dc-{i:02d} | {c.get('topic','')} | {chain} | {c.get('basis_type','')} |")
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    return path


# ── eval_fn（端到端结果指标：判真跑产出，不判 prompt 文本）──────────────────
def evaluate(example, state):
    """返回 (score, feedback)。feedback 是反思唯一通道，必须承载完整契约。"""
    raw = last_message_text(state)

    # ① 结构化命门：下游 json.loads().get("chains") 能 parse
    try:
        obj = json.loads(raw)
    except Exception as e:
        # 结构错返回 0.01 非 0（GEPA 帕累托前沿空集会 assert 崩，坑8）
        return 0.01, f"输出不是合法 JSON({type(e).__name__}: {e})。必须只输出 {{\"chains\":[...]}}，禁止 Markdown 包裹。{CONTRACT}"
    if not (isinstance(obj, dict) and "chains" in obj and isinstance(obj["chains"], list)):
        return 0.01, f"不是 {{\"chains\":[...]}} 外层结构，下游 get('chains') 解析不到。{CONTRACT}"

    chains = obj["chains"]
    score = 3.0  # 结构对

    if not chains:
        return score, f"结构对但 0 条链，没抽出决策链。决策链=一次完整决策。{CONTRACT}"

    # ② 落盘存在（结果真实落到路径）
    key = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    path = os.path.join(REGISTRY_DIR, f"registry_{key}.md")
    exists = False
    try:
        write_registry_md(chains, path)
        exists = os.path.exists(path) and os.path.getsize(path) > 0
    except Exception as e:
        pass
    if exists:
        score += 3.0

    # ③ 下游消费约束：字段全填 + 类型四选一 + action 可执行 + 无错误字段
    filled = bt_ok = executable = wrong = 0
    missing = []
    for c in chains:
        if not isinstance(c, dict):
            continue
        miss = [f for f in REQUIRED_FIELDS if not str(c.get(f, "")).strip()]
        if miss:
            missing.append("缺[" + ",".join(miss) + "]")
        else:
            filled += 1
        if c.get("basis_type", "") in BASIS_TYPES:
            bt_ok += 1
        if len(str(c.get("action", "")).strip()) >= 4:
            executable += 1
        for w in WRONG_FIELDS:
            if w in c:
                wrong += 1
    n = len(chains)
    score += 2.0 * (filled / n) + 2.0 * (bt_ok / n) + 2.0 * (executable / n)

    fb = f"抽出 {n} 条链。"
    if missing:
        fb += " 缺失: " + "; ".join(missing[:2]) + "。"
    if wrong:
        fb += f" 有 {wrong} 个错误字段(如 task_id/steps/target 不该出现)。"
    fb += (f" 七字段完整度 {filled}/{n}，类型合法 {bt_ok}/{n}，action 可执行 {executable}/{n}。"
           f"正确契约: {CONTRACT}。")
    return score, fb


# ── 数据：真实日报文件 ──────────────────────────────────────────────────────
def load_daily(n=3):
    import glob
    days = sorted(glob.glob("/Users/max/Documents/08_algo_copilot/daily/week23/20*.md"))
    if not days:
        days = sorted(glob.glob("/Users/max/Documents/08_algo_copilot/daily/week*/20*.md"))[:n]
    return [{"input": open(d, encoding="utf-8").read()[:1500], "additional_context": {}, "answer": ""}
            for d in days[:n]]


# ── sanity check：结果指标 evaluator 能否区分「烂 seed」和「好 prompt」───────
def sanity_check(adapter):
    print("=== sanity check（结果指标 evaluator 区分度）===", flush=True)
    for name, cand in [("烂 seed", SEED), ("好 prompt", {"system_prompt": ORIGINAL_PROMPT})]:
        batch = adapter.evaluate(
            batch=[{"input": TEST_CONTENT, "additional_context": {}, "answer": ""}],
            candidate=cand, capture_traces=False,
        )
        s = batch.scores[0]
        print(f"  {name} 得分 {s:.1f}/{FULL_SCORE:.0f}", flush=True)
    print("  → 分数高 = 抽出的链越能被下游消费（结构对/字段全/类型合法/action 可执行）", flush=True)


# ── 端到端对比：优化前后抽同一段日报，人工可读验证「高分=结果优」───────────
def end_to_end_compare(adapter, best_candidate):
    print("\n=== 端到端对比：优化前 vs 优化后，抽同一段真实日报 ===", flush=True)
    for name, cand in [("优化前(seed)", SEED), ("优化后(best)", best_candidate)]:
        batch = adapter.evaluate(
            batch=[{"input": TEST_CONTENT, "additional_context": {}, "answer": ""}],
            candidate=cand, capture_traces=False,
        )
        s = batch.scores[0]
        resp = last_message_text(batch.outputs[0]["state"])
        print(f"\n【{name}】得分 {s:.1f}/{FULL_SCORE:.0f}", flush=True)
        try:
            chains = json.loads(resp).get("chains", [])
            print(f"  抽出 {len(chains)} 条链：", flush=True)
            for c in chains:
                print(f"    · {c.get('topic','?')} | 决定:{c.get('decision','?')} | 依据:{c.get('basis_type','?')} | 行动:{c.get('action','?')}", flush=True)
        except Exception:
            print(f"  原始输出(前300字): {resp[:300]}", flush=True)


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true", help="只跑 sanity check（验判分区分度 + 图能跑）")
    ap.add_argument("--max-metric-calls", type=int, default=12)
    args = ap.parse_args()

    # task_llm（确定性抽取）+ reflection_llm（创造性变异），都走 langchain 连 9090
    task_llm = init_chat_model(MODEL, base_url=OLLAMA_BASE, temperature=0)
    reflection_llm = init_chat_model(MODEL, base_url=OLLAMA_BASE)
    reflection_lm = make_reflection_lm(reflection_llm)

    adapter = LangChainAdapter(
        rollout_fn=lambda candidate, example: build_graph(candidate["system_prompt"], task_llm)
        .invoke({"messages": [HumanMessage(content=example["input"])]}),
        eval_fn=evaluate,
        num_threads=1,  # 14B 并发仅 2 路最优，串行最稳
    )

    if args.sanity:
        sanity_check(adapter)
        return

    trainset = load_daily(2)
    valset = load_daily(1)
    print(f"[sanity] trainset {len(trainset)} 条 / valset {len(valset)} 条真实日报", flush=True)
    sanity_check(adapter)

    print("\n=== 开始 optimize（LangChainAdapter 端到端）===", flush=True)
    result = optimize(
        seed_candidate=SEED,
        trainset=trainset,
        valset=valset if valset else None,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_prompt_template=REFLECTION_TEMPLATE,  # 锁死七字段，防 14B 脑补
        max_metric_calls=args.max_metric_calls,
        run_dir=RUN_DIR,
        seed=0,
    )

    best = result.best_candidate
    print(f"\n=== 优化结果 ===", flush=True)
    print(f"最佳候选 system_prompt：\n{best['system_prompt']}", flush=True)

    # 端到端对比：证明分数高 = 结果符合人预期
    end_to_end_compare(adapter, best)


if __name__ == "__main__":
    main()
