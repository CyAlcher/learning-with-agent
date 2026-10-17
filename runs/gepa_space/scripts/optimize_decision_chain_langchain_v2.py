#!/usr/bin/env python3
"""GEPA LangChainAdapter 全链路端到端优化「决策链抽取」—— 用 LangChain + LangGraph 包，不用 DSPy。

v2 修正（Max 三问，2026-10-17）：
- 问2「评估为什么出错」：v1 判分把 result 算进「必填非空」，但契约 result 可空 → 判分与契约矛盾。
  根因 = 复用 v2(optimize_anything) 时把 REQUIRED_KEYS 里「已排除 result」的正确细节丢了。
  修 = 拆清「必填非空 5 字段(topic/state/decision/basis/action)」vs「可空 result」，result 不判非空。
- 问3「是真端到端吗」：v1 只到「落盘+判 action 长度」，没真走到「下游消费」。
  修 = 加下游消费者「正放执行」（basis_type→路由档 + action→拆解可执行步骤），
  evaluate 判「过程(结构/落盘) + 最终结果(字段/路由/执行)」全链路，下游 outcome 进分数。

能力边界（问1，LangChainAdapter vs optimize_anything，源码 langchain_adapter.py docstring）：
- rollout_fn 支持：单轮 chat / create_agent 工具 Agent / 自定义 LangGraph 图 / RAG —— 任意 LangChain pipeline。
- 比 optimize_anything 多：真跑 LangChain 生态的 pipeline（含 Agent 多步 rollout + tool calls），
  eval_fn 可 inspect state["messages"] 里的 tool calls；reflective_record_fn 可 override 暴露中间轨迹。
- 适用场景：只有当「要优化的东西本身是 LangChain/LangGraph 对象」时才用；否则 optimize_anything(裸 evaluate)
  更轻。约束：candidate 是 dict[str,str]；rollout_fn 必须返回 state dict；eval_fn 返回 (score, feedback) 二元组
  （不是 optimize_anything 的 (score, side_info)）；需 langchain + langchain-ollama + langgraph。
- 工具 Agent 接入：rollout_fn 里 create_agent(model, tools, system_prompt=candidate["system_prompt"])
  返回 agent.invoke({"messages":[...]}) 的完整 state（含 tool calls），见 vendor examples/langchain_adapter/big_number_arithmetic.py。
  本案例「决策链抽取」本身无需工具，用单节点 LangGraph 图；工具 Agent 场景未在本案例覆盖（是独立延伸）。

模型：qwenwm 14B@9090（task_lm + reflection_lm 都走 langchain init_chat_model('ollama:...')）。
运行：/Users/max/Documents/mini_conda/envs/agent/bin/python3 <本脚本> --sanity | <本脚本>
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
RUN_DIR = "/Users/max/Documents/08_algo_copilot/daily/week23/rsi_project/src/learning-with-agent/runs/gepa_space/runs/decision_chain_langchain_v2"
REGISTRY_DIR = os.path.join(RUN_DIR, "registry")

# 字段契约：拆清「必填非空」vs「可空」（问2 修复核心）
REQUIRED_NONEMPTY = ["topic", "state", "decision", "basis", "action"]  # 5 字段必填非空
OPTIONAL = ["result"]  # result 可空：key 存在即可，值空=日报无结果，不扣分
BASIS_TYPES = ["铁律", "偏好", "证据", "推断"]
WRONG_FIELDS = ["task_id", "target", "steps", "condition", "dependency", "verification", "step", "objective"]
CONTRACT = ("七字段=topic(决策事项)/state(决策前现状)/decision(做什么决定)/basis(为什么)/"
            "basis_type(四选一:铁律/偏好/证据/推断)/action(决策后行动)/result(结果,日报没有留空可空)")
# 满分 12 = 过程(结构3+落盘3) + 最终结果(字段2+路由2+执行2)
FULL_SCORE = 12.0

SEED = {"system_prompt": "从日报中提取决策链，输出 JSON。"}

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

REFLECTION_TEMPLATE = """You are improving a PROMPT that extracts DECISION CHAINS (决策链) from a daily report, so a DOWNSTREAM executor can route and execute each decision.

A decision chain = one complete decision, with EXACTLY these 7 fields (strict JSON):
1. topic: 决策事项(解决什么问题,一句话)
2. state: 状态(决策前的客观现状)
3. decision: 决策(做了什么决定)
4. basis: 依据(为什么这么决策)
5. basis_type: 依据类型,四选一:铁律/偏好/证据/推断
6. action: 行动(决策后做了什么,必须含动作动词,下游要拆解执行)
7. result: 结果(行动结果,日报没有就留空 "")

Output JSON shape MUST be: {"chains": [{"topic":...,"state":...,"decision":...,"basis":...,"basis_type":...,"action":...,"result":...}]}
Do NOT invent other fields (task_id/target/steps/condition/dependency/verification are WRONG).
Do NOT output Markdown or any text around the JSON. Only raw JSON.
topic/state/decision/basis/action MUST be non-empty; result MAY be empty string.

Current prompt:
```
<curr_param>
```

Examples of task inputs, the assistant's outputs, and feedback:
```
<side_info>
```

Propose an improved prompt that defines all 7 fields, forces ONLY-JSON output, and makes every action concretely executable by the downstream executor. Provide the new prompt within ``` blocks."""


# ── LangGraph 图 ─────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(system_prompt: str, task_llm):
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


# ── 下游消费者「正放执行」（问3 核心：结果真的输入给下游使用）──────────────
ACTION_VERBS = ["执行", "推进", "处理", "构建", "产出", "调整", "保留", "确定", "并行", "串行",
                "拆解", "沉淀", "验证", "排期", "评估", "挂边", "抢占", "负责", "分工", "单独",
                "混排", "review", "排序", "安排", "路由"]


def downstream_consume(chains):
    """下游「正放执行」消费决策链：basis_type→路由档 + action→拆解可执行步骤。返回 (routed, executed, plans)。"""
    routed = executed = 0
    plans = []
    for c in chains:
        if not isinstance(c, dict):
            continue
        bt = c.get("basis_type", "")
        # ① 路由：basis_type 四选一 → 处理档（下游按类型分派，非法则路由失败）
        if bt not in BASIS_TYPES:
            continue
        routed += 1
        # ② 执行：action 含动作动词且长度足够 → 能拆成可执行步骤
        action = str(c.get("action", "")).strip()
        if len(action) >= 4 and any(v in action for v in ACTION_VERBS):
            executed += 1
            plans.append(f"[{bt}] {c.get('topic','')}: {action}")
    return routed, executed, plans


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


# ── eval_fn（全链路端到端：过程 + 最终结果）────────────────────────────────
def evaluate(example, state):
    """返回 (score, feedback)。feedback 承载过程+最终结果，是反思唯一通道。"""
    raw = last_message_text(state)

    # 【过程】① 结构化命门：下游 json.loads().get("chains") 能 parse
    try:
        obj = json.loads(raw)
    except Exception as e:
        return 0.01, f"输出不是合法 JSON({type(e).__name__})。必须只输出 {{\"chains\":[...]}}，禁止 Markdown。{CONTRACT}"
    if not (isinstance(obj, dict) and "chains" in obj and isinstance(obj["chains"], list)):
        return 0.01, f"不是 {{\"chains\":[...]}} 外层结构，下游 get('chains') 解析不到。{CONTRACT}"
    chains = obj["chains"]
    score = 3.0  # 结构对
    if not chains:
        return score, f"结构对但 0 条链，没抽出决策链。{CONTRACT}"

    # 【过程】② 落盘真实路径
    key = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    path = os.path.join(REGISTRY_DIR, f"registry_{key}.md")
    exists = False
    try:
        write_registry_md(chains, path)
        exists = os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:
        pass
    if exists:
        score += 3.0

    # 【最终结果】③ 字段契约：必填非空 5 字段（result 可空不判非空，问2修复）
    filled = wrong = 0
    missing = []
    for c in chains:
        if not isinstance(c, dict):
            continue
        miss = [f for f in REQUIRED_NONEMPTY if not str(c.get(f, "")).strip()]
        if miss:
            missing.append("缺[" + ",".join(miss) + "]")
        else:
            filled += 1
        for w in WRONG_FIELDS:
            if w in c:
                wrong += 1
    n = len(chains)
    score += 2.0 * (filled / n)

    # 【最终结果】④ 下游消费 outcome：路由 + 执行（问3 核心）
    routed, executed, plans = downstream_consume(chains)
    score += 2.0 * (routed / n) + 2.0 * (executed / n)

    fb = (f"过程: 结构{'✅' if exists else '✅parse'} 落盘{'✅' if exists else '❌'} 抽出{n}条。"
          f"最终结果: 必填字段完整{filled}/{n} 路由{routed}/{n} 执行{executed}/{n}。")
    if missing:
        fb += " 缺失: " + "; ".join(missing[:2]) + "。"
    if wrong:
        fb += f" 有 {wrong} 个错误字段(如 task_id/steps 不该出现)。"
    if routed < n:
        fb += f" basis_type 不合法导致 {n - routed} 条无法路由。"
    if executed < n:
        fb += f" {n - executed} 条 action 缺动作动词无法执行。"
    fb += f" 正确契约: {CONTRACT}。"
    return score, fb


# ── 数据 ────────────────────────────────────────────────────────────────────
def load_daily(n=3):
    import glob
    days = sorted(glob.glob("/Users/max/Documents/08_algo_copilot/daily/week23/20*.md"))
    if not days:
        days = sorted(glob.glob("/Users/max/Documents/08_algo_copilot/daily/week*/20*.md"))[:n]
    return [{"input": open(d, encoding="utf-8").read()[:1500], "additional_context": {}, "answer": ""}
            for d in days[:n]]


# ── sanity check ────────────────────────────────────────────────────────────
def sanity_check(adapter):
    print("=== sanity check（全链路结果指标 evaluator 区分度）===", flush=True)
    for name, cand in [("烂 seed", SEED), ("好 prompt", {"system_prompt": ORIGINAL_PROMPT})]:
        batch = adapter.evaluate(
            batch=[{"input": TEST_CONTENT, "additional_context": {}, "answer": ""}],
            candidate=cand, capture_traces=False,
        )
        print(f"  {name} 得分 {batch.scores[0]:.1f}/{FULL_SCORE:.0f}", flush=True)
    print("  → 分数 = 过程(结构/落盘) + 最终结果(字段/下游路由/下游执行)，高分⟺下游能消费", flush=True)


# ── 全链路端到端对比：抽取 → 落盘 → 下游消费 ───────────────────────────────
def end_to_end_compare(adapter, best_candidate):
    print("\n=== 全链路端到端对比：优化前 vs 优化后（日报→抽取→落盘→下游消费）===", flush=True)
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
            routed, executed, plans = downstream_consume(chains)
            print(f"  抽出 {len(chains)} 条链 → 下游路由 {routed} 条 / 可执行 {executed} 条：", flush=True)
            for p in plans:
                print(f"    · {p}", flush=True)
            # 落盘核验
            print(f"  已落盘登记表到 {REGISTRY_DIR}/ 下", flush=True)
        except Exception:
            print(f"  原始输出(前300字): {resp[:300]}", flush=True)


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true", help="只跑 sanity check")
    ap.add_argument("--max-metric-calls", type=int, default=12)
    args = ap.parse_args()

    task_llm = init_chat_model(MODEL, base_url=OLLAMA_BASE, temperature=0)
    reflection_llm = init_chat_model(MODEL, base_url=OLLAMA_BASE)
    reflection_lm = make_reflection_lm(reflection_llm)

    adapter = LangChainAdapter(
        rollout_fn=lambda candidate, example: build_graph(candidate["system_prompt"], task_llm)
        .invoke({"messages": [HumanMessage(content=example["input"])]}),
        eval_fn=evaluate,
        num_threads=1,
    )

    if args.sanity:
        sanity_check(adapter)
        return

    trainset = load_daily(2)
    valset = load_daily(1)
    print(f"[sanity] trainset {len(trainset)} 条 / valset {len(valset)} 条真实日报", flush=True)
    sanity_check(adapter)

    print("\n=== 开始 optimize（LangChainAdapter 全链路端到端）===", flush=True)
    result = optimize(
        seed_candidate=SEED,
        trainset=trainset,
        valset=valset if valset else None,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_prompt_template=REFLECTION_TEMPLATE,
        max_metric_calls=args.max_metric_calls,
        run_dir=RUN_DIR,
        seed=0,
    )

    best = result.best_candidate
    print(f"\n=== 优化结果 ===\n最佳候选 system_prompt：\n{best['system_prompt']}", flush=True)
    end_to_end_compare(adapter, best)


if __name__ == "__main__":
    main()
