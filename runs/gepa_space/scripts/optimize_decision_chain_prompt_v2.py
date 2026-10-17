#!/usr/bin/env python3
"""
GEPA 应用层 v2 · 结果指标 evaluator（Max 纠正版）

v1 的错（Max 一眼看穿）：evaluate 只查「prompt 文本有没有覆盖要素」（过程指标），
漏了「结果能不能被下游用起来」的端到端约束 → 判分 15 分，实际抽取 0 条链。

v2 的修：evaluate 做「端到端结果验证」，score 覆盖三层「协同方能用的约束」：
  ① 结构化：结果必须是 {"chains":[...]} 外层结构（下游 json.loads().get("chains") 能 parse）→ 命门
  ② 落盘存在：结果必须写到真实文档路径（不是飘在内存的 JSON 字符串）→ 结果真实存在
  ③ 下游消费：字段全填 + basis_type 四选一 + action 可执行（正放执行能返回 decision+action）

满分 12 = 结构3 + 落盘3 + 字段2 + 类型2 + 可执行2。
GEPA 优化的是「抽取效果」，不是「prompt 好不好看」。

【运行】
  cd gepa-main && PYTHONPATH=$PWD envs/agent/bin/python3 <本脚本>
"""
import hashlib
import json
import os
import urllib.request

from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    OptimizationState,
    ReflectionConfig,
    optimize_anything,
)

OLLAMA = "http://127.0.0.1:9090/api/chat"
MODEL = "qwen3:14b-32k"

# ── 测试素材：真实日报「追加方案」段（决策密集）──────────────────────────────
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

# ── 落盘目录（结果必须真实存在到这里，下游消费方从这里拿）───────────────────
REGISTRY_DIR = "/Users/max/Documents/08_algo_copilot/daily/week26/gepa_space/runs/decision_chain_prompt_v2/registry"

# ── ① SEED：一句话（会出错，给反思留空间）────────────────────────────────────
SEED = "从日报中提取决策链，输出 JSON。"

REQUIRED_KEYS = ["topic", "state", "decision", "basis", "action"]  # result 可空
BASIS_TYPES = {"铁律", "偏好", "证据", "推断"}
FULL_SCORE = 12.0  # 3结构 + 3落盘 + 2字段 + 2类型 + 2可执行

BACKGROUND = """决策链抽取任务：从日报抽取「人的决策链」，落盘成结构化登记表供下游「正放执行」消费。

每条决策链 = 7 字段：topic(决策事项)/state(现状)/decision(决定)/basis(依据)/basis_type(铁律|偏好|证据|推断)/action(行动)/result(结果，可空)。

下游消费方（正放执行）要求：
- 结果必须是 {"chains": [...]} 外层结构，下游 json.loads(msg).get("chains") 才能解析
- 结果必须落盘到真实文档路径（登记表），下游才能查到
- 每条链 decision/action 必须有实质内容（能被查、能拆成步骤执行）
- basis_type 必须四选一（消费方路由按类型分派）"""


def extract(prompt, content):
    """用 prompt 抽日报，返回 (raw_text, parsed)"""
    full = prompt + "\n\n日报内容:\n---\n" + content + "\n---"
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": full}],
        "stream": False, "format": "json", "options": {"temperature": 0},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=600)
    raw = json.loads(resp.read().decode("utf-8"))["message"]["content"]
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    return raw, parsed


def write_registry_md(chains, path):
    """落盘：把 chains 写成 Markdown 登记表（复现 extract_decision_chains_14b.py 的 render_md）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# 决策链登记表",
        "",
        "| 决策链ID | 决策事项 | 决策链(state→decision→basis→action→result) | 依据类型 |",
        "|---|---|---|---|",
    ]
    for i, c in enumerate(chains, 1):
        chain = " → ".join(str(c.get(k, "")) for k in ["state", "decision", "basis", "action", "result"])
        chain = chain.replace("|", "\\|")
        lines.append(f"| dc-{i:02d} | {c.get('topic','')} | {chain} | {c.get('basis_type','')} |")
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    return path


# ── ② evaluate（裁判）：端到端结果验证，score 覆盖「协同方能用的约束」────────
def evaluate(candidate, opt_state: OptimizationState | None = None):
    side_info: dict = {}
    score = 0.0

    # 上游：抽取
    raw, parsed = extract(candidate, TEST_CONTENT)

    # ① 结构化（命门：下游能 parse）
    if not (isinstance(parsed, dict) and "chains" in parsed and isinstance(parsed["chains"], list)):
        side_info["error"] = "结果不是 {'chains':[...]} 外层结构，下游 get('chains') 解析不到"
        side_info["raw_head"] = raw[:200]
        side_info["scores"] = {"prompt_quality": 0.01}  # GEPA objective 前沿依赖 scores 字段
        print(f"[evaluate] score=0.01/{FULL_SCORE:.0f} 结构错(下游解析不到)", flush=True)
        # 注意：不能返回 0.0 —— GEPA 帕累托前沿空集会 assert 崩（select_program_candidate_from_pareto_front）。
        # 用极小正分 0.01 让 seed 进前沿，reflection 才有东西可选可改。
        return 0.01, side_info
    chains = parsed["chains"]
    score += 3.0

    if not chains:
        side_info["error"] = "结构对但 0 条链"
        print(f"[evaluate] score={score}/{FULL_SCORE:.0f} 0条链", flush=True)
        return score, side_info

    # ② 落盘存在（结果必须落到真实路径，下游才能查到）
    key = hashlib.md5(candidate.encode("utf-8")).hexdigest()[:8]
    registry_path = os.path.join(REGISTRY_DIR, f"registry_{key}.md")
    exists = False
    try:
        write_registry_md(chains, registry_path)
        exists = os.path.exists(registry_path) and os.path.getsize(registry_path) > 0
    except Exception as e:
        side_info["error"] = f"落盘失败: {e}"
    if exists:
        score += 3.0

    # ③ 下游消费约束（字段全填 + 类型四选一 + action 可执行）
    field_ok = bt_ok = executable = 0
    for c in chains:
        if not isinstance(c, dict):
            continue
        if all(str(c.get(k, "")).strip() for k in REQUIRED_KEYS):
            field_ok += 1
        if c.get("basis_type", "") in BASIS_TYPES:
            bt_ok += 1
        if len(str(c.get("action", "")).strip()) >= 4:
            executable += 1
    n = len(chains)
    score += 2.0 * (field_ok / n) + 2.0 * (bt_ok / n) + 2.0 * (executable / n)

    side_info.update({
        "scores": {"prompt_quality": score},  # GEPA objective 前沿依赖这个字段（v2 漏了会崩）
        "chains_count": n,
        "registry_path": registry_path,
        "diagnosis": f"结构✅ 落盘{'✅' if exists else '❌'} 字段{field_ok}/{n} 类型{bt_ok}/{n} 可执行{executable}/{n}",
    })
    if not exists:
        side_info["error"] = side_info.get("error", "结果没落盘到路径，下游查不到")

    print(f"[evaluate] score={score:.1f}/{FULL_SCORE:.0f} 链{n}条 "
          f"结构✅ 落盘{'✅' if exists else '❌'} 字段{field_ok}/{n} 类型{bt_ok}/{n} 可执行{executable}/{n}", flush=True)
    return score, side_info


# ── sanity check：结果指标 evaluator 能不能区分好坏 prompt ──────────────────
def sanity_check():
    """用一句话(烂)和原始 PROMPT(好)各跑一次 evaluate，看结果指标能否区分。"""
    original = """你是决策链抽取器。从日报抽取「人的决策链」,每条链 = 一次完整决策。
每条决策链字段(严格 JSON):
- topic: 决策事项(解决什么问题,一句话)
- state: 状态(决策前的客观现状)
- decision: 决策(做了什么决定)
- basis: 依据(为什么这么决策,一句话)
- basis_type: 依据类型,四选一:铁律(写死规则)/偏好(反复纠正的倾向)/证据(可溯源文件/数据)/推断(未验证)
- action: 行动(决策后做了什么)
- result: 结果(行动的结果,日报里没有就留空 "")
只输出 JSON,不要其他文字:
{"chains": [{"topic":"...","state":"...","decision":"...","basis":"...","basis_type":"偏好","action":"...","result":"..."}]}"""

    print("=== sanity check（结果指标 evaluator 区分度）===")
    s1, _ = evaluate(SEED, None)
    s2, _ = evaluate(original, None)
    print(f"  烂 seed  得分 {s1:.1f}/{FULL_SCORE:.0f}")
    print(f"  原始 PROMPT 得分 {s2:.1f}/{FULL_SCORE:.0f}")
    print("  ✅ 结果指标能区分好坏" if s2 > s1 else "  ⚠️ 没区分度，检查规则")


def main():
    sanity_check()
    optimize_anything(
        seed_candidate=SEED,
        evaluator=evaluate,
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir="/Users/max/Documents/08_algo_copilot/daily/week26/gepa_space/runs/decision_chain_prompt_v2",
                max_metric_calls=12,  # 每次 evaluate 真抽 14B（慢），控制规模
                track_best_outputs=True,
                cache_evaluation=True,
                frontier_type="objective",
            ),
            reflection=ReflectionConfig(
                reflection_lm="ollama/qwen3:14b-32k",
                reflection_lm_kwargs={"api_base": "http://127.0.0.1:9090"},
            ),
            refiner=None,
        ),
        objective="优化决策链抽取 prompt，让它抽取的结果能被下游「正放执行」消费：输出 {\"chains\":[...]} 外层结构、落盘到真实路径、字段全填、basis_type 四选一、action 可执行。",
        background=BACKGROUND,
    )


if __name__ == "__main__":
    main()
