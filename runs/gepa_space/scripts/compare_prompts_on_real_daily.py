#!/usr/bin/env python3
"""
用 4 个 prompt 实际抽取同一段真实日报，对比「抽取结果」——验证优化有没有真变好，不是看判分分数。

prompt 来源：
  - seed       : 一句话（GEPA 起点，2 分）
  - 原始 PROMPT : extract_decision_chains_14b.py 里 Max 在用的（基线）
  - GEPA best  : 优化后 Iter 1（15 分，GEPA 认定的最优）
  - Iter 6     : 优化后 Iter 6（15 分，我判断真最优，有 basis_type 业务含义 + 空数组边界）

测试素材：从 20260409.md「追加方案」段摘的决策密集片段（真实日报原文）。
"""
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OLLAMA = "http://127.0.0.1:9090/api/chat"
MODEL = "qwen3:14b-32k"

# ── 测试素材：真实日报的「追加方案」段（决策密集）────────────────────────────
TEST_CONTENT = """今天的任务编排：
- 中断槽：任务4（不确定性运维支持）单独保留，随时抢占，不和深度任务混排。
- 主业务串行链：2 -> 1/8 并行 -> 3。先把任务2（销量预测大模型2期）做成统一结论和数据口径，再并行产出任务1（badcase报告）、任务8（sku评估），最后做任务3（产品化排期）。
- 自动化资产链：6 -> 7 -> 5，9 并行挂边。先把任务6做成可复制工作流，再把任务7沉淀成模板/RAG资产，再用任务5做实验验证；任务9作为外部benchmark不阻塞主线。
- 结论：今天真正的P0是 4+2+6，其它都围绕这3个任务做并行支持。

人与AI分工：
- 你只做3件事：定优先级、做关键判断、做最终验收。
- codex cli 负责高歧义任务：拆解、追问、改写、review、定下一步。
- codex exec 负责低歧义任务：批量解析、生成模板、跑build/test/demo、产出日报草稿。
- 标准是 1人主控 + 多exec工人 + 最后review。

下一步自动化只做最小闭环：
- 把日报每天增量解析成结构化任务，形成统一任务池。
- 按标签自动匹配模板，生成prompt pack。
- 固定 build/test/smoke/demo/summary 5个命令，变成日报资产流水线。"""

# ── 4 个 prompt ─────────────────────────────────────────────────────────────
SEED = "从日报中提取决策链，输出 JSON。"

ORIGINAL = """你是决策链抽取器。从日报抽取「人的决策链」,每条链 = 一次完整决策。

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

ITER1_BEST = """你是一个决策链抽取器。请从日报中提取决策链，输出严格符合以下格式的JSON，不包含任何额外内容。决策链包含以下七个字段：topic（决策事项，一句话）、state（决策前的客观现状）、decision（决策内容）、basis（决策依据，一句话）、basis_type（依据类型，必须是四选一：铁律/偏好/证据/推断）、action（决策后的行动）、result（结果，若日报中无则留空）。请按以下步骤操作：1. 确定topic；2. 提取state；3. 识别decision；4. 分析basis及对应的basis_type（必须从铁律、偏好、证据、推断中选择）；5. 记录action；6. 查找result（若日报未提及则留空）。若某字段信息缺失，请留空。"""

ITER6_BEST = """你是一个决策链抽取器。你的任务是从日报中严格提取结构化决策链，仅输出JSON格式，不包含任何解释、说明或额外文字。决策链必须包含以下七个字段：
- topic（决策事项，一句话）
- state（决策前的客观现状）
- decision（决策内容）
- basis（决策依据，一句话）
- basis_type（依据类型，必须从以下四选一：铁律/偏好/证据/推断）
- action（决策后的行动）
- result（结果，若日报中无则留空，用空字符串""表示）

请按以下步骤操作：
1. 确定topic：明确决策解决的核心问题，用一句话概括。
2. 提取state：描述决策前的客观现状，避免主观判断。
3. 识别decision：提取明确的决策动作或选择。
4. 分析basis及basis_type：
   - 若依据是明确规则（如公司政策），选铁律；
   - 若依据是反复纠正的倾向（如个人偏好），选偏好；
   - 若依据可溯源文件数据（如报表、合同），选证据；
   - 若依据是未验证的推断（如假设），选推断。
5. 记录action：描述决策后的具体行动。
6. 查找result：若日报未提及结果，必须留空字符串""，不可缺失字段。

强制约束：
- 输出必须为纯JSON数组，格式为 [{"topic": "...", "state": "...", ...}]，禁止任何额外文本。
- basis_type必须严格四选一，若无法确定，根据上下文推断最接近的类型。
- 空值规则：字段信息缺失时，用空字符串""填充，不可省略字段。
- 若无决策链，输出空数组[]，不可输出其他内容。"""

PROMPTS = {
    "1_seed一句话": SEED,
    "2_原始PROMPT(基线)": ORIGINAL,
    "3_GEPA-best(Iter1)": ITER1_BEST,
    "4_Iter6(真最优)": ITER6_BEST,
}


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


def summarize(name, raw, parsed):
    """客观质量摘要（0-LLM 统计，不是主观判分）"""
    print(f"\n{'='*70}")
    print(f"【{name}】")
    print(f"{'='*70}")
    if parsed is None:
        print(f"  ❌ JSON 解析失败，原始输出前 300 字:\n{raw[:300]}")
        return
    chains = parsed.get("chains", parsed if isinstance(parsed, list) else [])
    print(f"  抽到 {len(chains)} 条链")
    legal_bt = {"铁律", "偏好", "证据", "推断"}
    bt_ok = 0
    field_ok = 0
    for i, c in enumerate(chains, 1):
        if not isinstance(c, dict):
            continue
        bt = c.get("basis_type", "")
        if bt in legal_bt:
            bt_ok += 1
        filled = sum(1 for k in ["topic", "state", "decision", "basis", "action"] if c.get(k, "").strip())
        field_ok += 1 if filled == 5 else 0
        print(f"\n  --- 链{i} ---")
        print(f"    topic  : {c.get('topic', '(空)')}")
        print(f"    state  : {c.get('state', '(空)')}")
        print(f"    decision: {c.get('decision', '(空)')}")
        print(f"    basis  : {c.get('basis', '(空)')}")
        print(f"    basis_type: {bt}")
        print(f"    action : {c.get('action', '(空)')}")
        print(f"    result : {c.get('result', '(空)')}")
    print(f"\n  质量摘要: basis_type 合法 {bt_ok}/{len(chains)}，五字段全填 {field_ok}/{len(chains)}")


def main():
    print(f"测试素材长度: {len(TEST_CONTENT)} 字\n")
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(extract, p, TEST_CONTENT): n for n, p in PROMPTS.items()}
        for fut in futs:
            name = futs[fut]
            raw, parsed = fut.result()
            summarize(name, raw, parsed)


if __name__ == "__main__":
    main()
