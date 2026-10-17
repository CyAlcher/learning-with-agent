#!/usr/bin/env python3
"""
GEPA 应用层 · 接地气版 —— 优化「决策链抽取 prompt」（来自 Max 的真实历史任务，不是 circle_packing 玩具题）。

【为什么这个任务接地气】
Max 的历史任务里有一条真实的 14B 抽取管线：
  rsi_project/scripts/extract_decision_chains_14b.py
它用一段 PROMPT 从日报抽「决策链 state→decision→basis→action→result + basis_type(四选一)」。
本脚本把「那段 PROMPT」当作要被优化的文本，把「判分标准」也从那条管线提炼（七字段 + basis_type 四选一 + JSON 约束）。
→ 用 GEPA 优化自己真实任务里的 prompt，而不是跑一个 26 圆打包的通用示例。

【应用层 3 件事】
  ① SEED       = 一句话退化版 prompt（故意会出错，给反思留改进空间，避免「seed 已满分→反思学不到东西」）
  ② evaluate   = 0-LLM 判分（检查 prompt 覆盖了决策链抽取的哪些结构化要素），返回 (score, side_info)
  ③ optimize_anything 跑 5 步循环（选→跑→反思→变异→收）

【验证目标】
  14B 能不能一步步把「一句话」改造成「结构化抽取 prompt」——每一步补上缺的要素，分数逐步提升。
  观察点：evaluate 打印每轮的 score + 缺哪些要素，就是 14B「一步步理解」的轨迹。

【运行】
  cd /Users/max/Documents/08_algo_copilot/daily/week23/rsi_project/vendor/02_self_evolving_agent/gepa-main
  PYTHONPATH=$PWD /Users/max/Documents/mini_conda/envs/agent/bin/python3 \
    /Users/max/Documents/08_algo_copilot/daily/week26/gepa_space/scripts/optimize_decision_chain_prompt.py
"""
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    OptimizationState,
    ReflectionConfig,
    optimize_anything,
)

# ── ① SEED：真实 PROMPT 的「一句话退化版」────────────────────────────────────
# 真源：extract_decision_chains_14b.py 的 PROMPT（7 字段 + basis_type 四选一 + JSON）。
# 这里故意退化成一句话 → 判分会很低 → 14B 反思才有「改进空间」。
SEED = "从日报中提取决策链，输出 JSON。"

# ── 真实业务背景（objective/background 给反思 LLM 的方向上下文）────────────────
BACKGROUND = """决策链抽取任务：从 Max 的日报（YYYYMMDD.md）里抽取「人的决策链」。

每条决策链 = 一次完整决策，含 7 个字段：
- topic: 决策事项（解决什么问题，一句话）
- state: 状态（决策前的客观现状）
- decision: 决策（做了什么决定）
- basis: 依据（为什么这么决策，一句话）
- basis_type: 依据类型，四选一：铁律(写死规则)/偏好(反复纠正的倾向)/证据(可溯源文件数据)/推断(未验证)
- action: 行动（决策后做了什么）
- result: 结果（行动的结果，日报里没有就留空）

消费侧：抽出的决策链供「正放执行」——given(state现状|该怎么做) 查库返回 decision+action 拆解执行。
好的 prompt 应该：定义角色、列全七字段、约束 basis_type 四选一、要求只输出 JSON、给 step-by-step 分步、说明空值规则。"""

# ── 判分标准（0-LLM，确定性可复跑；每个要素都标注来自真实 PROMPT 哪一行）────────
REQUIRED_FIELDS = ["topic", "state", "decision", "basis", "basis_type", "action", "result"]
BASIS_TYPES = ["铁律", "偏好", "证据", "推断"]
# 满分 = 1(角色) + 7(字段) + 2(basis_type四选一) + 2(JSON) + 1(只输出) + 2(分步) + 1(空值) = 16
FULL_SCORE = 16.0


def score_decision_chain_prompt(prompt: str):
    """0-LLM 判分：检查 prompt 覆盖了决策链抽取的哪些结构化要素。

    返回 (score, missing, detail)。规则全部确定性（关键词/字段存在性），不靠 LLM，
    所以可复跑、可评测——符合「评测器判通过」铁律。
    """
    detail: dict[str, bool] = {}
    missing: list[str] = []
    score = 0.0

    # 1. 角色定义（真实 PROMPT: "你是决策链抽取器"）
    if ("你是" in prompt or "角色" in prompt) and "抽取" in prompt:
        score += 1.0
        detail["角色定义"] = True
    else:
        missing.append("角色定义(你是决策链抽取器)")
        detail["角色定义"] = False

    # 2. 七字段定义（真实 PROMPT 逐字段列出）
    for f in REQUIRED_FIELDS:
        if f in prompt:
            score += 1.0
            detail[f"字段_{f}"] = True
        else:
            missing.append(f"字段 {f}")
            detail[f"字段_{f}"] = False

    # 3. basis_type 四选一（真实 PROMPT: "铁律/偏好/证据/推断"）
    hit = [t for t in BASIS_TYPES if t in prompt]
    if len(hit) >= 4:
        score += 2.0
        detail["basis_type四选一"] = True
    else:
        miss_types = "、".join(set(BASIS_TYPES) - set(hit))
        missing.append(f"basis_type 四选一(缺 {miss_types})")
        detail["basis_type四选一"] = False

    # 4. JSON 输出约束（真实 PROMPT: 只输出 JSON）
    if "json" in prompt.lower():
        score += 2.0
        detail["JSON约束"] = True
    else:
        missing.append("JSON 输出约束")
        detail["JSON约束"] = False

    # 5. 只输出 JSON（真实 PROMPT: "只输出 JSON,不要其他文字"）
    if "只输出" in prompt or "不要其他" in prompt:
        score += 1.0
        detail["只输出JSON"] = True
    else:
        missing.append("只输出JSON不输出其他文字")
        detail["只输出JSON"] = False

    # 6. step-by-step 分步指令（Max 铁律：Leader→flash 必须 step-by-step）
    if any(k in prompt for k in ["第1步", "第一步", "分步", "步骤", "先", "再"]):
        score += 2.0
        detail["分步指令"] = True
    else:
        missing.append("分步指令(step-by-step)")
        detail["分步指令"] = False

    # 7. 空值规则（真实 PROMPT: "result ... 日报里没有就留空"）
    if "留空" in prompt or "为空" in prompt:
        score += 1.0
        detail["空值规则"] = True
    else:
        missing.append("空值规则(result 日报没有就留空)")
        detail["空值规则"] = False

    return float(score), missing, detail


# ── ② evaluate（裁判）：返回 (score, side_info) ──────────────────────────────
def evaluate(candidate, opt_state: OptimizationState | None = None):
    score, missing, detail = score_decision_chain_prompt(candidate)

    if missing:
        diagnosis = "这个 prompt 缺少以下要素: " + "; ".join(missing)
    else:
        diagnosis = "已覆盖全部结构化要素(满分)"

    # side_info = 反思 LLM 读的「诊断反馈」。官方口径：error 字段 = actionable feedback。
    side_info = {
        "scores": {"prompt_quality": score},
        "missing": missing,
        "detail": detail,
        "diagnosis": diagnosis,
        "code": candidate,
        "stdout": diagnosis,
        "error": None if not missing else diagnosis,
    }

    # 打印进化轨迹：每一轮 14B 改出来的 prompt 得几分、还缺啥
    print(f"[evaluate] score={score:.0f}/{FULL_SCORE:.0f} 缺{len(missing)}项: "
          f"{'; '.join(missing) if missing else '(满分)'}", flush=True)
    return score, side_info


# ── sanity check：证明判分函数能区分「烂 seed」和「好 prompt」──────────────────
def sanity_check():
    """跑 GEPA 前先自证：判分函数给烂 seed 低分、给真实 PROMPT 高分。"""
    real_prompt = """你是决策链抽取器。从日报抽取「人的决策链」,每条链 = 一次完整决策。
每条决策链字段(严格 JSON):
- topic: 决策事项(解决什么问题,一句话)
- state: 状态(决策前的客观现状)
- decision: 决策(做了什么决定)
- basis: 依据(为什么这么决策,一句话)
- basis_type: 依据类型,四选一:铁律(写死规则)/偏好(反复纠正的倾向)/证据(可溯源文件/数据)/推断(未验证)
- action: 行动(决策后做了什么)
- result: 结果(行动的结果,日报里没有就留空 "")
只输出 JSON,不要其他文字:{"chains": [{"topic":"...","state":"...","decision":"...","basis":"...","basis_type":"偏好","action":"...","result":"..."}]}"""

    seed_s, seed_m, _ = score_decision_chain_prompt(SEED)
    real_s, real_m, _ = score_decision_chain_prompt(real_prompt)
    print(f"[sanity] 烂 seed  得分 {seed_s:.0f}/{FULL_SCORE:.0f}，缺 {len(seed_m)} 项")
    print(f"[sanity] 真实 PROMPT 得分 {real_s:.0f}/{FULL_SCORE:.0f}，缺 {len(real_m)} 项")
    print("[sanity] 判分函数有效(能区分好坏)✅" if real_s > seed_s else "[sanity] ⚠️ 判分函数没区分度，检查规则")


def main():
    sanity_check()

    optimize_anything(
        seed_candidate=SEED,
        evaluator=evaluate,
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir="/Users/max/Documents/08_algo_copilot/daily/week26/gepa_space/runs/decision_chain_prompt",
                max_metric_calls=16,  # 约 8-10 轮 proposal，够看完整进化轨迹；调大看更多轮
                track_best_outputs=True,
                cache_evaluation=True,
                frontier_type="objective",
            ),
            reflection=ReflectionConfig(
                reflection_lm="ollama/qwen3:14b-32k",
                reflection_lm_kwargs={"api_base": "http://127.0.0.1:9090"},
            ),
            refiner=None,  # 禁用 refiner（api_base 传递问题，见 gepa-optimization skill 坑6）
        ),
        objective="优化决策链抽取 prompt，让它覆盖：七字段(topic/state/decision/basis/basis_type/action/result)、basis_type 四选一(铁律/偏好/证据/推断)、只输出 JSON、分步指令、空值规则，从而能稳定从日报抽取结构化决策链。",
        background=BACKGROUND,
    )


if __name__ == "__main__":
    main()
