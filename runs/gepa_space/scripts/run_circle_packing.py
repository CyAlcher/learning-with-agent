#!/usr/bin/env python3
"""
GEPA 应用层代码（circle_packing 26 圆打包）——「怎么用 GEPA」的正确入口，不是框架源码。

【进度】本次 session 完成到「第一轮最简单优化实现」；应用 pipeline 的写法【未写好】，后续待完善。

【应用层 = 3 件事】用 GEPA 解决任何问题，只写这 3 件，其余交给框架黑盒：
  ① 定义「要优化什么」   → SEED_CODE(初始代码) + objective(目标) + background(背景)
  ② 定义「怎么评估好坏」 → evaluate 函数：返回 (score, side_info) 两样
  ③ 调 optimize_anything  → 把 ①② 丢给框架，框架自动跑 5 步循环(选→跑→反思→变异→收)

【杠杆点（花小力气撬大效果，只有 2 个）】
  1. evaluate 函数 = 最大杠杆（裁判）：
       - score 算错 → GEPA 往错的方向优化（裁判判错全废）
       - side_info 给得差 → 反思 LLM 诊断不了 → 变异瞎改
       - 「用 GEPA 的功夫 80% 花在写对 evaluate」
  2. SEED_CODE 起点质量 = 第二杠杆：
       - seed 好 → 起点高省预算；seed 烂 → 花更多 max_metric_calls 救回来

【谁和谁协同】
  SEED_CODE(起点) ──被评估──> evaluate(裁判) ──score+side_info──> 反思 LLM(诊断)
       ↑                                                          │
       └──────── objective + background(给方向的上下文) <──────────┘
  seed 烂没关系(GEPA 能救)，evaluate 歪全废(裁判歪)，objective 模糊反思瞎猜(方向歪)

【纠正】evaluator 不是「返回分数」，是返回两样（见 evaluate 函数第 83 行）：
  return score, side_info   # score=半径和(越大越好)，side_info=诊断信息(为什么坏)
"""
from examples.circle_packing.utils import (
    CIRCLE_PACKING_BACKGROUND,
    compute_multiple_metrics,
    execute_code,
    extract_best_circles,
)
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    OptimizationState,
    RefinerConfig,
    ReflectionConfig,
    optimize_anything,
)

# Seed code from ShinkaEvolve
SEED_CODE = '''
 import numpy as np

def main(timeout, current_best_solution):
    """Circle packing: returns dict with 'circles' (n,3) and 'all_scores'."""
    n = 26

    if current_best_solution is not None:
        circles = current_best_solution.copy()
    else:
        centers = np.zeros((n, 2))
        centers[0] = [0.5, 0.5]

        # Ring of 8 around center
        angles = 2 * np.pi * np.arange(8) / 8
        centers[1:9] = np.column_stack([0.5 + 0.3 * np.cos(angles), 0.5 + 0.3 * np.sin(angles)])

        # Outer ring for remaining 17
        angles = 2 * np.pi * np.arange(17) / 17
        centers[9:] = np.column_stack([0.5 + 0.7 * np.cos(angles), 0.5 + 0.7 * np.sin(angles)])

        centers = np.clip(centers, 0.01, 0.99)
        radii = compute_max_radii(centers)
        circles = np.column_stack([centers, radii])

    return {'circles': circles, 'all_scores': [float(circles[:, 2].sum())]}


def compute_max_radii(centers):
    """Compute maximum radii that don't overlap and stay in unit square."""
    n = len(centers)
    radii = np.minimum.reduce([centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]])

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii
'''



def evaluate(candidate, opt_state: OptimizationState | None = None):
    warm_start = extract_best_circles(opt_state)
    result = execute_code(candidate, 600, warm_start)

    circles = result["circles"]
    score = result["validation_details"]["sum_radii"]
    metrics = compute_multiple_metrics(result["all_scores"])

    side_info = {
        "scores": {"sum_radii": score},
        "metrics": metrics,
        "code": candidate,
        "circles": circles,
        "stdout": result.get("stdout", ""),
        "error": result.get("error"),
        "traceback": result.get("traceback"),
        "validation_details": result.get("validation_details"),
    }

    return score, side_info


def main():
    optimize_anything(
        seed_candidate=SEED_CODE,
        evaluator=evaluate,
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir="/Users/max/Documents/08_algo_copilot/daily/week26/gepa_space/runs/circle_packing",
                max_metric_calls=30,
                track_best_outputs=True,
                cache_evaluation=True,
                frontier_type="objective",
            ),
            reflection=ReflectionConfig(
                reflection_lm="ollama/qwen3:14b-32k",
                reflection_lm_kwargs={"api_base": "http://127.0.0.1:9090"},
            ),
            refiner=None,  # 禁用 refiner（避免 refiner 的 api_base 传递问题）
        ),
        objective="Optimize circle packing code to maximize sum of circle radii within a unit square for N=26 circles.",
        background=CIRCLE_PACKING_BACKGROUND,
    )


if __name__ == "__main__":
    main()
