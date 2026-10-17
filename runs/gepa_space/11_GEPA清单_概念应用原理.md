# GEPA 学习清单（概念 / 应用 / 原理 三协同）

> 主线 = **应用**（先用好）。原理 = 应用卡住时才深挖。概念 = 复述锚点（防「感觉会了其实不会」）。
> 打勾规则（jwasham）：**动手验证过才算 ✅**，读完/看过不算。状态：✅ 掌握（能跑通） / 🔶 待验证（学了没验） / ⬜ 未做。
> 推进顺序：永远先做「应用清单」里 ROI 最高、还没打勾的那一项。

---

## 一、应用清单（主线，先用好）——穷举 GEPA 的全部「用法」

GEPA 有 **5 个正交维度**的用法，当前只用了其中 1 个交叉点。

### 维度 A：三个 API 入口（从哪个函数进来）

| 用法 | 干什么 | 状态 | 依据 |
|---|---|---|---|
| `gepa.optimize()` | 核心引擎 API：优化**多组件系统**（`dict{组件:文本}`），写 adapter 接入 | ✅ 已跑（决策链抽取 dataset 形态：v1 0.01→0.2 跑偏，v2 补契约 0.01→0.9） | `src/gepa/api.py:46` |
| `dspy.GEPA` | 在 DSPy 里优化（README 原文「the most powerful way」「Recommended for AI Pipelines」） | ⬜ 未做 | README Quick Start 第 2 例 |
| `gepa.optimize_anything.optimize_anything()` | 便捷封装：优化**单 artifact**（`str`），写 evaluator 函数，可切 4 引擎 | ✅ 唯一在用 | `src/gepa/optimize_anything.py` |

### 维度 A 核心：两个 API 的本质差异 + 场景选择链路（2026-10-17 会话产出）

> 决策链路：**人看到问题场景 → 选 API → 设计 evaluator/adapter → 回到场景外部假设验证**。
> 两个 API 底层是**同一个 5 步循环**（`core.engine.GEPAEngine`），差异只在「优化对象抽象」+「评估→变异的契约通道」。

**差异对照（相同外部条件下的本质差异）：**

| 维度 | `optimize_anything`（便捷） | `gepa.optimize`（核心） |
|---|---|---|
| 优化对象 | **单 artifact**（`str`） | **多组件系统**（`dict{组件:文本}`） |
| 接入方式 | 写 **evaluator 函数** `(candidate, example)->(score, info)` | 写 **adapter**（`evaluate` + `make_reflective_dataset`） |
| 「为什么错、错在哪」放哪 | evaluator 的 **info（side_info）** + **objective/background**（→ `gepa_launcher.py:1635` `_build_reflection_prompt_template`） | adapter 的 **Feedback**（`make_reflective_dataset` 返回，`reflective_mutation.py:473`） |
| 变异读什么 | side_info + 反思模板里的 objective/background | reflective_dataset 的 Feedback（**唯一通道，无 objective 兜底**） |
| 引擎 | 可切 4 个（gepa/meta_harness/autoresearch/best_of_n） | 固定 GEPAEngine |

**场景 → API 选择（协作层链路）：**

| 看到什么场景 | 选哪个 API | 为什么 |
|---|---|---|
| 优化**一段** prompt/代码/配置（单 artifact） | `optimize_anything` | seed=str + evaluator 函数，最简 |
| 优化**多组件**（agent 多个 prompt 一起进化） | `gepa.optimize`（dict） | optimize_anything 的 str 做不到 |
| 要**切引擎**（Claude Code 当 proposer 等） | `optimize_anything`（`engine=`） | optimize 固定 GEPAEngine |
| 要**控制反思看到的轨迹**（按组件各自反馈） | `gepa.optimize`（adapter） | `make_reflective_dataset` 你说了算 |
| DSPy 生态内 | `dspy.GEPA` | 集成 |

**踩过的坑（联动「评估→变异」环节，v1 跑偏 → v2 修正）：** 走 optimize 通道时，契约唯一入口是 evaluate 的 Feedback（`gepa.optimize` 无 objective/background 兜底，`api.py:46`）。v1 Feedback 只写「字段缺失」不写「缺哪几个字段+语义」→ 变异环节拿不到契约 → 反思脑补出「任务链」，0.01→0.2 卡住。**v2 三处补契约后 0.01→0.9**：① Feedback 承载完整契约 ② 判分严格版（错误字段扣分）③ `reflection_prompt_template` 锁死字段。**铁律：选哪个 API，就把它那条「错在哪」通道填满契约（optimize_anything 走 info+objective/background；optimize 走 Feedback+reflection_prompt_template）。**

### 维度 B：两个变异算子（谁来生成新候选）

| 用法 | 干什么 | 状态 | 依据 |
|---|---|---|---|
| `reflection_lm` | 默认：LLM 文本进出，反思+变异一次调用完成 | ✅ 已用（本地 qwen14b） | `ReflectionConfig.reflection_lm` |
| `custom_candidate_proposer` | 挂**任意外部 agent** 当变异算子（结构化访问 reflective_dataset） | ⬜ 未做 | 契约 `(candidate, reflective_dataset, components_to_update)->dict[str,str]`；现成实现 `ClaudeCodeAgentProposer`（`oa/proposers/claude_code_agent.py`） |

### 维度 C：9 种接业务的方式（裸 evaluate 函数 + 8 个 adapter）

| 用法 | 干什么 | 状态 |
|---|---|---|
| 裸 `evaluate` 函数 | 自己写 `evaluate(candidate)->(score, side_info)` | ✅ 已用（决策链抽取） |
| `DefaultAdapter` | 单轮 LLM prompt 优化 | ⬜ |
| `ConfidenceAdapter` | logprob 置信度打分，惩罚瞎猜（llm-structured-confidence） | ⬜ |
| `DSPy Full Program` | 进化整个 DSPy 程序（signatures/modules/control flow），MATH 67%→93% | ⬜ |
| `Generic RAG` | RAG pipeline（chroma/lancedb/milvus/qdrant/weaviate） | ⬜ |
| `MCP` | 优化 MCP tool 描述 + system prompt | ⬜ |
| `TerminalBench` | 终端 agent（Terminus） | ⬜ |
| `AnyMaths` | 数学推理 | ⬜ |
| `LangChain` | LangChain pipeline（chat/tool/LangGraph） | ✅ 已跑通 v2：全链路端到端（含下游消费）0.01→12.0/12 满分，5 条链路由+执行全通 |

### 维度 D：4 个引擎（`OptimizeAnythingConfig.engine`）

| 用法 | 机制 | 状态 |
|---|---|---|
| `gepa`（默认） | 5 步循环 + 帕累托 | ✅ |
| `meta_harness` | Claude Code subprocess 当 proposer，读 frontier+history 写 pending_eval.json | ⬜ |
| `autoresearch` | Claude Code 黑盒：program.md/candidate.txt + eval.sh POST 打分服务 | ⬜ |
| `best_of_n` | 朴素 baseline，独立采样 N 次（评测 floor） | ⬜ |

### 维度 E：核心机制（5 步循环 + 2 增强）

| 机制 | 一句话 | 状态 |
|---|---|---|
| Select | 从帕累托前沿挑候选 | ✅ |
| Execute | minibatch 评估，捕获完整轨迹 | ✅ |
| Reflect | LLM 读轨迹诊断「为什么坏」 | ✅ |
| Mutate | LLM 生成改进（带诊断结论） | ✅ |
| Accept | 变好进池，更新前沿 | ✅ |
| **merge** | 合并两个帕累托最优候选的优点（杂交） | ⬜ 未做 |
| **ASI** | 评估器返回的 error/traceback/stdout = 文本版梯度 | ✅ 已用（side_info） |

**结论（答「应用方式穷举完了吗」）：5 个维度共 3+2+9+4+7 = 25 个用法，当前只打勾 7 个（基本集中在 `optimize_anything` + `reflection_lm` + 裸 evaluate 这一个交叉点），其余 18 个全未碰。穷举到此完毕，后续只在这 25 项里打勾，不再新增维度。**

---

## 二、概念清单（复述锚点——每项能一句话复述才算过）

| 概念 | 一句话 | 状态 |
|---|---|---|
| GEPA 是什么 | 让 LLM 当「进化算法的变异算子」，用「为什么失败」改进文本 | ✅ |
| 5 步循环 | 选→跑→反思→变异→收 | ✅ |
| 反思+变异合一 | 同一次 LLM 调用：边读「哪里错」边吐「改成啥」 | ✅ |
| 帕累托前沿 | 不是留一个冠军，是每个子任务上的冠军都留着 | 🔶 概念懂，没单独验证前沿/merge 行为 |
| ASI | 诊断信息（error/traceback/stdout）= 文本版梯度 | ✅ |
| 不是 MDP/RL | 是进化搜索，不学 policy，不需要预标注 reward | ✅ |
| seed 退化版 gotcha | seed 已满分 → 反思学不到东西（saturated signal） | ✅ 踩过 |
| 结果指标 > 过程指标 | 判分测什么就优化什么，漏了下游契约就被优化掉 | ✅ 血泪 |
| merge（杂交） | 把 A、B 两个候选的优点合并成新候选 | ⬜ 未学 |

---

## 三、原理清单（应用卡住时才深挖——源码在哪 + 触发条件）

> 原则：**不是现在看**。应用没用好（跑不通 / 效果差）时，按「触发条件」翻对应源码，其余一律不碰。

| 原理 | 源码位置 | 触发条件（什么时候看） |
|---|---|---|
| 反思+变异合一怎么实现 | `src/gepa/proposer/reflective_mutation/reflective_mutation.py` | 变异出的候选质量差、跑偏时 |
| 怎么判「收不收」（acceptance） | `src/gepa/strategies/acceptance.py` | 候选没进池、前沿不更新时 |
| 帕累托前沿数据结构 + merge | `src/gepa/proposer/merge.py` + 前沿相关 | 需要「多候选各有所长」融合时 |
| 预算/停止条件 | `src/gepa/utils/stop_condition.py` + `oa/budget.py` | BudgetExhausted 提前停、跑不完时 |
| evaluator 返回契约（score/side_info/scores） | `src/gepa/optimize_anything.py` + skill 坑 #8/#10 | assert 崩、前沿空集时 |
| Claude Code 挂变异器 | `src/gepa/oa/proposers/claude_code_agent.py` | 做维度 B 第 2 项时看 |

---

## 四、声明核心能力 vs 实际使用（缺口核对）

> 答「GEPA 声明中的核心能力使用了吗」：**没有，只用了约 2/9**。

| README 声明的能力 | 用了没 | 说明 |
|---|---|---|
| `optimize_anything`（任意 artifact） | ✅ | 但只走了 reflection_lm + 裸 evaluate 这一条路 |
| ASI（诊断反馈） | ✅ | side_info |
| 5 步循环 | ✅ | 跑通闭环 |
| `gepa.optimize()` 核心 API | ✅ | 已跑通（决策链抽取 dataset 形态，见维度 A 核心差异） |
| `dspy.GEPA`（README 推荐最强） | ❌ | 且 skill 记 dspy 3.1.3 版本冲突未解决 |
| 8 个 adapters | ❌ | 全没用（用裸 evaluate 函数） |
| `custom_candidate_proposer`（挂 Claude Code） | ❌ | 没碰 |
| merge（系统感知合并） | ❌ | 没碰 |
| Agent Skill（Claude Code 驱动 optimize_anything） | ❌ | 没碰 |

---

## 五、下一步最小行动（按 ROI 倒排，主线 = 先用好）

| 排序 | 行动 | ROI 理由 |
|---|---|---|
| 1 | 维度 A 核心收尾：给 evaluate 的 Feedback 承载完整七字段契约，重跑 optimize 验证「跑偏修正」 | 巩固「评估→变异契约通道」理解，端到端闭环 |
| 2 | 维度 B-2：`custom_candidate_proposer` 挂 Claude Code（复用中转） | 复用资产 + DSH 同构 |
| 3 | 维度 A-2：`dspy.GEPA` | README 推荐最强，需先解 dspy 版本冲突 |

> 待 Max 打勾/定优先级。默认按上表从 1 开始。

---

## 六、LangChainAdapter 落地路径（2026-10-17 源码调研）

**结论：不用 DSPy 就能优化 LangGraph 应用。** `LangChainAdapter` 是 GEPA 自带的、与 `dspy_adapter` **并列**的独立 adapter（`src/gepa/adapters/langchain_adapter/`），零 DSPy 依赖。DSPy 只是「另一个集成」，不是「优化 LangGraph 的前置」。

### 接入契约（3 个东西，源码 + `examples/langchain_adapter/pair_sum_product.py` 全链路）

```python
from gepa import optimize
from gepa.adapters.langchain_adapter import LangChainAdapter, last_message_text, make_reflection_lm

# ① rollout_fn(candidate, example) -> state dict：跑任意 pipeline，返回 LangGraph state dict
def rollout(candidate, example):   # candidate = {"system_prompt": "...", "node2_prompt": "..."}
    # 单轮 chat / create_agent 工具 agent / 自定义 LangGraph graph / RAG 都行
    return agent.invoke({"messages": [HumanMessage(content=example["input"])]})  # LangGraph state dict

# ② eval_fn(example, state) -> (score, feedback)：判真跑产出（结果指标），feedback 给反思读
def evaluate(example, state):
    resp = last_message_text(state)   # 支持 LangGraph 标准 state["messages"] 形状
    return (1.0, "Correct.") if example["answer"] in resp else (0.0, f"Wrong. Expected {example['answer']}.")

# ③ 组装 + optimize
adapter = LangChainAdapter(rollout_fn=rollout, eval_fn=evaluate, num_threads=...)
result = optimize(seed_candidate={"system_prompt": "..."}, trainset=..., valset=...,
                  adapter=adapter, reflection_lm=make_reflection_lm(reflection_llm), max_metric_calls=...)
```

### 关键差异（vs 之前学的 DefaultAdapter / 裸 evaluate）

- `LangChainAdapter` 走 `gepa.optimize(adapter=...)`，**不是** `task_lm=` + `evaluator=` 分传：task_llm 包在 `rollout_fn` 的 closure 里，reflection_llm 用 `make_reflection_lm()` 包成 `(prompt)->str`。
- `rollout_fn` 抛异常时 adapter 自动换 `{"messages":[AIMessage("ERROR:...")], "error": e}` 再调 eval_fn——**eval_fn 里查 `state.get("error")` 判失败给低分**（源码 163-164 行），别让它崩。
- `candidate` = `dict[str, str]` 命名文本组件，`rollout_fn` 决定各组件接到 pipeline 哪（system prompt / 某节点 prompt / 工具描述都可进化）。

### 用 qwenwm 14B@9090 跑（$0，关键配置）

- task_llm / reflection_llm 都走 `init_chat_model(provider:model, **kwargs)`，kwargs 直接传 `base_url/api_key`：
  `init_chat_model("ollama:qwen3:14b-32k", base_url="http://127.0.0.1:9090")`（需 `langchain-ollama` provider）。
- 例子的 `--task-model-kwargs` / `--reflection-model-kwargs` 就是 JSON kwargs，塞 `{"base_url": "http://127.0.0.1:9090"}`。
- 依赖：`gepa[langchain]` extra 只装 `langchain` + `langchain-core`；provider 包另装（`langchain-ollama` 连本地 9090）。

### 阻塞状态（2026-10-17 已解除，实验已启动）

原两个阻塞已全部解除，环境路径也定位到：

1. ✅ **qwenwm 9090 隧道已通**：`ssh` 进程在 9090 LISTEN，`curl /api/tags` 返回 `qwen3:14b-32k` 等模型。
2. ✅ **依赖已装 + 环境定位**：真实环境 = `/Users/max/Documents/mini_conda/envs/agent`（**不是** `/Users/max/mini_conda`，mini_conda 在 `~/Documents/` 下）。该 env 有 langchain 1.2.10 + litellm，gepa 0.1.4 已 editable install（`import gepa` 直接可用，`gepa.__version__` 属性不存在、验版本用 `pip show gepa`）。已补装 `langchain-ollama` + `langgraph`。

### 实验落地（2026-10-17）

脚本 `gepa_space/scripts/optimize_decision_chain_langchain.py`：决策链抽取案例用 **LangChain + LangGraph 包**，走 LangChainAdapter（不用 DSPy），端到端结果指标判分满分 12 = 结构3 + 落盘3 + 字段2 + 类型2 + 可执行2。

- sanity check 已跑通：烂 seed 0.01 vs 好 prompt 10.0，区分度成立（分数高 ⟺ 抽出的链越能被下游消费）。
- 完整 optimize 后台跑批中（max_metric_calls=12，task_lm/reflection_lm 都走 `init_chat_model('ollama:qwen3:14b-32k', base_url=9090)`）。
- LangGraph 外壳：`StateGraph` 单节点 `extract` 注入 candidate 的 system_prompt；rollout_fn 返回 `{"messages":[...]}` state dict。

### 打勾判定

**✅ 已打勾**（2026-10-17，jwasham「动手验证过」满足），经 v1→v2 两轮修正：

**v1（10.0 分，两处不闭环，被 Max 三问戳破）**：
- result 字段判分与契约矛盾（把「可空 result」算进「必填非空」）→ 好 prompt 被误扣 2 分。
- 「可执行」只用 action 长度≥4 粗判，没真走到下游消费。

**v2（12.0 满分，全链路闭环）**：脚本 `optimize_decision_chain_langchain_v2.py`
- 判分修复：`REQUIRED_NONEMPTY`（5 字段必填非空）vs `OPTIONAL`（result 可空），好 prompt 10.0→11.2。
- 全链路：日报 → 抽取(LangGraph) → 落盘 → 下游消费者「正放执行」（basis_type→路由档 + action→拆解可执行步骤），下游 outcome 进分数。
- 满分 12 = 结构3 + 落盘3 + 字段2 + 路由2 + 执行2（过程6 + 最终结果6）。
- 端到端：优化前(seed) 0.01/12 抽出错误结构 `{"decision_chain":...}`；优化后(best) **12.0/12**，抽出 5 条链，下游路由 5/5 + 执行 5/5，落盘 `registry_edae9072.md`。
- 分数证明成立：0.01 → 12.0 精确对应「抽不出 → 5 条链全可被下游路由+执行」。
