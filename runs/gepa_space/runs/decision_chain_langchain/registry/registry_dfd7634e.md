# 决策链登记表

| 决策链ID | 决策事项 | 决策链(state→decision→basis→action→result) | 依据类型 |
|---|---|---|---|
| dc-01 | 确定预测skill pipeline开发流程 | 需确认是否按提出的7步流程开发预测skill → 按提出的流程覆盖式写入spec → 参考ml业界重构最佳判断 → 输出skill spec并执行开发 →  | 推断 |
| dc-02 | 参数化分支代码设计 | 需通过外部参数表控制分支逻辑 → 采用参数表串联设计 → 参数表可灵活控制分支逻辑 → 设计参数表与代码的映射关系 →  | 偏好 |
| dc-03 | 训练skill pipeline spec输出 | 需输出训练脚本的spec文档 → 按预测spec格式输出训练spec → 训练与预测需保持流程一致性 → 编写train_param_save_with_prom_spec.md →  | 铁律 |
| dc-04 | pipeline开发spec纠偏 | 当前spec存在目录结构和执行主体不明确问题 → 明确skill.md联动规则和执行主体 → 需符合标准skill文件夹规范 → 修正spec中目录结构和执行主体说明 →  | 铁律 |