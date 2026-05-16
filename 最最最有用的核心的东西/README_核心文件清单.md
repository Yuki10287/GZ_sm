# 最核心文件清单

这个目录用于集中保存三问中最重要、最适合写论文和复现实验的材料。原始大目录中仍保留全部中间文件，这里只放主线文件。

## 全局说明

- `2026B成品油价格调控机制_总索引.md`：总索引，说明三问最终模型、关键结果、输出文件和说明文档用途。
- `数据处理与清洗流程整理.md`：按任务一、任务二、任务三整理全部数据来源、清洗、时间对齐、异常值处理和核心输出流向。

## 任务一

主脚本：

- `01_第九版最终模型_双因子状态空间_Ridge传导.py`
- `02_任务一1.2_NARDL完整检验.py`
- `03_任务一1.3_区间NARDL_LR检验.py`

核心输出在 `任务一/核心输出/`：

- `model_description.txt`：第九版模型说明。
- `ablation_transmission_metrics.csv`：传导层特征组合消融结果。
- `serial_fusion_validation.csv`：任务二使用的任务一核心输入表。
- `section_1_2_nardl_lr_complete.csv`：1.2 非对称传导检验结果。
- `section_1_2_nardl_robustness_complete.csv`：1.2 稳健性实验结果。
- `section_1_3_interval_nardl_lr_tests.csv`：1.3 区间检验结果。

## 任务二

主脚本：

- `task2_dynamic_pricing_dp.py`

核心说明：

- `任务二滚动优化主模型整理说明.md`
- `任务二_结果分析_最优调整策略特征与现行机制多维度对比.md`
- `任务二_三大核心特征的数据来源说明.md`
- `任务二_工作总结与输出说明.md`

核心输出在 `任务二/核心输出/`：

- `task2_strategy_comparison_summary.csv`：baseline 下现行实际、现行机制、最优策略总表。
- `task2_strategy_comparison_summary_by_scenario.csv`：不同政策权重情景比较。
- `task2_all_fuels_optimal_strategy.csv`：汽油、柴油 baseline 最优策略逐窗口结果。
- `task2_window_loss_breakdown.csv`：逐窗口损失分解。
- `task2_dynamic_pricing_report.txt`：任务二自动报告。
- `task2_baseline_gasoline_calibration.json`、`task2_baseline_diesel_calibration.json`：校准参数。
- `task2_apparent_consumption_monthly.csv`：表观消费量数据。

## 任务三

主脚本：

- `task3_rule_stability_under_noise.py`

核心说明：

- `任务三_第一部分_规则提取求解.md`：规则提取部分，重点是“调价压力缓冲账户下的分档比例释放规则”。
- `任务三_简化规则高斯扰动稳定性检验.md`：最终采用的鲁棒性检验和政策建议。
- `任务三_中国成品油价格调控机制改进建议.md`：最终政策建议，可直接支撑题目最后“提出具体建议”部分。

核心输出在 `任务三/核心输出/`：

- `task3_rule_stability_baseline.csv`：无扰动基准下简化规则指标。
- `task3_rule_stability_noise_simulations.csv`：加入高斯扰动后 500 次模拟结果。
- `task3_rule_stability_detailed_changes.csv`：每次扰动相对无扰动基准的变化。
- `task3_rule_stability_summary.csv`：鲁棒性检验汇总结果。

## 当前建议

论文主线建议使用：

1. 任务一第九版模型结果；
2. 任务二滚动优化主模型结果；
3. 任务三规则提取文档；
4. 任务三高斯扰动稳定性检验文档。

任务三之前做过的多情景压力测试和旧规则网格搜索可以作为备份，不建议作为论文主线。
