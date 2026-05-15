任务一最终附录代码说明
============================================================

本目录用于放入论文附录或提交给队友复现任务一结果。

01_第九版最终模型_双因子状态空间_Ridge传导.py
------------------------------------------------------------
任务一主模型脚本。
模型结构为：
双因子状态空间模型 + 修正调价机制层 + RidgeCV 政策传导层。
输出目录：
outputs/task1_serial_fusion_ridge


02_任务一1.2_NARDL完整检验.py
------------------------------------------------------------
对应新版文档 1.2。
包含：
1. ADF 平稳性检验；
2. 格兰杰因果检验；
3. 基础 NARDL-LR 似然比检验；
4. 扩大滞后、替换油价代理、Threshold NARDL 稳健性实验。

主要输出：
outputs/task1_serial_fusion_ridge/section_1_2_data_tests.csv
outputs/task1_serial_fusion_ridge/section_1_2_granger_tests.csv
outputs/task1_serial_fusion_ridge/section_1_2_nardl_lr_complete.csv
outputs/task1_serial_fusion_ridge/section_1_2_nardl_robustness_complete.csv
任务一第九版_1.2完整检验结果.txt


03_任务一1.3_区间NARDL_LR检验.py
------------------------------------------------------------
对应新版文档 1.3。
严格按照低油价、正常油价、高油价三个区间构造虚拟变量，
并将区间虚拟变量与 NARDL 长期正负累计项相乘。
检验方法为 LR 似然比检验。

主要输出：
outputs/task1_serial_fusion_ridge/section_1_3_interval_nardl_lr_tests.csv
任务一第九版_1.3区间NARDL_LR检验结果.txt


运行顺序建议
------------------------------------------------------------
1. 先运行 01，生成第九版主模型输出；
2. 再运行 02，生成 1.2 所需统计检验结果；
3. 最后运行 03，生成 1.3 所需区间 LR 检验结果。
