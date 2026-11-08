# TSD V2 对照实验结果

本轮已完成显式时值编码、生成语法约束、长旋律切窗和 POP909 导入。固定现有数据与原曲划分后重新训练一个 V2 模型，并完成 58 次生成对照。短窗口原始伴奏覆盖率从 V1.1 的 72.82% 提升到受约束 V2 的 84.35%，语法合法输出从 10/16 提升到 16/16。音乐听感尚未评价，覆盖率不能替代音乐质量。

实现与上游出处见 [UPSTREAM_ADOPTION.md](UPSTREAM_ADOPTION.md)。全部数值来自本地 [summary.json](data/tsd_v2_audit/evaluation/summary.json)，指标使用清理前的原始伴奏。

## 数据与训练控制

| 项目 | 本轮设置/结果 |
| --- | --- |
| 训练数据 | 原 40 首 MIDI 的 DP 弱标注，未加入 POP909 或和声提示 |
| 窗口配对 | 复用 V1.1 的 1530 个样本 ID、时间边界和音符数量 |
| 训练/验证 | 36/4 首原曲；1306/224 个窗口；划分与 V1.1 相同 |
| 时间表示 | 10 ms 网格；相对 TIME + 显式 Duration；词表 2178 |
| 平均序列长度 | 573.84 → 443.63 token，减少约 22.69% |
| V2 最大序列长度 | 825 token，模型上下文上限 1024 |
| 往返验证 | 全部参考窗口的量化旋律和伴奏音符精确还原 |
| 模型 | GPT-2，6 层、8 heads、512 embedding |
| 训练参数 | seed=42，batch=8，学习率 5e-4，最多 20 epoch，early stopping patience=3 |
| 实际停止 | epoch 13.7195，训练耗时 232.0 秒；保存最佳验证 checkpoint |
| 最终评估 loss | 1.654079；不同表示的 token loss 不直接用于比较音乐质量 |
| 运行环境 | Python 3.11、Torch 2.5.1+cu121、Transformers 4.57.1、RTX 4070 SUPER |

V2 模型位于 `model_output/accompaniment_gpt2_tsd_v2_e20/final_model`。目录后缀 `_e20` 表示最大训练预算，不能解释为实际跑满 20 epoch。

数据证据：[dataset_validation.json](data/tsd_v2_audit/dataset_validation.json)。训练证据：[training.log](data/tsd_v2_audit/training.log)、[train_results.json](model_output/accompaniment_gpt2_tsd_v2_e20/train_results.json)、[eval_results.json](model_output/accompaniment_gpt2_tsd_v2_e20/eval_results.json)。

## 短窗口：16 次/条件

从未参与训练的 `apple-seed`、`frieren-beyond-journeys-end-medley`、`gurenge`、`vogel-im-kafig` 各取首个和中间窗口，分别使用 seed=7、42。每次预算 384 token，temperature=0.7、top_k=30、top_p=0.9。以下为每次生成的均值，合法次数除外。

| 指标 | V1.1 | TSD 无约束 | TSD 语法+时间约束 |
| --- | ---: | ---: | ---: |
| 原始伴奏时间覆盖率 | 72.82% | 86.47% | 84.35% |
| 语法合法次数 | 10/16 | 10/16 | 16/16 |
| 时长 >3 秒的原始音符数 | 8.00 | 0.00 | 0.00 |
| 同音高重叠音符数 | 21.50 | 7.13 | 4.31 |
| 末尾空缺（秒） | 2.174 | 0.578 | 0.767 |
| 超出旋律末尾（秒） | 0.170 | 3.325 | 0.000 |
| 原始伴奏音符数 | 48.50 | 66.56 | 54.75 |

无约束 TSD 的覆盖更高，但仍会产生不完整事件和超时输出；加入约束后这批结果都满足语法与时间范围。16 次受约束输出都以 EOS 结束，其中 7 次由预算边界强制、1 次由时间边界强制，不能全部归为自然收尾。

这里的约束同时包含语法和时间边界，本次实验没有分别隔离这两种作用。零 >3 秒音符是该批原始输出的观察值，生成约束本身仍允许窗口内最长 8 秒的音符。

## 长旋律：2 次/条件

输入为 `test/test1.mid`，量化后旋律时长约 14.77 秒，seed=7、42。切窗为两段、每段独立预算；另增加单次生成的大预算对照，避免把额外 token 预算的收益都归因于切窗。

| 条件 | 原始覆盖率 | 末尾空缺（秒） | 有效总 token 预算 | 同音高重叠数 |
| --- | ---: | ---: | ---: | ---: |
| V1.1，单次 | 62.56% | 5.530 | 384 | 31.0 |
| TSD，无约束单次 | 80.20% | 2.925 | 384 | 0.0 |
| TSD，受约束单次 | 80.20% | 2.925 | 384 | 0.0 |
| TSD，受约束大预算单次 | 87.98% | 1.775 | 754 | 1.5 |
| TSD，受约束切窗 | 86.12% | 0.090 | 768 | 3.0 |

大预算单次请求 768 token，受输入长度限制实际可用 754；切窗为 2 × 384。所有 TSD 长旋律条件的原始 >3 秒音符数及超时均为 0。

切窗的主要收益是末尾覆盖更稳定；大预算单次的总体覆盖率略高。当前切窗独立生成，没有跨窗伴奏记忆或乐句规划，不能据此声称整曲连贯性已解决。58 次对照指 48 次短窗条件输出与 10 次长旋律条件输出；长旋律切窗输出各含两次模型调用。

## 指标口径与限制

- 覆盖率是旋律时间范围内被任一伴奏音符覆盖的时间并集占比，不是拍点准确率或和声匹配率，长持续音也会抬高该值。
- 末尾空缺为 `max(0, 旋律末尾 - 最后伴奏结束)`；超时为反向差值。同音高重叠按每个起音与此前同音高音符的最晚结束时间比较。
- 语法合法指没有未知 token、错误参数类型、残缺事件或错误角色事件；不等于音乐合理，也不保证自然 EOS。
- V2 不需要 NOTE_OFF，所以缺失 NOTE_OFF 的消失属于表示结构性质。
- 指标在清理前计算。输出 MIDI 包含旋律与经过音域、密度、重叠等清理的伴奏；清理报告另存 JSON，不能用清理后的数值替代模型原始表现。
- 只有一个训练随机种子、四首留出歌曲和一段长旋律，尚无主观盲听、跨曲风检验或统计显著性结论。

## POP909 接入验证

已下载官方 001–004 四首 MIDI，按轨道名导入 4950 个旋律/伴奏音符。BRIDGE 排除并单独统计；tempo、拍号、调号及分辨率保留在标注 JSON。

构建得到 85 个窗口，全部通过训练数据读取校验。23 个窗口被拒绝：22 个缺少旋律，1 个没有有效音符。按原歌曲划分后为 63 个训练窗口、22 个验证窗口，歌曲身份无交集。同曲不同版本共享 `source_piece_id` 的行为另有回归测试。

这 4 首样本只验证导入与训练输入兼容性，未加入本轮模型训练，因此不能将上面的生成改善归因于 POP909。完整数据集训练与拍点 sidecar 接入仍待后续验证。

证据：[import_report.json](data/pop909_sample_imported/import_report.json)、[window_report.json](data/pop909_sample_windows/window_report.json)、[source_manifest.json](data/pop909_sample/source_manifest.json)。

## 交付与验证

`python -m unittest discover -s tests -v`：32 项通过。默认生成命令已在真实模型上运行，生成的两个窗口 token 与对照实验 seed=7 一致。源码、数据、模型与结果的 SHA256 清单见 [manifest.json](data/tsd_v2_audit/manifest.json)，最终静态与数据检查见 [final_validation.json](data/tsd_v2_audit/final_validation.json)。这些大型/生成文件保存在本地 Git 忽略目录。

- [当前默认生成 MIDI](test/test1_tsd_v2.mid) / [生成诊断 JSON](test/test1_tsd_v2.json)
- [V1.1 对照 MIDI，seed=7](data/tsd_v2_audit/evaluation/v1_1/standalone_melody_seed7.mid)
- [TSD 大预算单次 MIDI，seed=7](data/tsd_v2_audit/evaluation/tsd_larger_budget/standalone_melody_seed7.mid)
- [TSD 切窗 MIDI，seed=42](data/tsd_v2_audit/evaluation/tsd_windowed/standalone_melody_seed42.mid)

下一步最有价值的是对这些对照 MIDI 做听感标注，确认持续音、密度和窗口衔接的改善是否转化为更合适的伴奏。
