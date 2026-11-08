# 带时间的和声条件：首次织体训练对照

2026-11-08，用户确认 test1 的柱式和弦与弱起时间正确。随后已完成 **旋律 + 带时间和声 → 伴奏织体** 的训练与推理接口，以及一次配对实验。模型会利用条件，但尚未达到稳定遵守和声或 Animenz 风格迁移验收的程度。当前默认模型保持不变，新模型单独存放。

## 已确认与待确认的边界

- test1 的柱式版已通过试听，原话、日期与音频/MIDI 哈希保存在 `examples/harmony/test1_d_major_4563.json` 的 `audition_acceptance`。
- test1 不进入训练或验证集，仅用于外部输入的生成检查。其弱起前段仍为 `X`，不是已知无和弦。
- 40 首本地曲目的和声来自参考伴奏自动识别，仍为弱监督，未因 test1 的验收而升级为人工真值。
- 四首 POP909 保留在标注库，本次配对训练没有使用它们。
- 新模型输出未经用户试听验收。没有用自动音高纠正、强制和弦音或补低音来抬高本次指标。

## 接口改动

新增 `pianalysis_timed_harmony_v2` 条件协议；音符继续使用 `pianalysis_tsd_v2`。条件保留：时间增量、持续时值、根音音级、独立低音音级、完整音级集合，以及彼此不同的 `N` 与 `X`。新增 39 个 token，词表从 2178 扩至 2217。表示音级集合使七和弦、减七和弦和转位都不必静默降成三和弦。

```text
[BOS] melody [HARMONY] timed_chord_spans [SEP] accompaniment [EOS]
```

和声原始整数 tick 通过该曲的速度图转换为秒；条件与音符共用 10 ms 网格。边界量化误差通常不超过 5 ms，两端量化到同一点的极短片段无法保留。模型输入不声称与原始 tick 完全等价。

切窗会裁切并平移持续中的和弦，填充 `X` 缺口，合并内容与来源/状态相同的相邻片段。原始完整标注仍保留逐拍证据。训练和推理调用相同的时间处理与编码函数，和声所占 token 也计入窗口预算。

旋律和和声提示均被 `-100` 屏蔽，不参与目标 loss。模型配置保存独立的和声协议版本；缺条件、版本不匹配、残缺条件或把全局 tick 计划误交给旧版无时间接口均会拒绝。

## 数据与训练

| 项目 | 结果 |
| --- | --- |
| 配对数据 | 40 首、1530 窗口；与旧 TSD 数据的 ID、时间边界、source/target token 逐项相同 |
| 划分 | 原曲级 36/4，1306 个训练窗口、224 个验证窗口 |
| 留出曲目 | apple-seed、frieren-beyond-journeys-end-medley、gurenge、vogel-im-kafig |
| 长度 | 平均 506.63 token，最大 915，上下文上限 1024 |
| 模型 | GPT-2，6 层、8 heads、512 embedding，seed=42 |
| 训练 | batch=8，学习率 5e-4，warmup=200，dropout=0.1，weight decay=0.01，最多20轮，早停耐心3次评估 |
| 实际执行 | 13.7195 轮、2250 step、244.4 秒；保存最佳 checkpoint-1500 |
| Trainer 验证 loss | 1.706151；该单项未优于原 TSD 的1.654079 |
| 硬件 | RTX 4070 SUPER，Torch 2.5.1+cu121 |

数据构建必须显式传 `--allow-unreviewed-harmony` 才能把未验收标签用于弱监督试验。本次保存了该标志及来源范围。旧模型、旧数据集和已通过试听的柱式文件不覆盖。

## 条件是否被使用

固定同一个新模型与相同 target，对全部 224 个留出窗口分别输入正确和声，以及所有和弦根音/低音/音级整体升半音的条件。旋律、时间、目标、mask 保持不变。

| 条件 | 目标 token 加权负对数似然，越低越好 |
| --- | ---: |
| 正确和声 | 1.59804 |
| 升半音和声 | 1.67078 |

193/224 个窗口使用正确条件时损失更低。该统计使用所有有效 target token（含 EOS）加权，与 Trainer 的批次平均口径不同。这支持“模型使用了条件”，不能推出“模型每个生成音都服从条件”。和声由目标伴奏提取，这是给定参考和声的条件任务，不是旋律自动配和声实验。

## 实际生成：54 次对照

每首留出曲目选首个与中间窗口，共 8 段；seed=7、42。三组分别为旧 TSD、正确和声的新模型、升半音和声的新模型，共48次短窗生成。test1 三组各两次，共6次长旋律生成。temperature=0.7、top_k=30、top_p=0.9；每窗预算384 token，均启用音符语法和时间上限约束。

和弦内音比例按音符与有和弦标签区间的交叠时长加权；低音音级也视为允许音。`X` 与 `N` 不参与该指标。经过音、倚音可能合理，所以这不是音乐质量评分。只有三组都有有效分母的相同15次短窗用于配对比较；旧模型的一次生成没有覆盖任何有和弦标注的时间，因此不能把它计为0或单独从某组平均中删除。完整16次覆盖率与音符数另计。

| 留出片段 | 旧 TSD | 新模型＋正确和声 | 新模型＋升半音和声 |
| --- | ---: | ---: | ---: |
| 对原参考和声的内音比例，配对15次 | 44.37% | 52.23% | 24.90% |
| 伴奏时间覆盖率，全部16次 | 84.35% | 87.00% | 89.44% |
| 平均原始音符数 | 54.75 | 62.81 | 58.56 |
| 末尾空缺，秒 | 0.768 | 0.531 | 0.439 |
| 语法合法 | 16/16 | 16/16 | 16/16 |

两个窗口的可用和弦标签时间占比偏低（约34.8%与11.8%），该处内音比例尤其不稳。所有逐次结果保存在 summary 的 runs 中，应结合标签复核解读。

| 外部 test1，两种 seed 均值 | 旧 TSD | 新模型＋正确和声 | 新模型＋升半音和声 |
| --- | ---: | ---: | ---: |
| 对已验收和声的内音比例 | 47.91% | 50.10% | 27.28% |
| 伴奏覆盖率 | 86.13% | 99.80% | 99.39% |
| 原始音符数 | 104.0 | 91.5 | 98.0 |

test1 覆盖更完整，但内音比例只改善2.19个百分点，seed间表现仍有波动。结论是“条件通路有效，和声服从度仍不足”。不能把高覆盖率或语法通过当成织体可接受，更不能据此认定40首足够学会目标风格。

## 试听与证据

- [新旧模型与柱式基线试听页](data/harmony_conditioning_v2/evaluation/index.html)：默认展示预先固定的 seed 7，包含错调诊断组。
- [新模型原始试听](data/harmony_conditioning_v2/evaluation/timed_harmony_test1_seed7.wav) / [MIDI](data/harmony_conditioning_v2/evaluation/timed_harmony_test1_seed7.mid)。原始生成无音高纠正或后处理；MIDI 使用通用秒时间导出，和声条件本身为10ms量化。
- [配对数据校验](data/harmony_conditioning_v2/dataset_validation.json)、[训练日志](data/harmony_conditioning_v2/training.log)、[目标条件消融](data/harmony_conditioning_v2/evaluation/target_nll.json)、[54次生成结果](data/harmony_conditioning_v2/evaluation/summary.json)。
- CLI 另有一次端到端验证 `data/harmony_conditioning_v2/cli_test1.json`；此文件包含既有保守后处理结果，未用于上述原始指标。

下一步先试听新织体，按失败窗口复核和声标签、低音与声部，再考虑增加可靠数据或加入和声控制约束。当前不把新模型提升为默认模型，也不把一次训练当成最终风格迁移结论。

## 验证记录与复现

TDD 目标是让时间条件跨数据构建、训练 mask、切窗推理保持一致。首批7项测试在缺失实现时出现 `NotImplementedError` 与缺少条件字段；后补两项测试先复现“缺条件”和“全局计划误入旧接口”未被拒绝。实现后9项全部通过；全量63项通过。没有额外结构重构。验收记录与实验脚本属于非TDD修改，使用真实 MIDI 哈希、数据配对、指标手算校准和产物检查验证。

```powershell
python -m unittest discover -s tests -v
python scripts/build_training_windows_v1.py --annotated-dir data/harmony_v2_review/annotated_notes --out-dir data/training_windows_harmony_timed_v2 --reference-windows data/training_windows_tsd_v2/dataset_windows_v1.json --tokenizer-version pianalysis_tsd_v2 --allow-unreviewed-harmony --max-length 1024 --window-seconds 8
python train_v2.py --data-file data/training_windows_harmony_timed_v2/dataset_windows_v1.json --output-dir model_output/accompaniment_gpt2_harmony_timed_v2_new --report-to none
python scripts/evaluate_harmony_conditioning.py --model model_output/accompaniment_gpt2_harmony_timed_v2_new/final_model --out-dir data/harmony_conditioning_v2/evaluation_new
```

已有实验模型为 `model_output/accompaniment_gpt2_harmony_timed_v2/final_model`。复现实验请使用新目录；评估脚本拒绝覆盖已有 summary。代码与验收配置进入 Git，大型数据与模型保留在本地忽略目录。
