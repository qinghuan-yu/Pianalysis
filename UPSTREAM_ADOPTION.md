# 上游经验采用记录

本轮把 MidiTok 的显式时值表示、事件类型约束，以及 POP909 的分轨语义接入现有训练和生成流程。默认协议是 `pianalysis_tsd_v2`，已重新构建数据、训练独立模型并完成对照实验。实测结果见 [TSD_V2_RESULTS.md](TSD_V2_RESULTS.md)。

## 来源与实际采用范围

| 上游来源 | 本项目采用的经验 | 实现状态 |
| --- | --- | --- |
| [MidiTok TSD](https://github.com/Natooz/MidiTok/blob/main/src/miditok/tokenizations/tsd.py) | 用 Pitch、Velocity、Duration 描述完整音符，避免依赖后续 NOTE_OFF | 已实现本地 TSD 风格编码；保留 10 ms 网格，没有引入 MidiTok 依赖或复制其编码器 |
| [MidiTok REMI](https://github.com/Natooz/MidiTok/blob/main/src/miditok/tokenizations/remi.py) | 明确允许的事件类型和后继关系 | 已将本项目语法接入 Hugging Face `LogitsProcessor`；上游类型图不等于自动开启生成约束 |
| [MidiTok 预处理说明](https://github.com/Natooz/MidiTok/blob/main/docs/tokenizing_music_with_miditok.rst) | 显式管理量化、轨道合并和音符边界 | 固定原有量化做对照；POP909 按轨道名保留角色，避免同为钢琴的旋律与伴奏被合并 |
| [POP909](https://github.com/music-x-lab/POP909-Dataset) | 利用已有 MELODY、BRIDGE、PIANO 角色，按原歌曲划分数据 | 已实现导入器，验证 4 首官方样本；同曲版本共享划分身份；尚未用于本轮模型训练 |
| [Compound Word 模型](https://github.com/YatingMusic/compound-word-transformer/blob/main/workspace/uncond/cp-linear/main-cp.py)及[数据流程](https://github.com/YatingMusic/compound-word-transformer/blob/main/dataset/Dataset.md) | 音高、力度、时值分因子表达；拍点对齐有独立的数据前提 | 当前使用分字段平铺 token；多头输出、嵌入聚合与拍点对齐尚未实现 |
| [AccoMontage2](https://github.com/billyblu2000/AccoMontage2/blob/master/docs/README.md) | 长伴奏需要乐句与和声结构 | 保留为后续方向；当前固定时间切窗没有移植其检索、乐句规划或调性处理 |

以上链接指向上游实现或说明。POP909 样本的下载来源、Git blob SHA 与许可分别保存在本地 `data/pop909_sample/source_manifest.json` 和 `LICENSE`；本项目编码器和生成约束是独立实现。

## 当前协议和生成行为

`scripts/tokenizer_v2.py` 使用以下事件：

```text
TIME TIME_<1..800>
NOTE_ON_MELODY PITCH_<0..127> VELOCITY_<1..127> DURATION_<1..800>
NOTE_ON_ACCOMP PITCH_<0..127> VELOCITY_<1..127> DURATION_<1..800>
```

- `TIME` 是相邻起音之间的相对时间；长停顿可拆成多个 TIME 事件。
- 默认每 tick 为 10 ms，单音时值支持 0.01–8 秒。超过范围明确报错，需要先切窗，不静默截短。
- Duration ID 为 1378–2177，完整词表大小 2178。V1.1 的参数与和声范围保留，NOTE_OFF ID 保留但不用于 V2 音符。
- 同音高重叠音符可独立往返编码。没有完成所有字段的音符不会被解码成持续音。
- V1、V1.1、V2 分别识别版本；V2 需要新数据和新 checkpoint，不能直接给旧模型换词表。

`scripts/duration_constraints.py` 在生成时约束 `ACC → pitch → velocity → duration`，事件边界仅允许 ACC、TIME 或 EOS。它为完整事件和末尾 EOS 留出预算，并在提供目标时长时限制时间推进和音符结束。JSON 中的 `forced_eos_reason` 区分预算耗尽与到达时间边界；输出 EOS 不一定代表模型学会了自然收尾。

约束不设置 3 秒的时值上限。结果表中的长音数基于原始输出，独立于 MIDI 导出前的清理步骤。语法约束也不保证和声、声部进行、密度或同音高重叠的音乐合理性。

对于旋律 MIDI，V2 默认使用 checkpoint 记录的训练窗长（本轮 8 秒）。密集旋律继续按时间细分，不截断输入 token；每个窗口独立生成，再恢复全局时间。无旋律的窗口跳过生成。导出保留原旋律音符的秒级起止与力度；MIDI 的 tempo、拍号等元数据目前保存在输入 JSON 中，没有承诺原样写回导出 MIDI。

当前各窗没有上一窗伴奏记忆，跨边界音符会在窗口内裁切，尚不能保证乐句连贯性。`--window-seconds 0` 关闭切窗。切窗暂不接受 `--harmony-json`；和声条件模型应使用适配的单窗输入。当前评估的 V2 模型没有加入和声提示。

## 使用与复现实验

从项目根目录运行，先在可用 Python 环境安装 `requirements.txt`。训练使用 Transformers Trainer，还需环境中安装 `accelerate`。模型和数据属于本地生成物，不随 Git 分发。

已有本轮模型时，直接使用新默认值：

```powershell
python generate_from_scratch.py --melody-midi test/test1.mid --seed 7 --out-midi test/test1_tsd_v2.mid --out-json test/test1_tsd_v2.json
```

默认模型为 `model_output/accompaniment_gpt2_tsd_v2_e20/final_model`，每窗预算 384，采样参数为 temperature=0.7、top_k=30、top_p=0.9。`--no-grammar-constraints` 用于消融对照。

以下命令以独立目录复建本轮训练数据和模型。需要已有 DP 标注与 V1.1 参考窗口；普通新数据构建可以省略 `--reference-windows`，但窗口数量可能改变，不能当作严格配对对照。

```powershell
python scripts/build_training_windows_v1.py --annotated-dir data/dp_cleaned_v1/annotated_notes --out-dir data/training_windows_tsd_v2_repro --tokenizer-version pianalysis_tsd_v2 --reference-windows data/training_windows_v1_1/dataset_windows_v1.json --window-seconds 8 --max-length 1024
python train_v2.py --data-file data/training_windows_tsd_v2_repro/dataset_windows_v1.json --output-dir model_output/accompaniment_gpt2_tsd_v2_repro --epochs 20 --seed 42
python scripts/evaluate_tsd_v2.py --candidate-model model_output/accompaniment_gpt2_tsd_v2_repro/final_model --candidate-dataset data/training_windows_tsd_v2_repro/dataset_windows_v1.json --out-dir data/tsd_v2_audit/evaluation_repro
```

评估器比较同一批参考窗口和原曲划分，并拒绝覆盖已有 `summary.json`。`_e20` 是最大 epoch 预算的命名，本轮模型实际提前停止于 epoch 13.72。不同硬件和库版本可能影响采样或训练复现。

POP909 导入示例（`--dataset-dir` 指向包含 `001/` 等编号目录的文件夹）：

```powershell
python scripts/import_pop909.py --dataset-dir data/pop909_sample --out-dir data/pop909_sample_imported
python scripts/build_training_windows_v1.py --annotated-dir data/pop909_sample_imported/annotated_notes --out-dir data/pop909_sample_windows --tokenizer-version pianalysis_tsd_v2 --window-seconds 8
```

`MELODY → melody`，`PIANO → accompaniment`，BRIDGE 明确排除并计数。未知轨道或缺少有效 MELODY/PIANO 时失败，不退回 Skyline 猜测。`--include-versions` 可纳入不同版本；`piece_id` 标识文件，`source_piece_id` 标识原歌曲，训练划分使用后者。MIDI timing、来源哈希和已有 sidecar 路径被保留；sidecar 的拍点、和弦内容尚未读入模型。

## 测试记录（RED → GREEN）

目标行为、开发阶段观察到的失败和对应最小实现如下。RED 为开发时观察摘要，未单独保存日志；当前代码已经进入 GREEN，下面命令用于复跑这些测试。

| 测试组 | RED：目标行为缺失 | GREEN：对应实现 |
| --- | --- | --- |
| `tests/test_tokenizer_v2.py` / `DurationEncodingTests` | 尚不支持 V2 协议；训练读取未校验 Duration 字段 | 新增显式时值 codec、版本分发、参数验证和固定参考窗口构建 |
| 同文件 / `DurationGenerationTests` | 生成可选 NOTE_OFF 或音符中途 EOS，预算可截断事件 | 接入有限状态语法与预算/时间边界约束；使用真实小型 GPT-2 验证 logits 屏蔽 |
| 同文件 / `WindowedGenerationTests` | 缺少切窗生成接口 | 增加窗口规划、全局时间恢复和静音窗口跳过；仅随机生成边界使用 mock |
| `tests/test_pop909_import.py` | 导入入口尚未实现，抛出 `NotImplementedError` | 按轨名映射角色、共享原歌曲身份、拒绝缺失旋律轨 |

对应测试命令：

```powershell
python -m unittest discover -s tests -p test_tokenizer_v2.py -v
python -m unittest discover -s tests -p test_pop909_import.py -v
python -m unittest discover -s tests -v
```

最终全量结果：32 项通过，其中既有协议回归 17 项，V2 编码/约束/切窗 12 项，POP909 导入 3 项。重构保持在版本分发和现有流程整合范围，没有另外替换模型架构；整合后全量测试通过。默认 CLI 两窗生成 token 与基准评估的 seed=7 输出完全一致。数据往返、真实训练和 58 次生成实验另见结果报告。

## 后续验证顺序

1. 对当前 MIDI 做盲听与错误标注，衡量和声适配、节奏、密度和窗口衔接。
2. 在固定歌曲划分下接入更多 POP909 数据，再验证角色标签对质量的实际影响。
3. 利用已保留的 tempo/拍号与 POP909 拍点信息验证 beat/bar 对齐，再进入 [TOKENIZER_V2.md](TOKENIZER_V2.md) 的 REMI 扩展。
4. 有了可靠的拍点和乐句表示后，再比较 Compound Word 多头模型与 AccoMontage 式结构规划。
