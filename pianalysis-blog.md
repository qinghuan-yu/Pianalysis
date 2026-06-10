---
title: "Pianalysis 技术解析：从旋律 MIDI 到风格化钢琴伴奏生成"
date: "2026-06-10"
desc: "从数据清洗、旋律抽取、token 设计、条件训练到 MIDI 生成闭环的完整工程复盘"
tags: ["深度学习","符号音乐生成","Transformer","MIDI","项目"]
---

# Pianalysis 技术解析：从旋律 MIDI 到风格化钢琴伴奏生成

## 0. 写在前面

Pianalysis 的目标是输入一段旋律 MIDI，让模型补全钢琴伴奏织体，最后输出一份保留原旋律、带有风格化伴奏的 MIDI。

这不是音频生成任务，而是符号音乐生成任务。模型不直接预测波形，而是在 MIDI 事件被编码后的 token 序列上做条件生成。

当前工程闭环是：

```text
MIDI 数据集
-> 增强 Skyline + 动态规划抽取旋律
-> 拆分 melody / accompaniment
-> 编码为条件 token 序列
-> 切成可训练窗口
-> GPT-2 只学习 accompaniment target
-> 输入 melody prompt 生成 accompaniment
-> 解码并导出 MIDI
```

这篇文章会完整拆解其中的技术原理、关键运算、工程选择、目前效果和后续改进方向。

## 1. 任务定义：条件编配

理想产品形态是：

```text
input:  melody.mid
output: arranged.mid = original_melody + generated_accompaniment
```

这里有一个关键选择：模型不负责重写旋律，只负责生成伴奏。也就是说，训练目标不是 `melody -> full arrangement`，而是：

```text
melody -> accompaniment
```

推理时再把原始旋律和生成伴奏合并：

```text
final_midi = input_melody_midi + generated_accompaniment_midi
```

这样做的好处是输入旋律不会被模型改坏，训练目标更清楚，模型只需要学习“如何围绕旋律补织体”。

当前版本使用 GPT-2 风格的 causal language model。训练序列被拼成：

```text
[BOS] source_melody [SEP] target_accompaniment [EOS]
```

模型仍然做 next-token prediction，但 loss 只计算 `[SEP]` 后面的 accompaniment 部分。

## 2. 为什么 MIDI 不能直接训练

原始 MIDI 虽然结构化，但它不会告诉我们：

```text
哪一个音是旋律？
哪一些音是伴奏？
哪一些音只是装饰音？
哪个声部是主线？
```

对于钢琴独奏尤其麻烦，因为右手旋律、右手分解和弦、左手低音、内声部常常混在一个或少数几个轨道里。

如果直接训练完整 MIDI，而 source 又来自粗糙旋律抽取，模型会学到很多脏关系：

```text
错误旋律 -> 错误伴奏
伴奏高音 -> 被误判为旋律
旋律八度重复 -> 被错误拆分
装饰音 -> 被当作主旋律
```

因此 Pianalysis 的第一步不是训练模型，而是构造一个稳定的数据生产线。

## 3. 旋律抽取：增强 Skyline + 动态规划

原始 Skyline 的基本假设是：

```text
同一时刻最高音 = 主旋律
```

它简单、快、容易实现，但对复杂钢琴编曲很容易误判。右手分解和弦、装饰音、八度铺陈、反旋律和高音区伴奏都可能被错当成主旋律。

当前工程采用更稳的轻量方案：

```text
每个 onset 取 top-k 候选音
-> 给候选音打局部分
-> 用动态规划寻找最连贯的旋律路径
-> 后处理去掉孤立跳进和部分八度重复
```

核心观念是：旋律不是某个瞬间最高的点，而是一条时间上连续、音高运动合理、节奏上有重心的线。

### 3.1 候选音分组

先把音符按 onset 时间量化分组：

```text
onset_tick = round(note.start * 1000 / quantum_ms)
```

当前默认：

```text
quantum_ms = 10
top_k = 5
```

每个 onset group 按音高从高到低取前 5 个候选音。这样旋律被装饰音短暂盖住时，仍有机会被动态规划选中。

### 3.2 候选音局部分数

每个候选音得到一个局部分数：

```text
local_score(note) =
  0.95 * pitch_height
+ 0.62 * duration_score
+ 0.22 * velocity_score
+ 0.48 * rank_score
+ metrical_weight
- density_penalty
- short_note_penalty
- low_pitch_penalty
```

其中：

```text
pitch_height = (pitch - 21) / (108 - 21)
duration_score = min(duration / 0.75, 1.2)
velocity_score = velocity / 127
rank_score = 1 / (rank_from_top + 1)
density_penalty = min((chord_size - 1) * 0.08, 0.45)
```

极短音会被惩罚：

```text
duration < 0.055s -> -1.20
duration < 0.10s  -> -0.55
duration < 0.16s  -> -0.20
```

这用于抑制装饰音、碎音和快速琶音误判。

### 3.3 动态规划转移分数

候选音之间的转移分数主要考虑：

```text
interval = abs(current.pitch - previous.pitch)
onset_gap = current.start - previous.start
rest_gap = current.start - previous.end
```

简化规则：

```text
transition_score =
  -0.055 * interval
  -0.45  if interval > 12
  -0.85  if interval > 19
  +0.18  if interval <= 2 and onset_gap <= 1.2
  +0.12  if interval <= 5 and onset_gap <= 1.2
  -0.50  if onset_gap < 0.08 and interval > 7
  -0.15  if rest_gap > 2.5
```

动态规划递推公式是：

```text
dp[t][j] =
  local_score(c_tj)
  + max_i(dp[t-1][i] + transition_score(c_{t-1,i}, c_tj))
```

同时记录 backpointer，最后从最高分状态回溯，得到旋律 note id 集合。

### 3.4 后处理

DP 路径之后还会：

1. 去掉短时值且前后都是大跳的孤立高音。
2. 对同 onset 的八度重复，默认保留高音，丢掉低八度。

这并不完美，但能减少一部分“旋律条件过厚”的问题。

## 4. 数据清洗产物

运行：

```powershell
python scripts/dp_melody_cleaning_v1.py --midi-dir MIDI --out-dir data\dp_cleaned_v1 --write-midi
```

会生成：

```text
data/dp_cleaned_v1/dataset_dp_v1.json
data/dp_cleaned_v1/cleaning_report.json
data/dp_cleaned_v1/annotated_notes/*.json
data/dp_cleaned_v1/melody_midi/*_melody.mid
data/dp_cleaned_v1/accompaniment_midi/*_accompaniment.mid
data/dp_cleaned_v1/annotated_midi/*_annotated.mid
data/dp_cleaned_v1/roundtrip_midi/*_roundtrip.mid
```

本地 40 首 MIDI 的清洗结果：

```text
Processed: 40
Failed: 0
Average melody ratio: 38.33%
Minimum melody ratio: 12.86%
Maximum melody ratio: 58.42%
Average sequence length: 20966 tokens
Maximum sequence length: 48859 tokens
```

这说明增强 Skyline + DP 可以批量生产弱标注，但仍然不能替代人工听检。

## 5. MIDI 到 token：可逆闭环

早期版本最大的问题之一是训练能跑，但 token 到底能不能还原成 MIDI 不确定。当前项目先做了闭环验证：

```text
MIDI -> note JSON -> token -> note JSON -> MIDI
```

当前 token 协议是：

```text
PAD = 0
BOS = 1
SEP = 2
EOS = 3
TIME = 4
NOTE_ON_MELODY = 10
NOTE_OFF_MELODY = 11
NOTE_ON_ACCOMP = 20
NOTE_OFF_ACCOMP = 21
```

NOTE_ON 事件携带：

```text
[NOTE_ON_*, pitch, velocity]
```

NOTE_OFF 事件携带：

```text
[NOTE_OFF_*, pitch]
```

TIME 事件携带：

```text
[TIME, delta_tick]
```

其中：

```text
delta_tick * quantum_ms = 时间推进毫秒数
```

对每个 note：

```text
start_tick = round(start_seconds * 1000 / quantum_ms)
end_tick = max(start_tick + 1, round(end_seconds * 1000 / quantum_ms))
```

再生成 note_on 和 note_off 事件，按 `(tick, note_off_before_note_on, pitch)` 排序，然后转成 delta-time token。

清洗后，每个训练样本编码为：

```text
[BOS] source_melody [SEP] target_accompaniment [EOS]
```

## 6. 为什么要切窗口

整曲 token 太长：

```text
平均约 20966 tokens
最长约 48859 tokens
```

而当前 GPT-2 配置：

```text
max_length = 1024
```

如果直接截断整曲，会导致大部分 target 丢失。于是项目引入窗口构建：

```text
scripts/build_training_windows_v1.py
```

默认窗口：

```text
window_seconds = 8.0
max_length = 1024
```

如果某个 8 秒窗口仍超过 1024 tokens，脚本会二分窗口继续切，直到满足长度，或者低于最小时长后丢弃。

本地窗口化结果：

```text
Processed pieces: 40
Accepted windows: 1530
Rejected windows: 8
Average window length: 573 tokens
Minimum window length: 17 tokens
Maximum window length: 1023 tokens
```

这意味着数据已经真正适配 `max_length=1024` 的 GPT-2 训练。

## 7. GPT-2 条件训练

当前模型是一个轻量 GPT-2 causal LM：

```text
vocab_size = 801
n_positions = 1024
n_embd = 512
n_layer = 6
n_head = 8
dropout = 0.1
params ≈ 19.85M
```

输入序列为：

```text
x = [BOS] + source + [SEP] + target + [EOS]
```

其中：

```text
target_start_index = len(source) + 2
```

训练时：

```python
labels = input_ids.copy()
labels[:target_start_index] = -100
labels[padding_positions] = -100
```

旧版本的 `labels = input_ids.clone()` 会让模型连 source melody 和 PAD 也一起预测，导致训练目标污染。现在只在伴奏 token 上计算 loss。

Causal LM loss 可以写成：

```text
L = - 1/N * sum_{t in target_positions} log P(x_t | x_<t)
```

其中：

```text
target_positions = {t | labels[t] != -100}
```

窗口数据按 `source_piece_id` 切分训练/验证，避免同一首歌的不同窗口同时出现在 train 和 eval。

第一版训练结果：

```text
Train samples: 1306
Eval samples: 224
Runtime: 64s on RTX 4070 Super
train_loss: 2.4991
eval_loss: 2.2972
```

这说明模型确实学到了一些伴奏 token 分布，但 4 epoch 只是 baseline。

## 8. 推理与 MIDI 导出

生成时输入：

```text
prompt = [BOS] + source_melody + [SEP]
```

模型从 `[SEP]` 后开始续写：

```text
generated = model.generate(prompt)
target_tokens = generated[len(prompt):]
```

遇到 `EOS` 则截断。然后分别解码：

```text
source_melody -> melody notes
target_tokens -> accompaniment notes
```

最后：

```text
output_midi = melody_notes + accompaniment_notes
```

当前第一版生成结果已经能被 MuseScore 正常打开，说明工程闭环成立。但音乐质量仍然是 baseline：伴奏偏短，织体不稳定，低音支撑不足，节奏有不规则碎片。

## 9. 已修复的工程问题

旧问题包括：

- 关键数据和模型产物缺失。
- 条件生成没有真正实现。
- padding 被纳入 loss。
- MIDI 编解码没有闭环。
- 配置和代码不一致。
- 没有 CLI、seed、数据验证、按曲目切分和实验记录。

当前修复包括：

- `data/training_windows_v1/dataset_windows_v1.json` 作为真实训练入口。
- `[BOS] melody [SEP] accompaniment [EOS]` 条件格式。
- `target_start_index` 控制 source loss mask。
- padding labels 设为 `-100`。
- `closed_loop_v1.py` 实现 MIDI/token/MIDI 验证。
- `dp_melody_cleaning_v1.py` 实现增强 Skyline + DP 标注。
- `build_training_windows_v1.py` 实现短窗口训练集。
- `train_v2.py` 支持 CLI、early stopping、按曲目切分、metadata 保存。
- `generate_from_scratch.py` 改为条件伴奏生成。

## 10. 仍然存在的问题

增强 Skyline + DP 仍然是启发式弱标注，会在复杂右手织体、内声部旋律、高音装饰、多主旋律等场景出错。

当前 token 是紧凑数字流，工程效率高，但音乐语义不够清晰。长期更好的设计是 compound vocabulary：

```text
BAR
POS_0
PITCH_60
DUR_480
VEL_80
ROLE_ACCOMP
```

当前模型也缺少小节、拍位、调性、和弦、风格标签、织体密度等音乐结构条件，所以生成结果容易节奏漂移、和声不稳。

此外，40 首 MIDI、1530 个窗口只能跑 baseline。真正想要稳定生成，需要更多曲目、更干净的人工标注和更一致的风格来源。

## 11. 下一阶段路线

短期建议：

```text
epochs: 20
batch_size: 8 or 16
监控 eval_loss
保存多组 generation samples
```

生成侧可以尝试：

```text
max_new_tokens = 900
temperature = 0.75
top_k = 30
top_p = 0.90
```

数据侧最重要的是人工修正：

```text
DP weak labels
-> vue-piano 可视化复查
-> approved annotated_notes
-> rebuild windows
-> retrain
```

优先修正：

```text
call-of-silence
in-the-pool
uchiage-hanabi
only-my-railgun
```

表示侧下一版可以引入：

- BAR
- POSITION
- DURATION
- VELOCITY_BUCKET
- CHORD
- STYLE
- DENSITY

目标是让模型从“事件流续写”升级为“音乐结构建模”。

## 12. 总结

Pianalysis 当前最大的进展不是生成质量已经多好，而是工程闭环真正成立了：

```text
MIDI 数据
-> 旋律/伴奏弱标注
-> 可逆 token
-> 可训练窗口
-> 条件 GPT-2
-> 伴奏生成
-> MIDI 导出
```

当前生成质量还只是 baseline，音乐上仍然粗糙；但现在问题已经变得清楚：不是模型完全不会学，而是数据标注、token 表示、音乐结构条件还不够好。

下一阶段的核心不应该是盲目堆模型，而是人工修正标注、更强 token 表示、更长训练和更严格生成后验证。

符号音乐生成里最朴素但最重要的经验是：

> 模型决定拟合能力，数据表达决定音乐上限。

