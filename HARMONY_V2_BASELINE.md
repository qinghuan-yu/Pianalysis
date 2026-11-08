# 和声标注与柱式和弦基线

本轮先完成「明确和声计划 → 简单柱式和弦」的可检查闭环。已有 TSD 工作已保存为本地提交 `c25973b`。本轮没有训练新的织体模型，也没有把自动标注声明为音乐人验收的真值。

## 当前首先试听的版本

用户指定 `test1.mid` 的和声应为 **D 大调 4–5–6–3**。当前按 **G–A–Bm–F♯m** 实现：每小节一个和弦，八小节重复两遍。旋律不变，低音使用根音；上方选择移动较小的紧凑配位。

| 小节 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 和弦 | G | A | Bm | F♯m | G | A | Bm | F♯m |

和声顺序来自用户；「一小节一个和弦」仍是待试听确认的时间假设。状态为 `progression_confirmed_timing_pending`，没有标成全部通过。之前的 A/B 提案保留为历史例子，已从默认试听页移除。

- [当前 4563 音频](data/harmony_v2_review/review/test1_4563.wav)
- [当前 4563 MIDI](data/harmony_v2_review/block_midis/test1_4563.mid)
- [完整试听页](data/harmony_v2_review/review/index.html)
- [可编辑和声配置](examples/harmony/test1_d_major_4563.json)

试听音频是固定的柔和加法合成音色，便于判断音程与换和弦时间；不是钢琴采样音源。MIDI 可用自己的钢琴音源播放。

## 数据标注范围

| 内容 | 本轮结果 |
| --- | --- |
| 数据 | 原有 40 首 DP 分声部标注 + 4 首 POP909 样本 |
| 音符 | 125,926 个，原始旋律/伴奏角色保留 |
| 和声片段 | 22,479 段，包含可识别和弦、明确无和弦与未知 |
| 音符节奏位置 | 起止 tick、MIDI 网格中的拍/小节索引、小节内四分音符偏移 |
| 时间轴 | PPQ、速度变化、拍号、拍点、小节起点、原 MIDI 调号 |
| 和声信息 | 原符号、根音、性质、音级集合、独立低音、起止 tick、来源与复核状态 |
| 调性 | 原 MIDI 调号 + 时长/力度加权的调性候选；候选相关系数不是正确概率 |
| POP909 额外标注 | chord_midi 的实际起止时间、beat_midi 的原始列、key_audio 的原始音频时间段 |
| 乐句结构 | 明确为 `not_annotated`，不把固定时间段冒充乐句 |
| 优先复核 | 9,525 段，原因包括稀疏音级证据、覆盖不足、未知和声、元数据冲突 |

自动规则用于初标，不能保证艺术和声选择正确。原曲组的和声来自真实伴奏，属于参考条件，不代表仅输入旋律时的和声预测能力。POP909 标注同样是提取结果，尚未逐段人工核对。

原 40 首 MIDI 都触发了 pretty_midi 的跨轨元数据警告。逐事件比较后，39 首只有与轨道 0 一致的重复事件；`only-my-railgun` 有 16 个跨轨调号不一致事件，需要复核。没有发现这批非零轨事件与轨道 0 的速度/拍号冲突。本轮保留轨道 0 的解释，没有偷偷改调号或移动音符时间。

所有新标注写入独立目录；原 DP 标注、训练集与 checkpoint 均保留。`ready_for_texture_training=false`，本轮没有把新标签接入训练。

## 采用的上游

- [chorder 0.1.4](https://github.com/joshuachang2311/chorder)：对参考伴奏做按四分音符网格的和弦识别，并考虑相邻两拍证据。本项目把输出映射到标准符号，保存证据覆盖度和待审原因。
- [mir_eval 0.8.2](https://github.com/mir-evaluation/mir_eval)：解析 Harte 和弦符号。例如 `F#:maj7/5` 的低音是 C♯，斜杠后的 `5` 是相对音级，不是 MIDI 音高。
- [POP909](https://github.com/music-x-lab/POP909-Dataset)：使用 001–004 的真实 `chord_midi.txt`、`beat_midi.txt`、`key_audio.txt`，下载来源与 Git blob SHA 记录在 `data/pop909_sample/sidecar_manifest.json`。保留原有数据许可。

依赖固定在 `requirements-harmony.txt`；本机安装于项目内 `.deps/harmony`，不会修改全局 Python 的这些包。

## 数据与渲染契约

`harmony_plan_v2` 使用原 MIDI PPQ 的整数 tick 和半开区间 `[start_tick, end_tick)`。校验器拒绝重叠、越界和符号/音级/低音不一致。切窗显式裁切并平移时间，记录 `origin_tick`。

`N` 表示无和弦，`X` 表示未知；缺少 sidecar 覆盖的时间填为 `X`。导出时二者都保持安静。超出 MIDI 结尾的 sidecar 片段被裁切并记录原因。

柱式和弦不经过 GPT 或原有生成清理器。每个标签起点同时奏出低音与该和弦的全部音级，在标签结束时停止；根音以外的指定低音也保留。自动初标通常每拍一个片段，因此重复和弦会按拍重奏。

导出后重新读取 MIDI，检查旋律的音高、力度和起止 tick，以及柱式和弦的音级集合、最低音与起止时间。速度使用整数微秒写出，避免浮点截断造成微小漂移。原旋律的同音高重叠使用独立 MIDI channel，防止后一个 note-off 提前关闭前一个长音。保留速度、拍号与已解析的调号；不尝试还原原编配踏板。

## 验证与试听集

全量测试 **51 项通过**：原有 32 项 + 本轮 19 项。

TDD 记录：

| 目标行为 | 观察到的 RED | GREEN |
| --- | --- | --- |
| 标准符号、时间轴、裁切、校验 | 对应入口尚未实现，测试抛出 `NotImplementedError` | 使用 mir_eval 解析、MIDI 时间映射和明确区间验证 |
| 柱式和弦、真实数据标注、试听输出 | 对应入口尚未实现 | chorder 标注、确定性配位、MIDI/WAV 输出与重新读取检查 |
| 保持原速度 | `437956` 微秒被写成 `437955`，断言失败 | 使用 mido 按整数微秒写出 |
| 保持重叠旋律长音 | 原应 3 秒结束的音变成 2 秒，断言失败 | 同音高重叠分配不同 MIDI channel |
| 区分重复与冲突元数据 | 分类入口尚未实现 | 对照轨道 0 的绝对 tick 和事件值 |
| 保留用户指定 4563 与来源 | 用户条件装配入口尚未实现 | 配置中的调性、来源、复核状态进入输出计划 |

开发阶段的 RED/GREEN 入口为 `python -m unittest discover -s tests -p test_harmony_plan.py -v`；最终整合验证为 `python -m unittest discover -s tests -v`。没有额外的无关重构。

当前试听页包含 14 段：用户指定的 test1 4563、已知 C–F–G7–C 终止式（含转位）、4 首留出曲目各 2 个窗口、4 首 POP909 各 1 段。真实曲目另提供同一合成音色的原编配参考。以 `review/cases.json` 为当前试听清单；目录中可能保留早期 A/B 产物。

程序检查证明 MIDI 正确执行了计划。和声计划本身是否合适、换和弦时机是否正确，仍由试听与标注复核判断。试听页支持记录具体秒数并下载反馈 JSON。

## 复现

```powershell
python -m pip install -r requirements.txt -r requirements-harmony.txt
python -m unittest discover -s tests -v
python scripts/build_harmony_review.py
```

默认命令复建当前项目的试听包，需要已有 DP 标注、原 MIDI、V1.1 参考窗口、POP909 四首 MIDI 与 sidecar，以及哈希匹配的 `test/test1.mid`。可以通过 `--manual-plans` 选择另一份和声配置；配置中的事件时间单位是四分音符，不是秒。

主要产物位于 `data/harmony_v2_review/`：`annotated_notes/`、`plans/`、`block_midis/`、`review/`、`priority_review.csv`、`summary.json` 和 `manifest.json`。该目录是本地生成物，受 Git 忽略；代码、依赖、示例计划与本文档进入版本管理。

下一步以用户指定的 4563 为准确认时间位置，然后修订复核集中的和声与低音，再进行条件织体模型的对照训练。
