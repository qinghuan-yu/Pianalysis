# 和声标注与柱式和弦基线

基线阶段完成「明确和声计划 → 简单柱式和弦」的可检查闭环。已有 TSD 工作保存为本地提交 `c25973b`。2026-11-08 用户确认 test1 试听通过；随后完成的和声条件训练对照见 [HARMONY_CONDITIONING_V2.md](HARMONY_CONDITIONING_V2.md)。其余自动标注仍不是人工验收真值。

## 当前首先试听的版本

用户指定 `test1.mid` 是ヨルシカ《花人局》尾奏，和声为 **D 大调 4–5–6–3 = G–A–Bm–F♯m**。最新反馈明确：前六音起音属于弱起，G 在第 6、7 音起音之间进入；第 12 音起音仍在 G 上，A 在第 12、13 音起音之间进入。换和弦可以发生在旋律延音中，不能吸附到下一音起点或上一音终点。

首个和声强拍为四分音符偏移 3（tick 288），随后每四拍换和弦。首六音的起音均早于此拍点，第六音持续越过该拍点。**这些位置先根据用户音间范围推定，后于 2026-11-08 获得用户反馈「我试听没问题。可以继续下一步了」。** 原 MIDI 的时间、拍号及旋律均保留，修正的音乐起点单独存入 `musical_timing`，不把文件零点当音乐小节起点。

| 和弦 | G | A | Bm | F♯m | G | A | Bm | F♯m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 四分音符偏移 | 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 |
| 起点秒数 | 1.385 | 3.231 | 5.077 | 6.923 | 8.769 | 10.615 | 12.462 | 14.308 |

0–1.385 秒的和声未知，标 `X`，试听只保留旋律；这不等于原曲明确没有伴奏。最后的 F♯m 截到片段结尾 14.769 秒，不能据此认定它是原曲终止和弦。整段状态为 `user_audition_accepted`，配置的 `audition_acceptance` 保存原话、日期、试听 WAV 与 MIDI 的 SHA-256。验收只覆盖这一片段的柱式和弦与换和弦时机。旧 A/B、零秒起 4563 均退出当前试听清单。

查阅了 [U-FRET 普通页](https://www.ufret.jp/song.php?data=66978) 与 [视频和弦页](https://www.ufret.jp/song.php?data=83899)（2026-11-08）。后者默认选择 Capo 1，页面中的 F♯、G♯、A♯m、Fm 指型需升半音才对应 G、A、Bm、F♯m。末段也包含不同的和弦顺序；该页面不是与本 MIDI 逐音对应的五线谱，因此只作和弦背景参考，不能据它声称本次推定的拍点已被原谱验证。当前以用户指定的 4563 为试听目标。

- [当前弱起 4563 音频](data/harmony_v2_review/review/test1_pickup_4563.wav)
- [当前弱起 4563 MIDI](data/harmony_v2_review/block_midis/test1_pickup_4563.mid)
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

所有新标注写入独立目录；原 DP 标注、训练集与 checkpoint 均保留。`ready_for_texture_training=false` 表示整批标签尚未人工验收。后续独立试验显式使用 `--allow-unreviewed-harmony`，保留弱监督来源；不会因 test1 通过而把 44 首标为通过。

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

全量测试 **54 项通过**：原有 32 项 + 和声相关 22 项。

TDD 记录：

| 目标行为 | 观察到的 RED | GREEN |
| --- | --- | --- |
| 标准符号、时间轴、裁切、校验 | 对应入口尚未实现，测试抛出 `NotImplementedError` | 使用 mir_eval 解析、MIDI 时间映射和明确区间验证 |
| 柱式和弦、真实数据标注、试听输出 | 对应入口尚未实现 | chorder 标注、确定性配位、MIDI/WAV 输出与重新读取检查 |
| 保持原速度 | `437956` 微秒被写成 `437955`，断言失败 | 使用 mido 按整数微秒写出 |
| 保持重叠旋律长音 | 原应 3 秒结束的音变成 2 秒，断言失败 | 同音高重叠分配不同 MIDI channel |
| 区分重复与冲突元数据 | 分类入口尚未实现 | 对照轨道 0 的绝对 tick 和事件值 |
| 保留用户指定 4563 与来源 | 用户条件装配入口尚未实现 | 配置中的调性、来源、复核状态进入输出计划 |
| 延音中换和弦与音间范围校验 | `timed_events` 未支持，两个新测试报 `KeyError: events` | G 在 tick 288、A 在 tick 672 进入；第 6、12 音保持完整；拒绝范围外拍点与歧义复调编号 |
| 试听保留逐段复核状态 | 新测试报 `KeyError: review_status`，原先会合并相同和弦的不同状态 | 记录每段来源/状态/起音范围，状态不同不合并，表格显示状态 |

开发阶段的 RED/GREEN 入口为 `python -m unittest discover -s tests -p test_harmony_plan.py -v`；最终整合验证为 `python -m unittest discover -s tests -v`。本次 GREEN 为 22 个和声测试通过；无额外重构。配置、资料说明属于非 TDD 修改，通过解析、真实 MIDI 往返与产物审计验证。

当前试听页包含 14 段：用户指定的 test1 4563、已知 C–F–G7–C 终止式（含转位）、4 首留出曲目各 2 个窗口、4 首 POP909 各 1 段。真实曲目另提供同一合成音色的原编配参考。以 `review/cases.json` 为当前试听清单；目录中可能保留早期 A/B 产物。

程序检查证明 MIDI 正确执行了计划。和声计划本身是否合适、换和弦时机是否正确，仍由试听与标注复核判断。试听页支持记录具体秒数并下载反馈 JSON。

## 复现

```powershell
python -m pip install -r requirements.txt -r requirements-harmony.txt
python -m unittest discover -s tests -v
python scripts/build_harmony_review.py
```

默认命令复建当前项目的试听包，需要已有 DP 标注、原 MIDI、V1.1 参考窗口、POP909 四首 MIDI 与 sidecar，以及哈希匹配的 `test/test1.mid`。可以通过 `--manual-plans` 选择另一份和声配置；`timed_events` 中 `start_quarter/end_quarter` 是从文件起点计算的四分音符偏移，不是秒。可选的 `onset_between_notes: [6, 7]` 要求和弦起点严格位于第 6、7 音的起音之间。音符从 1 编号，只接受无交叠单音旋律；缺口填为 `X`，每段单独记录来源和复核状态。旧数组 `events` 仍可用于已有人工计划，不能与 `timed_events` 混用。

主要产物位于 `data/harmony_v2_review/`：`annotated_notes/`、`plans/`、`block_midis/`、`review/`、`priority_review.csv`、`summary.json` 和 `manifest.json`。该目录是本地生成物，受 Git 忽略；代码、依赖、示例计划与本文档进入版本管理。

柱式基线已通过用户试听。下一阶段已完成时间条件接口与一次弱监督训练，结果、局限及新的织体试听入口见 [HARMONY_CONDITIONING_V2.md](HARMONY_CONDITIONING_V2.md)。
