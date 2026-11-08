# 迭代记录

Pianalysis 的任务始终是保留输入旋律、生成伴奏。实现路线从直接续写 MIDI 事件，逐步收缩为“给定可靠和声，先学配位，再学节奏”。这里的版本包含不同实验协议，不代表越新越好；当前听审基准是 V4。

## 为什么逐步拆分任务

| 阶段 | 解决的问题与实现 | 结果与后续决定 |
| --- | --- | --- |
| V1：闭环与弱标注 | MIDI→音符 JSON→条件 token→MIDI；增强 Skyline + 动态规划分离旋律；整曲切短窗口，只对伴奏计算损失 | 40 首数据跑通流程，但弱标注不是可信声部标签，工程可运行不等于音乐正确 |
| V1.1：协议修复 | 分离控制 token 与参数 ID，修复参数值 `3` 被误读为 EOS；增加版本校验 | 提前结束错误得到修复，NOTE_ON/OFF 闭合与长曲覆盖问题仍在 |
| TSD V2：显式时值 | 用 Pitch/Velocity/Duration 表示完整音符；加入事件语法、时间边界和长旋律切窗 | 同批窗口平均长度从 573.84 降到 443.63 token；解决了表示负担，尚不能保证和声或跨窗乐句质量 |
| 和声条件 V2 | 加入按 tick 对齐的和声计划，区分未知和声与无和弦；先做柱式和弦试听，再训练条件织体 | 用户确认 D 大调 IV–V–vi–iii 与弱起边界；正确条件下复杂织体仍不好，转向更小的局部任务 |
| V3：局部配位 | 24 个经符号证据复核的 POP909 片段；学习当前和弦的一低三高四音配位 | 训练目标是明确标记的规则合成练习；柱式与简单分解通过尾奏听审，节奏仍由规则提供 |
| V4：真实节奏 | 冻结 V3；从 4 首、67 段真实 PIANO 伴奏学习三通道起音、时值和相对力度 | 原留出曲目起音 F1 从固定分解的 0.382 提高到 0.715；用户认为新节奏更适合尾奏，成为听审基准 |
| V5：扩充与可控生成 | 12 首、195 段；对照“原 V4 / 仅扩充数据 / 起音优先且带密度条件的 V5” | 部分时值误差改善，疏密条件可影响输出，但尾奏听审仍偏好 V4；保留 V5 为实验能力 |

TSD V2 与局部 V3/V4/V5 的目标、词表和上下文不同，权重不能混用。旧入口 `generate_from_scratch.py` 默认仍走 TSD V2；局部流程有独立生成脚本。

## 和声标注先于织体评价

尾奏样例来自《花人局》。用户确认 D 大调的 G–A–Bm–F♯m，并指出前六音属于弱起：和弦从第六音到第七音之间进入，后续也在乐句间换和弦。按用户纠正建立时间计划，再以柱式和弦试听确认，避免把错位和声的失败归咎于织体模型。

语料使用 POP909 的命名 MELODY/PIANO 轨、和弦 sidecar 和拍点。源文件哈希、曲目划分、逐段三音和低音证据保存在 [`trusted_subset_v3.json`](examples/harmony/trusted_subset_v3.json)、[`trusted_subset_v4.json`](examples/harmony/trusted_subset_v4.json)、[`trusted_subset_v5.json`](examples/harmony/trusted_subset_v5.json)。`assistant_symbolic_reviewed` 表示符号证据复核，不等于用户听审；和弦内音覆盖率也不是标注准确率。

## 听审改变了哪些决定

| 对照或变化 | 用户反馈 | 当前结论与记录 |
| --- | --- | --- |
| V3 柱式 / 简单分解 | “两版都可接受” | 局部配位基线可继续；[记录](examples/harmony/local_voicing_v3_acceptance.json) |
| V4 / 固定分解 | “新节奏可接受，整体更合适” | 接受该尾奏的学习节奏；[记录](examples/harmony/local_rhythm_v4_acceptance.json) |
| V4 / 扩充版 / V5 中等 | “已验收V4更自然” | 扩充版与 V5 不替换 V4；[记录](examples/harmony/rhythm_v5_audition.json) |
| V4 只删音 / 改部分时值 / 改部分力度 | “我觉得都挺自然的” | 单因素干预未定位不自然感的单一原因；[记录](examples/harmony/naturalness_audition.json) |
| 伴奏整体升八度 | “目前效果听起来不错” | 本例接受 +12 半音，换曲仍检查音区；[记录](examples/harmony/register_v4_audition.json) |
| 换采样钢琴 | “现在生成的质量还是不错的” | 原 MIDI 已是钢琴 program 0，改善来自音频渲染器；[记录](examples/harmony/piano_v4_acceptance.json) |
| 新旋律《窗边小句》 | “说实话，我觉得这挺完美的” | 冻结 V4 在第二个简单案例上通过听审；未明确比较固定分解；[记录](examples/harmony/original_melody_v4_acceptance.json) |

《窗边小句》是先写好旋律与 G–Em–C–D｜Em–C–D–G 和声，再用同一 V3/V4 权重生成一次的八小节短句。没有重训、筛选候选或手工修正预测；两种伴奏统一升八度并使用相同采样钢琴与共同增益。它提供了新输入上的正面证据，但不是复杂编曲或广泛泛化结论。

V5 在尾奏的七个完整和声段中，有六段重复同一起音图案；对等长历史节奏内容的扰动也不敏感。这是机制诊断，不足以确定听感回退的原因。已有局部干预未复现明显不自然感，因此没有按猜测改和声标签或再次重训。

## 上游经验与采用边界

| 来源 | 实际采用 |
| --- | --- |
| [MidiTok TSD / REMI](https://github.com/Natooz/MidiTok) | 借鉴显式时值与事件后继约束；本项目独立实现编码器，没有直接替换为 MidiTok |
| [POP909](https://github.com/music-x-lab/POP909-Dataset/tree/d83e6edba6872a704f5d3b8b32f5cb540088dae6) | 命名声部、和弦、拍点与按原歌曲划分；局部实验使用其中 12 首的选定片段 |
| [Compound Word Transformer](https://github.com/YatingMusic/compound-word-transformer) | 参考字段分解思路；尚未采用其多头复合词架构 |
| [AccoMontage2](https://github.com/billyblu2000/AccoMontage2) | 乐句与和声结构的后续参考；尚未移植其检索或宏观规划 |
| [FluidSynth](https://github.com/FluidSynth/fluidsynth)、[GeneralUser GS](https://github.com/mrbumpy409/GeneralUser-GS) | 固定版本采样钢琴试听；不算模型能力变化 |

## 下一步的判断依据

继续用冻结 V4 和相同音色作为基准，增加小调、不同节奏、不同音区及更长乐句的试听覆盖。V5 优先拆开验证“起音优先”和“密度条件”的各自作用，并检查模型是否真的利用旋律时序；通过听审后再决定是否替换基准。

独立低音节奏、跨和弦延音、扩展和弦、踏板及长程曲式仍未完成。早期关于堆训练轮数或直接做复杂风格迁移的计划，不作为当前执行路线。V5 的具体实现与运行方法见 [V5_IMPLEMENTATION.md](V5_IMPLEMENTATION.md)。
