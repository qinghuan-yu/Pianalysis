# Pianalysis

输入旋律 MIDI 与带时间的和声计划，生成钢琴伴奏并保留原旋律。项目目前聚焦简单三和弦的配位、节奏、时值和力度；和声计划由外部提供，尚未实现可靠的自动配和声或完整风格化编曲。

## 当前状态

- **V4 是已验收基准**：固定 V3 配位，学习真实伴奏的节奏、时值和相对力度。《花人局》尾奏与新写的《窗边小句》均获得用户正面听审。
- **V5 是可控节奏实验**：扩充到 12 首、195 个和声片段，加入疏密条件，先预测全部起音再预测时值。它改善了部分时值指标，但用户在尾奏对照中仍偏好 V4，尚未替代基准。
- 试听使用采样钢琴；伴奏升八度是显式后处理，不计为模型学到的能力。两例听审也不等于广泛泛化验证。

## 文档

| 文档 | 内容 |
| --- | --- |
| [迭代记录](ITERATION_HISTORY.md) | 从 MIDI 闭环、协议修复到和声条件、V3/V4/V5 的取舍和实测结论 |
| [V5 实现](V5_IMPLEMENTATION.md) | 数据、事件表示、训练、解码、对照指标和运行方法 |

旧阶段说明已合并；完整旧文档可从 Git 提交 `7358001` 查阅。

## 环境与运行

使用 Python 3.11，在项目根目录安装依赖：

```powershell
python -m pip install -r requirements.txt -r requirements-harmony.txt
```

仓库提供代码、标注配置和听审记录。**源 MIDI、模型权重、生成数据与音色库不随 Git 分发**，新克隆不能直接生成音频；需先准备 [V5 实现](V5_IMPLEMENTATION.md) 所列本地资源。旧版 `train_v2.py` 使用 Transformers Trainer，运行该旧流程时还需安装 `accelerate`。

已有本地 V3/V4 权重及钢琴渲染器时，复现新旋律对照：

```powershell
python scripts/setup_piano_renderer.py
python scripts/audition_original_melody_v4.py --out-dir data/original_melody_preview
```

输出目录含模型伴奏、固定分解、单独旋律和 `index.html`。输出目录必须是新目录；新渲染不会自动继承旧音频的听审结论。钢琴准备脚本下载经固定校验值验证的 FluidSynth 和 SoundFont 到本地，不做系统安装。

## 代码入口

| 路径 | 职责 |
| --- | --- |
| [`scripts/build_trusted_harmony.py`](scripts/build_trusted_harmony.py) | 校验源文件与和声标注，构建训练片段 |
| [`scripts/local_voicing.py`](scripts/local_voicing.py) | V3 配位表示、简单织体与 MIDI 导出 |
| [`scripts/rhythm_v4.py`](scripts/rhythm_v4.py)、[`scripts/rhythm_v5.py`](scripts/rhythm_v5.py) | 真实伴奏投影、节奏表示和 V5 解码 |
| [`scripts/train_rhythm_v5.py`](scripts/train_rhythm_v5.py)、[`scripts/generate_rhythm_v5.py`](scripts/generate_rhythm_v5.py) | 训练对照与可控生成 |
| [`scripts/render_piano_preview.py`](scripts/render_piano_preview.py) | 相同采样钢琴与共同增益的试听渲染 |
| [`examples/harmony/`](examples/harmony/) | 来源校验、曲目划分、输入样例和绑定媒体哈希的听审记录 |

`generate_from_scratch.py` 仍是旧 TSD V2 入口，不会自动调用 V4/V5。`MIDI/`、`data/`、`model_output/`、`test/` 和本地回归测试目录 `tests/` 均由 Git 忽略；评估、核验脚本继续保留在 `scripts/`。项目许可见 [LICENSE](LICENSE)。
