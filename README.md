# Pianalysis

两阶段钢琴伴奏训练：**旋律 → 自动和声 → 旋律与和声条件下的伴奏织体**。和声基于 [ReaLchords](https://github.com/lukewys/realchords-pytorch)，织体基于 [Compose & Embellish](https://github.com/slSeanWU/Compose_and_Embellish)。

本仓库仅保存当前训练代码、配置和 Colab 入口。模型、原始数据、试听产物、测试与历史材料保存在项目私有备份中。

## 当前结果

| 阶段 | 已验收 checkpoint | 说明 |
| --- | --- | --- |
| 自动和声 | 2488 步 | 仅输入旋律预测和弦 |
| 伴奏织体 | 1900 步 | 首轮在 2700 步早停，选取验证最优模型 |

五段初始试听与后续 24 段新旋律串联试听均获用户整体认可。后者每段八小节、三个固定种子，共 72 次织体生成；未提供逐曲、逐种子评分。当前范围为已知小节起点的 4/4 拍，不代表自动弱起识别、长曲结构或特定演奏家风格已经解决。

## 准备与训练

使用 Python 3.11 和 NVIDIA GPU。CUDA 12.1 环境示例：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-auto-accompaniment.txt
python scripts/fetch_auto_accompaniment_data.py
python scripts/prepare_auto_accompaniment_data.py
python scripts/fetch_auto_models.py
python scripts/verify_auto_training_data.py
python scripts/run_auto_training.py harmony --output-dir model_output/harmony_run1 --max-hours 2
python scripts/run_auto_training.py texture --output-dir model_output/texture_run1 --max-hours 2
```

配置见 [harmony_train.json](examples/auto_accompaniment/harmony_train.json) 和 [texture_train.json](examples/auto_accompaniment/texture_train.json)。续训使用 `--resume <同一运行目录>/last.pt`，保持数据、配置与训练协议一致；运行入口会核验数据并设置进程时长上限。

## Colab Pro

先制作私有训练包，再打开 [Pianalysis_Train.ipynb](colab/Pianalysis_Train.ipynb)：

```bash
python scripts/estimate_auto_training_cost.py
python scripts/package_auto_training.py
```

将 `data/auto_accompaniment/` 下生成的所有 `.partNNN` 分片和对应 `.manifest.json` 上传至 `MyDrive/Pianalysis/inputs/`。Notebook 挂载 Drive 后选择训练阶段与小时上限；日志和 checkpoint 保存在 `MyDrive/Pianalysis/runs/`。保存使用完成标记与 SHA256 校验，保留最近两代 `last` 和一代 `best`。训练结束后需主动断开并删除 Colab 运行时。

已验收模型另存私有 Drive 备份，保留完整训练状态及代码快照，不随 Git 分发。恢复既有 Colab 运行时保留原 `runs/` 布局；直接使用备份 checkpoint 本地续训时，使用其保存的有效训练配置，不能把基础配置当作该次云端运行的最终配置。

## 数据与许可

标准数据池包含 811 首 POP909-CL 和 22,889 个 Hooktheory 片段；织体使用经筛选的四小节 4/4 窗口，仅监督 PIANO 伴奏，原旋律在生成后合并保留。划分按作品组隔离，但上游预训练可能接触过公共语料。

源码许可见 [LICENSE](LICENSE)，上游声明见 [THIRD_PARTY_NOTICES.txt](examples/auto_accompaniment/THIRD_PARTY_NOTICES.txt)。数据来源、版本和限制见 [data_sources.json](examples/auto_accompaniment/data_sources.json)，官方模型来源见 [model_sources.json](examples/auto_accompaniment/model_sources.json)。Hooktheory 数据采用 CC BY-NC-SA 3.0；代码许可不替代数据及作品权利。当前微调模型仅做私有研究备份，尚未公开托管于 Hugging Face。
