# myo-exo

这是一个基于 MuJoCo/MJWarp 的肌肉骨骼强化学习项目，当前主要研究 80 块肌肉人体在平地、斜坡和楼梯上的行走，以及髋关节外骨骼助力。

目前使用 SAC 训练人体策略，支持：

- 319 维统一观测，包括人体状态、参考动作、足端状态和前方地形；
- 完整状态 reset 和在线/离线 recovery bank；
- 平地上坡专家、平地上楼专家及按位置硬切换的双专家 MoE；
- 左右对称网络、步态约束和可选外骨骼策略头；
- 自动评估、视频导出和训练指标记录。

## 主要目录

```text
myo_exo_train/      当前训练代码和说明
configs/            实验配置
reference_exports/  参考轨迹
scripts/            数据处理、分析和视频工具
results/            checkpoint、日志和视频
docs/               实验交接与训练记录
cleanrl/            早期训练代码，保留作历史参考
```

## 启动训练

```bash
python -m myo_exo_train.main \
  --config path/to/config.json \
  --reference path/to/reference.npz \
  --resume path/to/checkpoint.pt \
  --outdir results/experiment_name
```

人体骨架、关节、肌肉和执行器由配置中的 `model.source_xml` 定义；参考文件保存每帧的完整姿态和速度，但不包含模型骨架本身。训练代码的具体结构见 [`myo_exo_train/README.md`](myo_exo_train/README.md)。
