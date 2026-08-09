# myo-exo

基于 MuJoCo/MJWarp 的肌肉骨骼强化学习项目，研究人体在平地、斜坡和楼梯上的行走，以及髋关节外骨骼助力。目前主要使用 SAC，支持 22/80 肌肉模型、参考动作跟踪、完整状态 bank reset、多地形专家和 Exo 条件适应。

## 代码位置

- 网络结构：[`myo_exo_train/rl/networks.py`](myo_exo_train/rl/networks.py)
- SAC 更新：[`myo_exo_train/rl/sac.py`](myo_exo_train/rl/sac.py)
- 训练主循环：[`myo_exo_train/rl/trainer.py`](myo_exo_train/rl/trainer.py)
- 环境、观测和奖励：[`myo_exo_train/env/`](myo_exo_train/env/)
- checkpoint 加载与兼容：[`myo_exo_train/checkpoint.py`](myo_exo_train/checkpoint.py)
- 训练超参数：[`configs/`](configs/)

入口：

```bash
python -m myo_exo_train.main \
  --config path/to/config.json \
  --reference path/to/reference.npz \
  --resume path/to/checkpoint.pt \
  --outdir results/experiment_name
```

## 低激活与 Exo

完整流程是：导出成功轨迹的完整状态，建立肌肉 activation 到关节力矩的局部模型，在下一帧 activation 真实可达的约束内优化低激活和低抖动 Exo 控制，再蒸馏回可部署策略。

求解使用 CasADi 建模、IPOPT 非线性优化和 MUMPS 线性方程求解。主要入口位于：

- [`scripts/build_flat22_muscle_torque_target.py`](scripts/build_flat22_muscle_torque_target.py)
- [`scripts/optimize_course22_hard_constrained_exo.py`](scripts/optimize_course22_hard_constrained_exo.py)
- [`scripts/build_course22_exo_distillation_dataset.py`](scripts/build_course22_exo_distillation_dataset.py)
- [`scripts/evaluate_course22_horizon_residual_mjwarp.py`](scripts/evaluate_course22_horizon_residual_mjwarp.py)

## 平地 Exo 策略

[`deployment/flat22_exo/`](deployment/flat22_exo/) 提供三套可直接读取的策略及具体参数：

- 8 帧髋运动直接预测左右力矩的蒸馏网络；
- 网络预测目标角偏移、固定 PD 输出力矩；
- 无网络的左右共享三阶周期力矩曲线，共 8 个参数。

训练 checkpoint、视频、日志和 recovery bank 保存在本地，不进入 Git；这里只提交部署所需的两份权重和周期参数。
