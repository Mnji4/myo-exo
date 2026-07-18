# myo_exo_train

这是当前 80 肌肉多地形 SAC 训练代码。

## 代码结构

```text
myo_exo_train/
├── main.py                 # 入口
├── checkpoint.py           # checkpoint 加载、兼容和保存
├── evaluation.py           # 策略评估、视频和诊断 CSV
├── env/
│   ├── model.py            # MuJoCo 模型、控制映射、物理地形
│   ├── reference.py        # 参考轨迹加载和场景对齐
│   ├── observation.py      # 319 维 observation
│   ├── reset.py            # 普通 reset、离线 bank、在线 bank
│   ├── reward.py           # 奖励入口和人体能耗项
│   ├── reward_locomotion.py# 步态、落脚、楼梯等具体奖励
│   └── runner.py           # MJWarp 并行环境和单步仿真
└── rl/
    ├── networks.py         # Actor、Q 网络、左右对称和 Exo head
    ├── replay_buffer.py    # SAC replay buffer
    ├── sac.py              # 一次 SAC 更新及训练调度
    ├── hard_switch.py      # U/S 专家按位置切换及双专家状态
    └── trainer.py          # 组装以上模块并执行训练主循环
```

## 训练流程

1. `main.py` 读取命令行参数，进入 `trainer.py`。
2. 加载模型、单条完整参考轨迹和 checkpoint。
3. `runner.py` 并行执行仿真，生成 observation、reward 和完整 reset 状态。
4. 普通训练把样本交给一个 SAC；MoE 训练按身体位置把样本分别交给 U/S replay buffer。
5. `sac.py` 更新 Actor、两个 Q 网络和温度参数。
6. 定期保存 U/S checkpoint，并由 `evaluation.py` 导出视频和指标。

## 当前约束

- observation 固定为 319 维。
- reference 和 recovery bank 必须包含完整 `qpos/qvel`；bank 还保存肌肉激活等动态状态。
- 地形使用显式 box 几何体，不使用 hfield 参与接触。
- 策略架构只保留 gated-reference SAC、左右对称网络及可选 Exo head。
- 多专家采用按前进位置 hard switch，U/S 可以同时训练。

## 启动

```bash
python -m myo_exo_train.main \
  --config path/to/config.json \
  --reference path/to/reference.npz \
  --resume path/to/checkpoint.pt \
  --outdir results/experiment_name
```

