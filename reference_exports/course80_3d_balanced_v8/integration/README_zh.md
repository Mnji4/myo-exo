# course80 v8 跨机器接入

## 必需文件

- `../course80_3d_balanced_v8.npz`：动作参考及语义时间线。
- `terrain_course_include.xml`：与参考对齐的可碰撞地形，共 23 个 box geom，不使用 hfield。
- `terrain_contract.json`：坐标轴、完整地形尺寸、模型维度、joint/qpos 地址及文件哈希。

另一台机器还必须已有同版本 `myoLeg80_HMEDI` 及其 mesh。模型没有复制到这里；启动时至少核对 contract 中的依赖哈希以及 `nq=35`、`nv=34`、`nu=82`，不一致时不要训练。

`summary.json`、`validation_report.json` 和 MP4 只用于追溯、自动验收和人工审核，不是训练输入。

## 读取参考

```python
import numpy as np

raw = np.load("course80_3d_balanced_v8.npz", allow_pickle=True)
metadata = raw["metadata"].item()
series = raw["series_data"].item()
qpos = series["qpos_full"]  # (1066, 35)
qvel = series["qvel_full"]  # (1066, 34)
fps = float(metadata["sample_rate"])  # 30 Hz
labels = metadata["label_ranges"]
```

这是一条 1066 帧、约 35.53 秒的非循环长参考。训练端负责课程终点后的 reset；控制频率不是 30 Hz 时，应按时间重采样，并用 MuJoCo 的 quaternion-aware 位置插值/差分处理 root quaternion，不能对四元数直接逐分量线性插值。

## 地形和坐标

模型沿世界 `+y` 前进，`+z` 向上，`+x` 为侧向。metadata 中保留的 `x0/x1` 是历史命名，实际都是世界 `y` 坐标。root qpos 前 7 项依次为：

`lateral_x, forward_y, vertical_z, qw, qx, qy, qz`

地形顺序为：

`低平地 -> 4 m 上坡 -> 高平地 -> 4 m 下坡 -> 低平地 -> 8 级上楼 -> 高平台 -> 8 级下楼 -> 低平地`

- 坡度：高差 `0.64836016 m / 4 m`，约 `9.207 deg`。
- 上楼：踏步高 `0.127 m`、进深 `0.31749953 m`，高平台高度 `1.143 m`。
- 下楼：踏步高 `0.127 m`、进深 `0.29 m`，从 `1.143 m` 高平台下降。
- 地形半宽：`1.25 m`；摩擦参数：`1.0 0.005 0.0001`。

精确边界和值以 `terrain_contract.json` 为准。目标模型当前在顶层 include `models/terrain_config80.xml`；接入时应把这一项替换为 `terrain_course_include.xml`，也可按 contract 生成等价 geom。两种方式都必须发生在 `MjModel` 编译前。训练进度使用 root `y`。

## 接入边界

NPZ metadata 中指向 Camargo 原素材、构造中间件的绝对路径只用于 provenance，运行时不要读取。运行时只依赖完整 `qpos_full/qvel_full`、本机通过校验的 80 肌肉模型和上述地形。

该文件是逐帧运动学参考，不是已经通过动力学的 policy rollout。地形几何可碰撞，但策略能否稳定踩踏仍需在目标 trainer 中跑接触仿真验证。

## 重新生成

仓库默认假定 `myo-exo_upload` 和 `myoassist` 是相邻目录：

```bash
python scripts/generate_course80_terrain.py
```

目录结构不同则显式指定模型；`--outdir` 可用于先在临时目录核对输出：

```bash
python scripts/generate_course80_terrain.py \
  --source-xml /path/to/myoassist/models/80muscle/myoLeg80_HMEDI/myolegs_HMEDI.xml \
  --outdir /tmp/course80-terrain
```
