# course22 长参考

## 这是什么

`course22_v1` 是审核通过的 `course80_3d_balanced_v8` 的 22 肌肉平面模型投影版。
时序和段落边界没有重拼，仍为 1066 帧、30 Hz、35.53 秒：

`平地 -> 9 度上坡 -> 高平地 -> 9 度下坡 -> 平地 -> 上楼 -> 高平台 -> 下楼 -> 平地`

目标模型：

- `models/22muscle_2D/myoLeg22_2D_BASELINE.xml`
- `models/22muscle_2D/myoLeg22_2D_HMEDI.xml`

两个 XML 的 qpos/qvel 拓扑相同，所以同一份参考可以用于两者。HMEDI 版只多两个外骨骼执行器。

## 文件

目录：`reference_exports/course22_v1/`

- `course22_v1.npz`：完整 `qpos_full[1066,53]`、`qvel_full[1066,53]` 和旧加载器需要的标量关节序列。
- `terrain_contract.json`：精确地形段、摩擦和训练器开关。
- `training_config_fragment.json`：可合并进训练 config 的最小配置片段。
- `terrain_course22_include.xml`：reference-only 渲染使用的等价 box 地形。
- `validation_report.json`：逐段足底净空和关节限位检查。
- `videos/course22_v1_side.mp4`：480x272、30 fps、H.264 完整预览。

只复制 NPZ 不够。训练时至少同时复制 `training_config_fragment.json` 或
`terrain_contract.json`，否则动作和地形会错位。

## 映射

- 80 模型世界 `+y` 前进映射为 22 模型 `pelvis_tx / +x`。
- 髋屈伸、踝和 MTP 直接映射。
- 膝关节符号取反，因为两个 XML 的膝屈曲正方向相反。
- 80 模型的侧向根位移、骨盆侧倾/旋转、髋内收/旋转和距下关节被丢弃。
- 22 模型腿长和足底位置不同，不能直接照抄骨盆高度。脚本按慢速支撑侧求足底接触高度，
  再做对称 9 帧平滑；统一前进位置补偿为 `+0.05 m`。

地形不用 hfield。斜坡、高低平台和上下楼都由 box 构成。下楼段的 22 训练器定义与
course80 差一节高度，转换脚本已把 descending `base_height` 减去一个 riser，不能再手动改回。

## 重新生成

```bash
cd /home/lzn/myo-exo_upload
conda run -n myoassist-mjwarp \
  python scripts/build_course22_from_course80.py
```

渲染：

```bash
LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MUJOCO_GL=glfw \
conda run -n myoassist-mjwarp \
  python scripts/render_course22_reference.py
```

渲染器逐帧设置 reference state，不跑 policy 或动力学；RGB 帧直接 pipe 给 FFmpeg 输出 H.264。
当前 WSL 的 MuJoCo EGL 缺少 `PLATFORM_DEVICE`，所以这里用 WSLg + llvmpipe。

## 当前限制

足底接触检查通过，推断支撑脚没有低于地形 3 cm 的帧。原 course80 的足锁 IK 本身允许关节
越过 XML limit；忠实投影后，22 版也有髋伸展和踝背屈超限，详见
`validation_report.json` 的 `tracked_joint_limit_violations`。

这版适合先审核动作和作为 motion prior 数据。若用于严格的 22 模型动力学跟踪，应另做
“限位内重定向”，重新优化髋/踝和足底接触；不要直接逐帧 clip，逐帧裁剪会制造速度平台和新的滑脚。
