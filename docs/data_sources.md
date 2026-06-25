# 可复用数据和历史资源

## 旧 2D 记录

主记录：

```text
/home/lzn/exoskeleton_terrain/configs/myoassist_2d_multiterrain_mainline/RUN_20260618.md
```

关键结论：

- 48 帧短程可作为参考。
- 96 帧没有走通。
- 失败主要是身体/骨盆跟不上脚。

旧短程 checkpoint：

```text
/home/lzn/myoassist/rl_train/results/train_session_swing_limb_symmetry24_120k/trained_models/best_model.zip
```

这个 checkpoint 是肌肉策略，可作为视频参考、activation prior 来源，或以后做蒸馏数据源。

## MyoAssist 2D 模型

基础肌肉模型：

```text
/home/lzn/myoassist/models/22muscle_2D/myoLeg22_2D_BASELINE.xml
```

这个 XML 只有 22 个 muscle actuator。当前 MJWarp 路线直接使用这些 actuator，不再生成 torque-only XML。

## 楼梯和多地形参考

Camargo selected contact-aligned 参考：

```text
/home/lzn/exoskeleton_terrain/data/camargo_reference_selected_contact_aligned/
```

之前看视频时，比较可用的是：

- 上楼梯：`AB08 stairascent`
- 其他四个：`AB06`

对应视频：

```text
/home/lzn/exoskeleton_terrain/data/camargo_reference_selected_contact_aligned/videos_overview/
/home/lzn/exoskeleton_terrain/data/camargo_reference_selected_contact_aligned/videos_stair_profile/
```

Gait120 五地形参考：

```text
/home/lzn/exoskeleton_terrain/data/gait120_reference/
```

匹配地形参数：

```text
/home/lzn/exoskeleton_terrain/configs/gait120_matched_terrain_params.json
```

## 当前用法

这些 reference 暂时不作为强 imitation 数据。

它们只用于：

- 看人类动作大概长什么样。
- 估计楼梯高度/深度。
- 平地 MJWarp muscle PPO 使用 `short_reference_gait.npz` 做 reference reset 和短程 imitation 先验。
- 后续做离线蒸馏或分地形弱 reference。

当前 MJWarp muscle pilot 默认参考：

```text
/home/lzn/myoassist/rl_train/reference_data/short_reference_gait.npz
```

当前只跟踪 sagittal 2D 相关量：

```text
pelvis_tilt, pelvis_ty
hip_flexion_r, knee_angle_r, ankle_angle_r, mtp_angle_r
hip_flexion_l, knee_angle_l, ankle_angle_l, mtp_angle_l
```

不跟踪绝对 `pelvis_tx`，reset 时把它归零。
