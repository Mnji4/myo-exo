# Myo Exo 接手文档

这份文档给接手当前 2D MyoAssist/MJWarp 肌肉控制实验的人看。它不是论文综述，也不是完整历史记录，只说明现在这份仓库能做什么、怎么跑、观测/动作/奖励是什么、参考轨迹怎么来的，以及目前已知的问题。

## 1. 当前状态

当前主线是：

```text
2D MyoAssist 人体模型 -> 22 个 muscle actuator -> SAC 训练 -> 先学参考步态/上坡/平-上-平
```

这不是外骨骼控制器。当前策略直接输出人体肌肉激活，不输出关节 torque，也不输出 exoskeleton torque。

仓库是精简上传版，只包含必要代码、小 reference 和配置：

- `cleanrl/`: 训练环境和 SAC/PPO 风格入口。当前主要用 `sac_muscle_mjwarp.py`。
- `configs/`: 可跑的训练配置。重点是 9 度上坡和 flat-up-flat。
- `scripts/`: 参考轨迹构造、课程配置生成、checkpoint 渲染、协同分析。
- `data/camargo_transition_references/`: 小体积 Camargo 转换参考。
- `reference_exports/`: 已筛过的小 reference、review 视频元数据、协同数据。

没有上传：

- `results/`
- `results_old/`
- checkpoint
- TensorBoard logs
- 大量训练视频
- MuJoCo/MyoAssist XML 模型本体

外部模型默认路径是：

```text
/home/lzn/myoassist/models/22muscle_2D/myoLeg22_2D_BASELINE.xml
```

如果机器上没有这个文件，需要改配置里的 `model.source_xml`。

## 2. 最小运行入口

推荐先从两个配置族看：

```text
configs/stageG_uphill9_footlocked035_clip_staged/
configs/stageG_flat_up_flat9_staged/
```

单个配置训练：

```bash
python cleanrl/sac_muscle_mjwarp.py \
  --config configs/stageG_flat_up_flat9_staged/muscle_2d_mjwarp_stageG_fuf9_A_h48_islands_imit_sac.json
```

按 manifest 跑课程：

```bash
python scripts/run_stageG_staged_pipeline.py \
  --manifest configs/stageG_flat_up_flat9_staged/manifest.json
```

渲染 checkpoint：

```bash
python scripts/render_sac_checkpoint_videos.py \
  --config configs/stageG_flat_up_flat9_staged/muscle_2d_mjwarp_stageG_fuf9_F_h192_full_imit_sac.json \
  --checkpoint /path/to/agent_step.pt
```

实际开发时建议先短 horizon、先专项。不要一上来直接长 horizon 混训，之前多次表现为站不稳、学不动或内存炸。

## 3. Action 是什么

策略输出的是 22 维 raw action，对应 MuJoCo XML 里的 22 个 muscle actuator。

代码里不是直接把 raw action 写进 actuator，而是先做：

```text
actor sample/mean -> tanh -> action in [-1, 1] -> clamp -> 0.5 * (action + 1) -> activation in [0, 1]
```

训练环境里最终写入 `ctrl` 的是 22 维 muscle activation。直观理解：

- 0 表示这个肌肉基本不激活。
- 1 表示这个肌肉接近满激活。
- 肌肉系统有延迟和低通效果，所以动作不是 torque motor 那种立刻生效。

这也是为什么上坡/切换地形难：策略需要提前一点激活肌肉，否则脚和身体节奏容易接不上。

当前仓库没有实现 muscle synergy latent 或 action chunk。`reference_exports/synergy_datasets/` 和 `reference_exports/synergy_analysis/` 只是为后续找协同关系留下的数据和分析结果。

## 4. Obs 是什么

Obs 由 `cleanrl/ppo_muscle_mjwarp.py` 里的 `build_policy_obs_tensor` 构造。主要包含这些部分：

```text
qpos, qvel, act,
ref_q - q,
ref_dq - dq,
phase sin/cos,
foot features,
optional reference_valid,
optional frame stack,
future reference window,
terrain preview
```

说人话：

- `qpos/qvel`: 当前身体姿态和速度。
- `act`: 当前肌肉 activation 状态，不只是上一帧 action。
- `ref_q - q`: 当前姿态离参考姿态差多少。
- `ref_dq - dq`: 当前速度离参考速度差多少。
- `phase sin/cos`: 参考轨迹相位编码。但一些最新配置里 `phase_obs` 是 `none`，会把它置零，避免策略死记绝对相位。
- `foot features`: 脚相对骨盆的位置、脚离地高度、脚下地面坡度、当前接触和参考接触。
- `frame_stack_prev_steps`: 把前几帧关键状态拼进 obs，当前 staged 配置一般是 2 帧历史。
- `future reference window`: 给未来几步参考姿态/脚位置，让策略知道下一步脚应该去哪。
- `terrain preview`: 采样前方地形高度，flat-up-flat 配置里常见 32 个 preview 点。

重要原则：策略不应该依赖世界绝对 x 坐标。原因是同一段动作可能出现在低处平地、高处平地、坡上不同位置。如果 obs 里有可利用的绝对坐标，策略容易学成“第几米该做什么”，换位置就崩。

当前代码里有两层相关处理：

- `localize_root` 可以把 root x 清零，并把 pelvis y 改成相对地形高度。
- 最新地形配置还补了脚相对地面、脚下坡度、contact 和历史帧，让策略不需要靠绝对位置猜地形。

如果以后发现“低处平地会走，高处平地不会走”，第一优先检查 obs 里是否还有绝对高度/绝对 x 泄漏，第二检查 reference 在那一段是不是本身姿态或速度不连续。

## 5. Reward 是什么

Reward 不是一个单独公式，而是很多 term 乘以配置里的权重后相加。权重在每个 JSON 的 `reward` 和 `reward_schedule` 里。

当前重要 term 可以按功能理解：

- `tracking_qpos_penalty`: 姿态追踪。让 qpos 接近参考 qpos。
- `tracking_qvel_penalty`: 速度追踪。让 qvel 接近参考 qvel。
- `tracking_foot_site_penalty`: 四个脚底 site 的 x/z 位置追踪。
- `tracking_swing_foot_site_penalty`: 摆动腿脚位置追踪，权重更高，主要管抬脚和落脚。
- `tracking_future_foot_site_penalty`: 当前脚位置对未来几步参考脚位置的追踪，帮助提前规划。
- `tracking_swing_limb_penalty`: 摆动腿 hip/knee/ankle 等局部追踪。
- `tracking_pelvis_penalty`: pelvis 高度、tilt、前进速度的局部惩罚。
- `upright`: 躯干接近竖直。之前在 flat-up-flat 里放大过，用来压往后仰。
- `height`: pelvis 离地形太低会罚。
- `fall`: 摔倒终止相关惩罚。
- `activation_l2` / `tracking_energy_penalty`: 肌肉激活不要过大。
- `activation_smooth`: 相邻控制步 activation 不要跳。
- `foot_slip`: stance 脚不要沿地形切向滑。

`foot_slip` 的计算要特别注意。它不是直接看“真实行为看起来有没有滑”，而是：

```text
判断 stance 脚 -> 取该脚这一控制步的世界位移 -> 投影到脚下地形切向 -> 平方惩罚
```

stance 的来源：

- reference 有效时，用 reference 的 `foot_contact_ref`。
- reference 无效或 post-reference 时，用当前脚离地高度阈值判断接触。

这意味着如果 reference 的 contact 本身有问题，`foot_slip` 会惩罚不该惩罚的脚，或者漏掉该惩罚的脚。之前已经遇到过 reference 会滑、真实行为不明显滑的情况，所以不要盲目把 `foot_slip` 权重拉很大。更稳的方向是先保证 reference contact 和地形对齐，再让 slip 成为辅助项。

## 6. Reference 是怎么构造的

当前最重要的 reference 是：

```text
reference_exports/stageG_flat_up_flat_9deg_topfilled_20260624-232908/stageG_flat_up_flat_9deg_myoassist_3d.npz
```

它对应：

```text
低处平地 -> 9 度上坡 -> 高处平地
```

构造脚本主要看：

```text
scripts/build_stageG_flat_up_flat_reference.py
scripts/build_stageG_long_course_reference.py
scripts/create_stageG_flat_up_flat9_staged.py
```

构造逻辑用人话说：

1. 先拿已有 Camargo/MyoAssist 片段：
   - 平地循环 walk。
   - 平地到上坡 transition。
   - 上坡 steady clip。
   - 上坡到高处平地 transition。
2. 把每段转成统一的 q 系列和脚 site 系列。
3. pelvis x 用连续位移排布，pelvis y 按地形高度重新校正。
4. 拼接时不直接硬接，而是找相似姿态帧。
5. 相似度不只看关节角，还看：
   - pelvis 姿态和速度。
   - hip/knee/ankle 姿态和速度。
   - 四个脚 site 的相对 x/z。
   - 左右脚接触状态。
   - 哪只脚在前。
6. 如果左右脚前后关系反了，会加很大代价，避免“前段像迈右脚、后段像迈左脚”的小碎步问题。
7. 过渡处做短窗口线性/平滑混合，减少位置和速度跳变。
8. 最后重新算导数、脚接触和 metadata，保存为 `.npz`。

之前的问题主要来自几类：

- 只按姿态相似接，会分不清左右脚相位。
- 平地素材多、坡上素材少，拼接时应该优先牺牲平地，不要牺牲坡上关键动作。
- 边界如果落在换脚或双脚很近的相位，视觉上可能变成小碎步。
- 高处平地和低处平地看起来应该等价；如果不等价，要查 obs 是否泄漏绝对位置，或 reference 的高处平地段是否速度/步幅不同。

## 7. 当前课程怎么理解

`stageG_uphill9_footlocked035_clip_staged` 是 9 度上坡单地形专项。它适合用来验证坡度、脚离地、contact、frame stack 和 reward 是否基本可学。

`stageG_flat_up_flat9_staged` 是混合课程。它不是一开始就让策略跑完整长轨迹，而是分阶段：

- A: 短 horizon，分散在关键片段上，先学局部动作。
- B/C: 加入 transition 和更长一点的窗口。
- D/E/F: 扩到混合和更长 horizon。
- G h288 曾尝试过，但更容易 OOM，不能默认当稳定路线。

经验判断：

- A/B 阶段不能太短就跳过，至少要能走出几步再进入更复杂阶段。
- 早期 imitation 应该占主导，否则视频上看不到模仿。
- 多地形混训比单独平地或单独上坡难很多，不要只看 reward 数字，要同时看 checkpoint 视频。

## 8. 已知风险

1. 长 horizon 容易 OOM  
   SAC replay buffer 会按 `buffer_size * obs_dim` 存 `obs` 和 `next_obs`。加 frame stack、terrain preview、future reference 后 obs 变大，h288 + 大 buffer 容易炸内存。

2. 自动导视频可能炸 WSL/OpenGL  
   之前训练中自动渲染触发过 MuJoCo/OpenGL framebuffer 问题。需要视频时尽量控制分辨率、步数和频率，必要时串行导出。

3. Reference contact 会影响 reward  
   `foot_slip` 和 swing/stance 相关 reward 会吃 reference contact。如果 reference contact 错，策略会被错误约束。

4. 不要只相信累计 reward  
   中间 checkpoint 可能视频变好，后面又变差。需要保存并看多个阶段视频，尤其看是否往后仰、步幅是否越走越大、transition 是否小碎步。

5. 外部路径没有完全自包含  
   仓库里有小 reference，但没有 `/home/lzn/myoassist` 和部分原始 Camargo 资源。重新构造全部 reference 可能需要本机原始数据。

## 9. 下一步建议

最稳的接手顺序：

1. 先确认环境能加载 XML 和跑 smoke。
2. 先跑 9 度上坡短 horizon，确认能站住并迈步。
3. 看 obs 统计和各项累计 reward，确认 tracking foot、upright、height、fall、foot_slip 的量级正常。
4. 再跑 flat-up-flat A/B/C，不要直接跳长 horizon。
5. 每隔固定 step 导短视频，优先看几个关键 phase，不要一开始导完整长视频。
6. 如果又出现“高处平地不会走”或“坡上半段会、上半段不会”，先查 obs 是否绝对位置泄漏，再查 reference 的 label range 和拼接点。
7. 如果要做 muscle synergy，先用 `reference_exports/synergy_datasets/` 里的平地/上坡策略采样数据分析肌肉相关性，不要直接把 decoder 固定成只会平地的先验。

当前代码还没有真正解决多地形肌肉控制，只是把 reference、obs、reward 和课程整理到了一个能继续迭代的形态。接手时应该优先保持实验可解释：每次只改一个因素，看 reward 分项和视频，不要同时改网络、reward、reference 和课程。
