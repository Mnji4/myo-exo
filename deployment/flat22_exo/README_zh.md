# 平地髋关节 Exo 策略

三套策略均输出 `[右髋, 左髋]` 力矩，单位 Nm，范围 `[-10, 10]`。运行示例：

```bash
python deployment/flat22_exo/inference_example.py
```

## 策略

1. `direct_exo_8frame.pt`：最近 8 帧髋角度和角速度输入 MLP，直接输出左右力矩。输入 32 维，约 7.5 万参数。
2. `target_pd_exo_8frame.pt`：共享 MLP 根据最近 8 帧髋状态和历史助力预测目标角偏移，再用 `Kp=20`、`Kd=0.5` 计算力矩。输入 48 维，约 2.3 万参数。
3. `fourier3_shared.json`：不使用网络。左右腿共享三阶 Fourier 力矩曲线，左腿错开半周期；7 个曲线系数加 1 个周期参数。

前两种调用 `reset(hip_state4)` 后逐帧调用 `step(hip_state4)`。`hip_state4` 顺序为 `[右髋角, 左髋角, 右髋角速度, 左髋角速度]`，单位为 rad 和 rad/s。

周期策略调用 `reset(phase_rad)` 后逐帧调用无参数的 `step()`。当前周期是 `1.065 s`，实际使用时需要在起步时对齐相位；它只适用于当前固定平地步态。

三阶周期策略拟合直接蒸馏策略的全程力矩 RMSE 为 `0.64 Nm`，留出末段为 `0.73 Nm`。这只是离线力矩拟合，尚未进行闭环稳定性和节能验证。
