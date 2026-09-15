# V3 训练可观测性与阶段耗时审计

日期：2026-09-08。只读核验 V3 已完成十轮与当前 V4 实现；未执行 GPU profiling、安装依赖、改训练代码/配置或运行看板。

180 个完成阶段的 elapsed_seconds 合计：平均每轮 253.53 分钟，范围 243.01–278.30 分钟。包含 worker 启动/模型加载/执行/退出；不是纯 kernel 时间，也不是精确 commit-to-commit 墙钟时间。未涵盖独立的 run 初始化及未计时 CPU bookkeeping。

| 阶段 | 平均分钟/轮 | 占计时阶段合计 |
|---|---:|---:|
| caption_tts_rollout | 83.92 | 33.1% |
| tts_grpo | 67.03 | 26.4% |
| caption_grpo | 38.90 | 15.3% |
| audio_caption_rollout | 35.20 | 13.9% |
| caption_caption_reconstruction | 3.55 | 1.4% |
| caption_anchor_reconstruction | 3.16 | 1.2% |
| sftcal_caption_current | 2.88 | 1.1% |
| caption_sft | 2.73 | 1.1% |
| sftcal_caption_anchor | 2.61 | 1.0% |
| tts_sft | 2.11 | 0.8% |
| sftcal_tts_current | 1.88 | 0.7% |
| audio_tts_reconstruction | 1.82 | 0.7% |
| sftcal_tts_anchor | 1.57 | 0.6% |
| audio_tts_anchor_reconstruction | 1.51 | 0.6% |
| caption_reload | 1.37 | 0.5% |
| tts_reload | 1.21 | 0.5% |
| caption_audio_quality | 1.17 | 0.5% |
| audio_caption_quality | 0.92 | 0.4% |

## 已有指标与缺口

- V4 telemetry.py 的 MetricLogger 已写 metrics.jsonl，并尝试导入 TensorBoard SummaryWriter；任意异常会静默回退。当前共享 midasheng-captioner 与 qwen3-tts 环境 importlib 查询均无 tensorboard，V3 全部轮次 checkpoint 中没有 events.out.tfevents.* 文件。不能仅启动 TensorBoard 就看到现有 JSONL，需要安装并启用未来写入或外部转换旧日志。
- Captioner 两阶段：training_metrics/rank_NNN/metrics.jsonl，含 loss、grad_norm、step、sample_id、padded 等。loss 是跨 rank 汇总值，不能把各 rank 当作独立 loss 再重复计数，也不能按本 rank sample_id 把该全局 loss 直接划成 anchor/cycle loss。分来源曲线需补本地分子/分母等日志。
- TTS 两阶段：training_metrics/metrics.jsonl；TTS GRPO training_progress/rank_NNN.jsonl 另含时间、候选 KL、帧数、显存和步骤心跳。阶段 checkpoint 的 dual_isl_train_training.json 保存 epoch_mean_kl、mean_loss、参数同步等。
- 每轮 summary.json/rewards 包含真实训练量、语义/格式 GRPO 组数、非零 advantage 组数、入选数量；V4 新增旧 gate 诊断选择与实际选择的区分。
- 当前没有统一 round/stage/global-step 曲线；尚缺完整 rollout 子阶段计时、连续 GPU 利用率、统一吞吐率、逐步 KL/clip fraction/advantage 分布与固定独立验证曲线。

r9 示例：Captioner GRPO mean_loss=-5.43344e-5、epoch_mean_kl=0.000241334、max_grad_norm=0.0968604；TTS GRPO mean_loss=-1.26121e-5、epoch_mean_kl=0.00118125、max_grad_norm=2.11496。组内中心化 advantage 可使 policy loss 接近零但仍有梯度，不能据此判断未训练。

## 优化判断与边界

- 四个主阶段（TTS rollout、TTS GRPO、Captioner GRPO、Captioner rollout）约占 88.7%；SFT 合计平均仅约 4.84 分钟，旧 gate 校准四个模型阶段合计约 8.93 分钟。V4 循环 SFT 增多后的时间尚未知。
- rollout 内每卡逐候选执行，且每条生成后另做 policy 与 reference replay；TTS replay 包括逐帧/逐 codebook 循环，Captioner 使用原生逐 token replay。rollout 阶段不能误称为只有生成耗时。
- 优先候选：用短 profiling 区分生成、policy/ref replay、音频处理/解码/写盘、同步等待；核实轮初 policy/reference 参数和计算模式等价后复用同一 replay 结果；优化单卡批量推理、只读前端输入复用和推理分片负载。均未实施，也未测得提速。
- r9 rollout PyTorch max_memory_allocated：Captioner 各卡约 17.66–18.01 GiB，TTS 约 4.62–5.24 GiB；这不等于 nvidia-smi 总显存或 GPU 利用率，只说明批量推理值得评估，不能盲改训练 batch。
- 原实现已用 TTS GRPO 长度分桶与逐候选反向；不要作为新优化重复宣称。Captioner GRPO 每 16 token detach KV cache，直接改 chunk 或改成全序列 teacher forcing 可能改变梯度语义，不是无损性能开关。
- 8 卡已用于当前阶段的数据并行；把两个阶段硬拆成 4+4 不保证更快。常驻模型减少加载成本可研究，但已有阶段计时不足以分离纯加载占比。
- 样本数、四候选、epoch、最大长度、reward/anchor 计算的削减会改变实验，应和工程加速分开。

建议先建立只读指标汇总/看板，再对主要阶段做短 profiling；没有证据承诺从 4 小时压到某个固定时长。

来源：[完整阶段时间与文件哈希](stage_times.csv)；代码：V4 dual_isl_train/telemetry.py、workers/qwen3_captioner.py、workers/qwen_voice_design.py、scripts/midasheng_captioner_candidate.py、distributed.py。
