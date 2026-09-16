> GitHub 归档：此文保留运行时结论。音频及试听 ZIP 未上传；服务器命令和原路径仅用于追溯。仓库内无模型复算入口见 [实验首页](../../../../README.md)。

# 双向评分报告 · pilot30

评分来源：A `v1`；B `v1`。

## A

已评分 25 / 30；严格第一比例：36.0%。

胜负平：{'win': 51, 'loss': 24, 'tie': 0}；并列第一：0。

状态：{'human_uncertain': 5, 'scored': 25}。

## B

已评分 30 组；有可比较人工偏好的 23 组。

人工最佳命中：56.5%；对应随机基线：33.7%。

人工结论：{'preferred': 23, 'all_bad': 3, 'uncertain': 4}；模型平局 0 组。

四条并列最佳 0 组；人工确有排序区别 23 组。全组与有排序区别子集分别报告，不能把前者的全命中当作排序能力。

有排序区别子集：命中 56.5%；随机基线 33.7%。

仅比较当前样本。分组 bootstrap 区间不包含训练随机性或全部数据来源相关性；少量试运行用于验证执行链路。

## 数值检查

{'a': {'sample_id': 'A001', 'repeat_max_abs': 0.0, 'padding_max_abs': 0.0, 'first_frame_sequential_max_abs': 1.5735626220703125e-05, 'future_prefix_max_abs': 0.0, 'masked_padding_content_max_abs': 0.0, 'first_frame_sub_same_hidden_max_abs': 1.1444091796875e-05, 'repeat_tolerance': 1e-05, 'alternate_kernel_tolerance': 0.0001, 'codec_target_sha256': 'd9b12608dcec6ec221050af070e66b5cd4bcba657a7bef3eb52a0e631f751a8e', 'environment': {'sdpa_backend': 'math', 'model_dtype': 'float32', 'torch': '2.11.0+cu126', 'cuda_runtime': '12.6', 'gpu_name': 'NVIDIA H100 80GB HBM3', 'tf32_matmul': False, 'tf32_cudnn': False, 'bf16_reduced_precision_reduction': False, 'math_sdp_low_precision_reduction': False}, 'differences': {'padding': {'target_tokens': 1136, 'max_abs': 0.0, 'mean_abs': 0.0, 'score_delta': 0.0, 'worst_flat_index': 0}, 'first_frame_sequential': {'target_tokens': 16, 'max_abs': 1.5735626220703125e-05, 'mean_abs': 5.185604095458984e-06, 'score_delta': 5.960464477539062e-07, 'worst_flat_index': 3}}, 'note': 'Fixed SDPA math backend; BF16 alternate tolerance 0.05, FP32 1e-4. Same loaded weights; codec encoder not promoted.', 'passed': True}, 'b': {'sample_id': 'B001', 'blind_id': 'A', 'repeat_max_abs': 0.0, 'padding_max_abs': 0.0, 'prefix_only_max_abs': 0.0, 'masked_padding_content_max_abs': 0.0, 'target_tokens': 6, 'repeat_tolerance': 1e-05, 'alternate_kernel_tolerance': 0.05, 'environment': {'sdpa_backend': 'math', 'model_dtype': 'bfloat16', 'torch': '2.6.0+cu124', 'cuda_runtime': '12.4', 'gpu_name': 'NVIDIA H100 80GB HBM3', 'tf32_matmul': False, 'tf32_cudnn': False, 'bf16_reduced_precision_reduction': False, 'math_sdp_low_precision_reduction': False}, 'differences': {'padding': {'target_tokens': 6, 'max_abs': 0.0, 'mean_abs': 0.0, 'score_delta': 0.0, 'worst_flat_index': 0}, 'prefix_only': {'target_tokens': 6, 'max_abs': 0.0, 'mean_abs': 0.0, 'score_delta': 0.0, 'worst_flat_index': 0}}, 'passed': True}}

逐 token 检查明细见 numeric_audit.json 或 summary.json。

平局容差敏感性：[{'epsilon': 1e-05, 'a_strict_top1_rate': 0.36, 'a_tied_top1_count': 0, 'b_agreement': 0.5652173913043478, 'b_model_tie_groups': 0}, {'epsilon': 0.0001, 'a_strict_top1_rate': 0.32, 'a_tied_top1_count': 2, 'b_agreement': 0.5652173913043478, 'b_model_tie_groups': 0}, {'epsilon': 0.001, 'a_strict_top1_rate': 0.28, 'a_tied_top1_count': 4, 'b_agreement': 0.5652173913043478, 'b_model_tie_groups': 0}]

## 边界

- 本次不训练；双向一致不等于正确。
- 官方标签和生成指令不替代当前人工核验。
- A 来自历史反复评测集；B 复用历史训练候选。
- B 随机基线按每组人工最佳集合计算；全部不合格与无法判断单列。
- 主要评分为全部目标 token 均值；A 不含 EOS，不是波形概率密度。
