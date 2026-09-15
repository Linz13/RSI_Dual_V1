# 打标代码

这里保存从原部署目录导出的打标适配器与专家调用代码，模型权重和第三方环境单独部署。

**训练时的属性分工以各框架代码为准：**

- V5：[MODEL_FIELDS 与评审逻辑](../frameworks/v5/dual_isl_train/labeling.py)。
- V6：[MODEL_FIELDS 与固定模型](../frameworks/v6/dual_isl_train/labeling_v6.py)。

[SCHEME.md](SCHEME.md) 描述迁移时的原始多模型一致性打标方案，包含后续 V5/V6 已调整的分工，不能直接当作当前训练 reward 的方案。

`reference/` 保留旧打标 prompt、历史汇总和少量示例；`environment/` 保留原环境记录和依赖约束。它们说明实验背景，不代表在当前机器重新验证过全部环境。

`source/Experiment/labeling2/pipeline.py` 提供 V5/V6 复用的后端入口；`source/Experiment/acc_model_pool/Caption_Bench/` 保留 Qwen/Gemini 请求实现。API key 通过环境变量配置，仓库中没有原私有 API 示例脚本。
