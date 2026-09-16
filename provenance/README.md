# 来源与校验

`files.json` 记录自动导出的文件：原服务器路径、原文件 SHA256、导出相对路径、导出 SHA256、大小和转换方式。生成的汇总 CSV 标为 generated_summary；文字样例标明取原文件前两组。没有重新对所有模型权重计算哈希。

`bidirectional_scoring.json` 是独立双向评分实验的追加来源清单，与原训练快照分开维护。保留源码、最终人工标注、全部文本分数、数值检查和汇总；不包含音频、权重、缓存或完整日志。`tools/export_bidirectional_scoring.py` 可在原共享服务器重新导出，拒绝覆盖已被编辑的导出文件；运行它不会训练或推送。

`bidirectional_results_verification.json` 保存克隆内的无模型复算结果；它不重新校验未上传的音频。原服务器音频检查证据在实验结果目录内。`bidirectional_snapshot_verification.json` 保存加入新实验后的全仓库轻量检查。所有检查只用本地 CPU，不调用真实模型或 API。

`v5_migration/` 保存 V5 第 1 轮后切换 FastResume 的原始记录。目录名、配置中的绝对路径及旧报告中的外链用于追溯，不代表这些外部文件包含在仓库内。

`verification.json` 是整理结束时的 CPU 检查结果。检查范围为语法、结构、密钥模式、大小、导出哈希和新文档导航，不执行模型推理、训练或 API。测试文件中的唯一 URL 凭据例外是用于测试日志脱敏的固定虚构字符串；实际 API demo 没有导出。

`source_check.json` 核验导出来源在整理期间未变化。用户后续修改仓库源码时，可保留这些记录表示“首次导出基线”，Git commit 再记录后续差异；此时初始 hash 检查提示差异是预期的。

导出工具用于原服务器上按清单重新整理；会拒绝覆盖已被本地修改的导出文件。`tools/summarize_results.py` 可仅用仓库数据重建结果总表。
