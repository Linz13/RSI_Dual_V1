# V6 前三轮 benchmark

| 训练轮次 | ET SPIDEr | ET FENSE | ParaSpeechCaps | StyleCap | DSD 中文 | DSD 英文 | DSD 平均 |
|---|---|---|---|---|---|---|---|
| 4 | 0.0553 | 0.8737 | 0.2297 | 0.4026 | 81.7000 | 76.6000 | 79.1500 |

缺失/未完成结果不记为零。保留原 benchmark 的完整属性和提示，不按 V6 四属性裁剪。
DSD 使用与 base 重评分相同的 API 和 Gemini-2.5-Pro；每轮中英文各 1,000 条。
