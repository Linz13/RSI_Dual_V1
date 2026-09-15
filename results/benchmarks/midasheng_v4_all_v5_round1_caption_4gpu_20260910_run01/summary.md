# MiDasheng V4 / V5 Captioner 评测

轮次为从 1 开始的训练轮次；r0 即第一轮。缺失指标保持缺失。

| 模型 | 训练轮次 | ET SPIDEr | ET FENSE | ParaSpeechCaps | StyleCap |
|---|---:|---:|---:|---:|---:|
| midasheng_rewardv4_r0 | 1 | 0.054659 | 0.872867 | 0.227051 | 0.406812 |
| midasheng_rewardv4_r1 | 2 | 0.054821 | 0.875179 | 0.229067 | 0.407776 |
| midasheng_rewardv4_r2 | 3 | 0.054293 | 0.871691 | 0.238723 | 0.403278 |
| midasheng_rewardv4_r3 | 4 | 0.054381 | 0.868282 | 0.248635 | 0.404884 |
| midasheng_rewardv4_r4 | 5 | 0.053629 | 0.866081 | 0.249797 | 0.406812 |
| midasheng_rewardv4_r5 | 6 | 0.053898 | 0.866846 | 0.263192 | 0.404242 |
| midasheng_rewardv4_r6 | 7 | 0.054005 | 0.865887 | 0.271829 | 0.407134 |
| midasheng_rewardv5_r0 | 1 | 0.055535 | 0.875418 | 0.231022 | 0.404563 |

EmotionTalk 沿用 standard_public 口径；完整指标及不可用原因见各任务报告，不把缺失值记为零。
不同训练轮数不属于等预算对照；V4 第一轮与 V5 第一轮可作同轮次比较。
