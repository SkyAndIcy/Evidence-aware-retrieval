# 实验可执行性检查（2026-10-06）

> 历史阶段记录：以下内容记录“只有 APPID、尚未编写框架”时的检查。后续已新增实验实现和新版配置；当前使用说明见 `README_ZH.md`，实现边界见 `METHOD_STATUS_ZH.md`。当时的 API 探测数字仍保留为历史记录，不代表持续可用性或正式实验结果。

## 已实际完成

- 共发起 29 次真实 API 探测：23 次成功，6 次 HTTP 429。
- 20 路并发批次实际峰值为 20 个在途请求；该批次 15 次成功、5 次 429。此数值不是 token 吞吐或正式实验并发效率。
- 429 未自动重试；未轮换其他 APPID 绕过同一模型的限流。
- API 探测不计入论文 completed_runs。没有开展检索、回答准确率、消融或人工审核实验。

| 配置模型 | 请求成功/尝试 | 服务返回的 model | 失败情况 |
|---|---:|---|---|
| gpt-4o-mini | 6/6 | gpt-4o-mini-2024-07-18 | 无 |
| qwen-plus-latest | 5/5 | qwen-plus-latest | 无 |
| qwen3.5-baidu | 5/5 | qwen3.5-397b-a17b | 无 |
| kimi-k3 | 0/5 | 未取得 | 429 |
| gpt-4.1 | 1/1 | gpt-4.1-2025-04-14 | 无 |
| gpt-4.1-mini | 1/1 | gpt-4.1-mini-2025-04-14 | 无 |
| aws.claude-haiku-4.5 | 0/1 | 未取得 | 429 |
| gemini-2.5-flash | 1/1 | google/gemini-2.5-flash | 无 |
| LongCat-Flash-Chat | 1/1 | LongCat-Flash-Chat | 无 |
| DeepSeek-V4-Flash | 1/1 | deepseek-flash | 无 |
| DeepSeek-V3.2-Meituan | 1/1 | DeepSeek-V3.2-Meituan | 无 |
| gemini-2.5-flash-lite | 1/1 | google/gemini-2.5-flash-lite | 无 |

其中 GPT-4o-mini 的 6 次包括 1 次串行预检和 5 次并发探测。其余补测模型只测试第一个配置 APPID，不能推断全部 APPID 均有权限或足够配额。

## 与论文协议的差异

| 项目 | 附件旧设置/可用入口 | 论文当前协议 |
|---|---|---|
| 检索 | bm25_local，top-3 | 2018 Wikipedia，BM25s，top-5 |
| 数据 | HotpotQA、2Wiki，各 500 测试/200 开发的旧配置 | 五个英文主体，目标合计 4,125 测试题；HotpotQA 开发 500 |
| 随机种子 | 42 | 17、29、43 |
| 主比较模型 | 附件未提供 Qwen3-8B 入口 | 同一冻结 Qwen3-8B checkpoint |
| Kimi 消融 | kimi-k3 | Kimi-k2.5，需作者明确是否修订 |
| 本地小评审 | 未提供 | Qwen3-4B 及冻结 revision |
| 阈值 | 旧固定阈值，calibrate=false | 仅开发集风险约束校准 |

## 正式实验的必要前置条件

1. 提供已有实验代码仓库/目录，或明确要求按当前论文协议从零实现。当前工作区只有论文与协议，没有可执行的方法实现。
2. 提供 2018 Wikipedia passages、BM25s 索引和数据集所在目录，或提供具备下载/构建空间的计算服务器。配置中的 ../WebDetective 不存在。
3. 提供主比较及训练型基线模型的 checkpoint/运行入口，明确 TIR、R1-base/R1-instruct 的映射；未实现的基线不能用同名 prompt 顶替。
4. 解决 Kimi/Claude 的限流配额，并冻结别名到模型版本的映射。
5. 确认 API 总调用/费用上限和每题 token/时间上限。若 16 方法 × 4,125 题 × 3 seeds 全量交叉，仅主表已为 198,000 个问题-条件运行，尚未计消融与多轮调用。
6. 安排两名独立人工标注者和分歧裁决者，或明确将人工部分保留为尚未完成；模型互评不能替代人工数据。

本机检查：24 GiB 内存，检查时约 88 GiB 可用磁盘。未下载大型语料、未部署模型、未启动后台全量运行。

## 可追溯性

- 当前论文协议 SHA-256：`88b30c214036aee12da1db8d08b2a7a681ae2486e2b53b76489a3d9a9f20e178`。
- runs/preflight.json：单请求预检。
- runs/concurrency20.json：20 个 APPID/四个模型的并发探测。
- runs/remaining_models.json：其他八个模型别名各一次探测。
- probe_gateway.py：实际使用的脱敏探测程序；成功、429、空内容、异常结构、重定向和并发上限均通过 MockTransport 验证。
- 论文 PDF、Overleaf ZIP、表格结果与 completed_runs 均未因探测而改变。
