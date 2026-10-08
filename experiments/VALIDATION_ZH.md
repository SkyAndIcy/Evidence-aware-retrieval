# 验证记录（2026-10-06）

这是工程验证，不是论文实验结果。正式实验未完成，不更新论文结果表。

## 可重复的离线检查

在工程根目录安装 requirements.txt 后运行：

```sh
.venv-experiments/bin/python experiments/validate_framework.py
```

测试使用虚构语料；模型请求由 httpx.MockTransport 模拟，不需要凭据、不收费。报告写入 tmp/experiments/framework_validation.json。

已通过：真实 BM25s 建库及 mmap 加载、top-5 与无匹配查询、11 档融合权重、五种数据集归一化不导入 gold context、开发/测试拆分、两跳改写与拒绝后接受、答案泄漏哨兵检查、成功任务断点续跑、配对 bootstrap、独立模型预标注、盲审导出、拒绝把模型标签当人工标签、pilot 禁止读取 test、无效证据 ID 拒绝、模拟峰值 20 请求并发、429 冷却且不轮换密钥。

另通过 Python 编译检查、serve_qwen.sh 的 bash 语法检查、标注界面内嵌 JavaScript 语法检查。尚未完成浏览器交互验收。

MoreHopQA 官方完整 JSON 的字段结构已读取核对：共 1118 条，扩展问题使用 question/answer，父问题使用 previous_question/previous_answer。导入器读取扩展问题，不把父问题当新问题。父题映射仍需单独核对；去重后样本不足时必须调整协议，不能重复采样凑数。

## 已进行的真实 API 联调

运行 ID：09a2ce3917c90858d4ab83b17143deb5edd52fd103d800c5ece78c8567e0be1a。

虚构 Blue Lantern/Leon Moss 问题经过两轮真实 BM25 检索，返回 Lakeside；该次调用 4 次真实 API，复用了已有初始规划缓存。前一次试跑因模型引用错误证据 ID 而拒答，失败记录保留。随后提示明确要求使用原始字符串 ID，未把无效引用强行改成有效引用。

该记录验证的是当时版本的 API 联通与流程；后续功能及成本记账改动通过离线回归，但没有对最终打包版本重新收费联调。单个虚构问题答对不能报告为论文准确率，也不证明本地 Qwen 模型已部署。

## 尚未验证或尚未完成

- 全量 2018 Wikipedia 下载、完整 BM25/dense 索引及资源需求。
- 五个真实数据集的全部导入、父题交叉去重和正式冻结切分。
- Qwen3-8B、Qwen3-4B 和训练基线真实权重部署、版本及 chat template 对齐。
- 全部基线的官方实现一致性；Search-o1、Search-R1 等适配器尚无真实 checkpoint 端到端验证。
- 开发集人工裁决、正式阈值及闭环验证、正式测试与显著性分析。
- TIR 方法身份、中文适配、五样本 self-consistency 和其余文档列明的扩展实验。

仅提供模型预标注及真人标注工具，没有生成任何真实人工标签。
