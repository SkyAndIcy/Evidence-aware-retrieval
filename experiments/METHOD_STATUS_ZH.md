# 方法实现状态

这里的“实现”指有可执行控制流程，不代表已在正式数据上完成实验、达到原论文性能或通过完整复现审查。

| 方法 | 当前实现与限制 |
|---|---|
| EARA | 子问题 DAG、依赖调度、状态更新、证据约束改写、adaptive 上限 6、CAC 阈值判定、预算耗尽/格式失败拒答。 |
| Direct / CoT | 无检索 JSON 答案接口；CoT 使用先推理再给最终答案的提示，不把内部推理日志当作评测标签。 |
| RAG | 单次原问题检索和答案生成；移植 CAC 后可在统一预算内继续检索。 |
| Self-Ask / ReAct / ReAct+abstain / IRCoT | 分别采用 follow-up question、reason/action、显式拒答、交替推理步骤/检索的 JSON 接口重实现；需要与官方实现/提示做复现对照。 |
| CRAG-local | 固定 Wikipedia 内的相关性筛选与纠错查询适配；不是原论文包含外部 Web 搜索的完整 CRAG，不能不加说明直接声称“原版 CRAG”。 |
| Search-o1 | 固定骨干下的原生 search-query/search-result 标签控制与独立 reason-in-documents 步骤；最终答案接口改为统一 answer 标签，属于适配实现。 |
| Search-R1 / ZeroSearch-base / ZeroSearch-instruct | 独立 checkpoint endpoint 的 search/answer 标签工具回路；不能使用 main_generator 代替这些训练后的模型。实际权重、chat template 和 stop behavior 尚须部署验证。 |
| R1-base / R1-instruct | 独立模型入口已有；名称对应哪个训练 checkpoint 仍未明确，正式运行会阻止未确认映射。 |
| TIR | 用户尚未明确论文/实现身份；网格保留条目并明确阻止执行，未伪造同名方法。 |

## 网格

- 主比较：16 行（含未确认方法）；不是 16 个已完成结果。
- 拆解数：4 个 API 骨干 × 6 档，共 24 个条件。
- 改写：4 个骨干 × D0W0/D0W1/D1W0/D1W1，共 16 个条件；固定拆解条件默认 m=3，W 配对使用同一个初始 plan cache。
- 检索：0:10 至 10:0 共 11 档，两端只执行非零权重检索器。
- 组件移植：10 个 host × 4 个 CRP/CAC 组合，共 40 行；TIR 对应四行保持阻塞。

## 不夸大实现范围

当前主方案每次仅生成一个候选。论文中另提的五样本 self-consistency 敏感性实验、中文查询适配、人工改写质量的完整统计、以及 missing-link 等受控证据扰动实验，尚未在本版接入，不能填入相应结果。模型评审非劣效结论也不会因高 κ 自动生成。

初始规划被冻结后共享给配对条件。记录分别包含实际 API 成本和冻结规划的原始成本，复用规划仍计入控制器/token 预算；比较成本时应披露一次性规划开销与缓存状态。正式 paper 模式要求先冻结初始规划，避免一侧付规划成本而另一侧免费复用。
