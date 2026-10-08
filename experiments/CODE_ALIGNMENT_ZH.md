# 旧代码、当前论文与新版实现对照

检查日期：2026-10-08。只读检查 AuthorKit27_2；未修改旧工程、论文、原图或旧实验数值。

## 一、结论

AuthorKit27_2 不是空工程：实际 Python 项目在 `code/eara_release/`。`eara_release 2/` 和 `eara_release 3/` 是相同副本；按相对路径与 SHA256 比较，每份 36 个文件一致（排除 .DS_Store、__pycache__）。不能按文件夹后缀理解成三代不同算法。

旧工程不能原样代表当前 WWW 论文。它同时提供拆解式 EARA 和 IRCoT/ReAct＋充分性门控两种不同实现；旧文主实验更接近后者。当前 WWW 稿明确引入依赖子问题、证据约束改写、top-5 全语料检索和独立审核，已改变实验协议。

新版以现有 `experiments/` 框架为基础补强并独立打包，未把旧版改个名字冒充新版，也没有不必要地重写已工作的 2,000 多行实现。

## 二、旧工程到底实现在哪里

以下旧路径均相对于 `AuthorKit27_2/code/eara_release/`。

| 文件/入口 | 实际功能 | 与当前论文的差异 |
|---|---|---|
| `run.py:40` / build_retriever | 汇集本次 benchmark 的 context_passages 建局部索引 | 不是完整的独立 2018 Wikipedia passage corpus；包括基准提供的支持/干扰上下文 |
| `run.py:62` / run_method | 选择 eara、eara4、eara_ircot、tau 扫描及基线 | 多个实验变体硬编码门槛/步数，不能只改 config 就统一所有分支 |
| `methods/eara.py:50` / decompose | 解析编号列表，最多四个子问题 | 没有显式依赖 DAG；不能提供当前 m=6/adaptive 控制 |
| `methods/eara.py:120` / run_eara | 拆解、改写、局部检索、monitor、五次自一致性及多阈值决策 | top-3；不是当前单候选标量 CAC 主方案 |
| `methods/eara_v4.py:72` / run_eara_v4 | ReAct 自主查询，提出答案后充分性打分 | 没有独立拆解模块，不能拿该结果声称验证显式 CRP |
| `methods/eara_v4.py:186` / run_eara_ircot | 替换为 IRCoT 系统提示，调用同一 v4 控制器 | 属于门控宿主变体，不是依赖规划实现 |
| `core/retriever.py:75` | rank_bm25 的局部检索；Wikipedia API 为另一后端 | BM25 实现和语料均不同；WebDetective 分支直接抛 NotImplementedError |
| `evaluation/metrics.py:19` | 以 gold 是否出现在生成文本中判断正确 | 不是标准 EM，可能误接收否定答案或长段含答案文本 |
| `core/llm.py` | 网关、缓存、工具调用与凭据池 | 不沿用按键轮换处理限流的逻辑；新版采取路由冷却和停止派发 |
| `data/datasets.py` | 多种数据集加载器 | 加载函数存在不等于冻结、去重后的五数据集正式评测已准备好 |

旧论文 `paper/eara_overleaf/eara_submission.tex:291` 本身说明实验使用轻量 ReAct 检索＋同模型评审，而非单独工程化拆解模块。旧 README 的主要复现入口为 `eara_ircot_tau03`。因此，图中的一般架构、eara.py 和旧主实验运行入口不能当作同一个实现。

## 三、已实际复现的问题

使用 `audit_legacy.py`，将旧控制器的模型调用替换为固定返回值；没有读取凭据配置或发送网络请求。原始配置文件仅参与文件一致性 hash，不输出其内容。

1. 未提出 FINAL ANSWER 时，v4 的末尾分支把 `Still reasoning.` 返回成非拒答结果；没有充分性审核。
2. 设置 `max_rejects=0` 的变体中，充分性为 0.0 的候选仍被直接返回；默认较大上限不一定走到该分支，不能据此断言所有旧实验都触发此问题。
3. 充分性输出 `Step 1: Missing facts. Sufficiency score: 0.0` 被首数字正则解析成 1.0。
4. `is_correct('Not Paris', 'Paris')` 返回 True，说明该指标不是 EM。

静态检查另发现：
- `methods/eara.py` 的校准函数使用 `from .metrics import abstention_auroc`，但没有 `methods/metrics.py`。
- `evaluate` 将二值拒答决定作为 AUROC score，不是论文中通常理解的连续充分性排序；旧曲线需要重新核查原始逐题输出。
- 外层 max_steps 限制的是循环轮数；一轮多个工具调用不等于一个检索，不能直接解释为总检索次数预算。
- `Passage.__repr__` 截断为前 200 字符；传给模型的证据与完整索引段落不同。
- ReAct 主入口被固定为 4 步，而不同 EARA 分支使用其他上限；不能宣称旧比较总检索预算统一。
- 并行运行摘要明确写 per-worker cost not aggregated，因此不能把该摘要当作完整模型成本。

上述是实现和协议风险，不足以推断旧论文所有数值虚假或必然受影响。需要旧原始运行日志才能追溯具体结果。当前 release 未提供与所有论文表格一一对应的逐题实验记录。

## 四、新版实现位置

新版路径相对于独立代码包根目录。

| 内容 | 文件/函数 | 状态 |
|---|---|---|
| CLI | `experiments/eara/__main__.py` | 可执行 |
| CRP 依赖规划、冻结共享计划、调度、改写、动态补问题 | `agents.py` 的 plan/select_subquestion/crp_query/expand_plan | 已实现 |
| CAC 标量支持门控与拒答 | `agents.py` 的 review_tuple/gate/run | 已实现，人工阈值仍待准备 |
| BM25s、向量检索、weighted RRF | `retrieval.py` | BM25s 及融合离线验证；全量 dense 与模型部署待验证 |
| 语料导入、数据规范化、去重和划分 | `data.py` | 已实现；不把 gold contexts 插入索引 |
| 请求并发、预算、模型版本、成本 | `gateway.py` | 已实现；429 不轮换键绕过配额 |
| 调度、续跑、网格、运行指纹 | `runner.py` | 已实现；formal/pilot 隔离 |
| EM/F1、coverage/yield、配对 bootstrap | `metrics.py` | 已实现；不沿用子串“准确率” |
| 独立模型标注、盲审、人审结果、阈值 | `audit.py`、`annotation_ui.html` | 工具已实现；没有代造人工标签 |
| 协议防错与能力清单 | `protocol.py` | 本次新增；防止未实现配置静默失效 |
| 不收费端到端验证 | `validate_framework.py` | 合成数据＋真实 BM25s＋mock API |
| 控制器边界测试 | `test_controller_contract.py` | 本次新增 |

## 五、当前论文对齐状态：不能统称“全部完成”

已实现当前主方法：CRP 显式依赖规划、证据改写、CAC 单候选门控、top-5 主检索、10 次逻辑预算、五英文数据集入口、各类核心网格和人审导入/导出。

**2026-10-08 补齐的四项论文 specifies 但代码缺失项（对照 ch.pdf 逐节核查）：**

1. **§6.1 规划器交叉实验**（"a secondary crossed test varies the decomposition model while holding the answer generator and reviewer fixed"）——新增 grid family `planner_crossed`（`experiments/configs/grid_planner_crossed.json`，4 模型 × 6 档 = 24 条件，generator 固定 main_generator、reviewer 固定 frontier），与同模型 decomposition 网格分开标记。
2. **§6.2 改写消融的 CAC 关闭对照**（"CAC is first held fixed, then disabled in a separate control"）——新增 grid family `rewriting_control`（`grid_rewriting_control.json`，4 模型 × D/W 四组合 × cac=False = 16 条件，id 后缀 `_noCac`）。
3. **§3.2 硬上限终止单独报告**——`metrics.summarize()` 新增 `hard_limit_terminations`（retrieval_budget_exhausted / wall_clock_limit / controller_turn_limit / generation_token_limit 四类）与 `abstention_reasons` 全量分解，不再与主动拒答混计。
4. **§3.3 Eq.(3)(4) 组件增益与交互**——`metrics.transfer_effects()` 实现 ΔC=U10−U00、ΔA=U01−U00、I_C,A=U11−U10−U01+U00，CLI 新命令 `transfer-effects --u00/--u10/--u01/--u11`；四条件必须覆盖相同 question/seed 集合、全部跑完才计算，未定义指标（None）显式传播不作零处理。

仍未完整实现的可编码扩展：五样本自一致性、中文查询翻译/CCMOR 集成、证据扰动执行器、人工改写质量统计、评审器非劣效和样本量工具。新版对非 1 候选数直接报错，不会忽略配置假装做了实验。S2G-RAG/ChainRAG 属于后续研究升级建议，也未伪装成已实现基线。

外部依赖与待定身份：完整 Wikipedia/索引、五数据集实际文件、Qwen3-8B/4B 和训练型基线端点、冻结模型版本、人工标注、正式校准；TIR、R1-base/instruct 的确切身份仍缺失。现有 API Kimi-k3 与论文 Kimi-k2.5 不一致；正式模式保持阻止，不能直接改标签。

基线忠实性：Self-Ask/ReAct/IRCoT 等目前为统一 JSON 接口重实现；CRAG 是语料受限适配；Search-o1 是固定骨干适配；Search-R1/ZeroSearch 是独立权重的工具协议接入。均需官方实现对照，不能据代码能跑就宣称完整复现原论文性能。训练型基线的原始训练流程不在本代码包内。

## 六、本次修复与验证边界

- 新增校验：不支持的候选采样数、无效阈值、格式重试配置、子问题数会明确报错。
- paper 主表禁止改成 top-3、其他逻辑检索预算或偷偷使用 hybrid；后端消融须有明确 family。
- 2026-10-08：新增 `planner_crossed` / `rewriting_control` 网格族与 `transfer-effects` 命令；`summarize()` 输出硬上限终止与拒答原因分解；`protocol.capabilities()` 同步更新清单。全部离线验证重跑通过（validate_framework.py、test_controller_contract.py 17/17、transfer_effects 数值与守卫单测、CLI 冒烟）。
- adaptive 提示使用实际配置上限，不再固定写成六个。
- 全部测试无收费 API；当前打包不包含真实实验结果。
- 原论文及原图表保持不变；本代码包不是 Overleaf 项目，也不是已经完成匿名审查的投稿附件。

## 七、建议使用方式

不要继续从三个旧 release 随机选一个跑新论文。以新版代码包为单一运行入口，先执行离线回归，再准备真实数据和模型，最后冻结开发集阈值。原旧工程仅保留作溯源。

本次交付的是现有可执行核心及所有配套源码，不是“未知基线、尚未集成的扩展、模型部署与人工研究全部完成”的承诺。无需作者提供信息的核心迁移和安全边界已处理；未完成项通过能力清单公开。
