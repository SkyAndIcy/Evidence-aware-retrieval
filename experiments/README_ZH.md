# EARA 实验框架 v0.1

2026-10-08：已与 AuthorKit27_2 旧实现对照，新增协议参数防错、控制器回归测试及独立源码打包。详见 `CODE_ALIGNMENT_ZH.md`。运行 `python -m experiments.eara capabilities` 可查看实现范围；这不是所有论文扩展实验已经实现的声明。

已实现可运行的 EARA/固定骨干基线框架、新版配置、真实 BM25s 索引、消融网格、断点续跑和独立模型预标注。**代码可运行不等于正式论文实验已完成。** 全量 Wikipedia/基准数据、部署的 checkpoint 和真实人工标签尚未就绪；论文 `completed_runs` 仍为 0。

## 1. 安装与凭据

从包含 `experiments/` 的工程根目录运行：

```sh
uv venv .venv-experiments --python 3.12
uv pip install --python .venv-experiments/bin/python -r experiments/requirements.txt
export FRIDAY_CONFIG="/本机原始接口配置文件路径"
.venv-experiments/bin/python -m experiments.eara doctor
```

框架只从原附件读取 models 的网关/认证信息，**不采用附件中的旧 top-3、500 test、200 dev、seed 42 或固定阈值配置**。不要把 APPID 复制进 YAML、命令参数、论文或 ZIP。HTTPS 是默认要求；仅本机 loopback 允许 HTTP，以便连接本地推理服务或 SSH 隧道。

## 2. 新版论文配置

`configs/paper.yaml` 固定 top-5、每题总计 10 次逻辑检索、五个英文数据集、随机种子 17/29/43、全局最多 20 个在途 API 请求。HotpotQA 开发目标 500，测试目标分别 1000/1000/1000/125/1000。

新增实现默认值：每题 600 秒、64 个控制器调用、16,384 个生成 token；开发集风险目标 5%，95% Wilson 上界约束，至少 30 个接受样本。**这些是运行前配置，不是已经达到的实验结果。** 未取得人工开发标注前不会生成正式阈值；pilot 的 0.5 仅用于调试。

主比較默认 Qwen3-8B，本地小评审 Qwen3-4B；外部评审默认已探测成功的 GPT-4.1 路由。Kimi 配置采用附件可用的 kimi-k3，并标记替代原论文 Kimi-k2.5 的协议变更；未完成版本/论文说明前禁止正式 paper 模式。latest 类型别名没有冻结版本时也会阻止正式运行。服务返回的模型字符串不单独证明底层权重版本。

## 3. 先跑不计入论文的联调样例

仓库内 8 个 passage 和 2 个问题均为**合成虚构数据**，不是 Wikipedia 或基准样本。

```sh
.venv-experiments/bin/python -m experiments.eara --config experiments/configs/smoke.yaml import-corpus --input experiments/fixtures/passages.tsv
.venv-experiments/bin/python -m experiments.eara --config experiments/configs/smoke.yaml index-bm25
.venv-experiments/bin/python -m experiments.eara --config experiments/configs/smoke.yaml freeze-splits
.venv-experiments/bin/python -m experiments.eara --config experiments/configs/smoke.yaml run --method eara --mode pilot --split dev --seeds 17 --limit 1 --confirm-api-use
```

导入、索引和冻结操作拒绝覆盖已存在数据；已准备过时直接运行最后一行。删除或更换目录前保留来源记录。没有 `--confirm-api-use` 时 run 只打印计划，不发起收费请求。

## 4. 准备正式语料与五个数据集

数据入口和别名/父题要求见 `DATA_SOURCES.md`。下载 DPR passage 文件需要显式给出下载大小上限；以下 12 GiB 是下载上限，不是总运行空间估计。

```sh
.venv-experiments/bin/python -m experiments.eara download-corpus --output experiments/data/wikipedia2018/psgs_w100.tsv.gz --max-gib 12
.venv-experiments/bin/python -m experiments.eara import-corpus --input experiments/data/wikipedia2018/psgs_w100.tsv.gz
.venv-experiments/bin/python -m experiments.eara index-bm25 --allow-large-index
.venv-experiments/bin/python -m experiments.eara normalize-dataset --name hotpotqa --input /数据/hotpot_dev_distractor_v1.json --source-url https://hotpotqa.github.io/
```

其余数据集的 `--name` 为 `2wikimultihopqa`、`musique`、`bamboogle`、`morehopqa`。MoreHopQA 还需 `--parent-mapping`。全部导入后运行 `freeze-splits`。未取得足量、可去重样本时不会用重复数据补足目标。

全量 BM25s 建库需要在内存中保留 tokenized corpus；`--allow-large-index` 只代表显式允许大任务，不保证本机资源足够。运行前评估磁盘和内存。索引加载使用 mmap；文档正文保存在 SQLite，不从数据集 gold context 构造检索库。

可选 dense：安装 `requirements-dense.txt`，冻结 BGE-M3 的真实 revision，再执行 `index-dense --allow-model-download --device cuda`。当前 dense 搜索为有界内存的精确分块点积，不是 ANN；大语料检索可能很慢，不能把其时间与另一套 ANN 配置混用。程序会检查 float32 embedding 文件所需磁盘空间。

## 5. Qwen3 与训练型基线入口

提供 `serve_qwen.sh`，作为另一个已安装、验证过 vLLM 的模型服务环境的启动入口。它绑定 loopback；远程服务建议通过 SSH 隧道连接。本 Mac 未安装/部署 vLLM，不能把以下命令视为已验证的 GPU 部署。

```sh
export EARA_GENERATOR_REVISION="真实的不可变模型提交版本"
bash experiments/serve_qwen.sh generator
export EARA_SMALL_REVISION="真实的不可变模型提交版本"
bash experiments/serve_qwen.sh reviewer
```

在实验客户端设置 `EARA_GENERATOR_BASE_URL=http://127.0.0.1:8000/v1` 和 `EARA_SMALL_BASE_URL=http://127.0.0.1:8001/v1`，并填写对应 revision。模型服务如启用认证，再设置相应 API_KEY 环境变量。运行时显式关闭主 Qwen3 模型的 thinking 模式；CoT 通过其条件提示实现，配置与请求记录均保留。

Search-R1/ZeroSearch 必须配置各自 checkpoint 服务，不能把同一普通聊天接口重命名为训练型方法。R1-base/instruct 和 TIR 尚有身份映射待确认。具体实现/适配边界见 `METHOD_STATUS_ZH.md`。

## 6. 运行、20 路并发与续跑

预生成的 `configs/grid_*.json` 覆盖主比较、拆解数、改写、11 档检索比例和每 host 四行迁移组合。也可通过 `make-grid --family ... --output ...` 重新生成。

```sh
.venv-experiments/bin/python -m experiments.eara run --grid experiments/configs/grid_decomposition.json --condition-id gpt4o_mini_m2 --mode pilot --split dev --seeds 17 --limit 20 --confirm-api-use
```

- 同一 run root 只允许一个进程持有运行锁，进程内最多 20 个在途 API 请求；不是每个 APPID 各 20 路。
- 429 触发同一模型路由的冷却并停止继续派发；不自动更换 APPID 绕过配额。其他基础设施故障也停止新任务，保留已运行记录。
- 默认每轮最多授权 1,000 次 API 调用；先小样本确认成本。要继续预算耗尽的**同一份**运行，可以显式加 `--additional-request-budget N`，并在配额/网络恢复后加 `--retry-failures`。每题基础设施最多两次尝试，成功条目不重跑。
- 配置、数据、模型身份和代码版本进入运行指纹。原样执行命令会续跑；修改科学配置或代码会建立新运行，避免污染旧结果。
- `api_ledger.jsonl` 保存所有请求尝试和 usage；未知 usage 不伪造成 0。结果保留前次失败尝试成本；若崩溃使成本记录不完整，聚合报告明确标记缺失。
- API 不支持 seed 时不偷偷添加不支持的参数；记录 `seed_applied=false`。这些只能描述为重复请求，不可声称精确控制了模型随机种子。

正式 `--mode paper --split test` 不允许 limit，要求五个主体数据集、完整指定语料、冻结模型版本、已校准并通过开发集闭环验证的阈值。CRP 配对条件先通过 `freeze-plans` 冻结初始规划。pilot/development 不允许读取 test split。

## 7. 标注流程：模型与人工分开

```sh
.venv-experiments/bin/python -m experiments.eara export-proposals --results /运行/results.jsonl --output /标注/tuples.jsonl
.venv-experiments/bin/python -m experiments.eara label-models --tuples /标注/tuples.jsonl --roles frontier small_reviewer --output /标注/models --confirm-api-use
.venv-experiments/bin/python -m experiments.eara model-agreement --labels /标注/models/machine_labels.jsonl --frontier-threshold 0.5 --small-threshold 0.5 --output /标注/model_agreement.json
```

上面的阈值仅为命令格式示例，正式分析使用各自开发集冻结阈值。两个模型独立读取相同 question/candidate/passages，不互相看到标签、不看到 reference answer。输出永远是 `label_source=model`；可以统计 >=t 数量、混淆矩阵与模型间 κ，但不会据此宣称等同人工或模型能力相同。

`make-audit` 导出已打乱顺序的 A/B 盲审任务和独立 private 映射。真实评审者用浏览器打开 `annotation_ui.html`，分别加载自己的任务文件，独立判断并导出标签；不应把模型预标注复制为人工标签。`audit-stats` 拒绝模型来源标签、同一评审员充当两人或未完成任务；对抽样概率使用权重，保留 uncertain 与退化 κ 情况。

开发集流程：`calibration-data` 整理经人工裁决的数据 → `calibrate` 选阈值 → `run --mode development` 按阈值重跑顺序控制器 → 对新的接受案例独立人工审核 → `validate-operating-point`。通过后才允许 held-out paper 运行。没有合格阈值或足够证据时明确报告失败，不伪造结果。

目前未创建任何真实人工标签；非劣效统计、样本量设计和人工组织工作不因代码存在而自动完成。

## 8. 输出与测试

每次运行包含 `run_manifest.json`、`jobs.sqlite`、`api_ledger.jsonl`、逐题/逐尝试 traces、`results.jsonl`、`summary.json`。准确率/coverage/yield/failure rate 分开报告，零回答时 selective accuracy/risk 为 null。`compare` 要求两个方法拥有相同题目/seed 集合，并按问题聚类 bootstrap，而不把不同 seed 当成独立问题。

验证记录见 `VALIDATION_ZH.md`。正式模型、全量 dense、全部基线复现尚未验证。这个代码包不包含 APPID、模型权重、全量语料或真实人工标签，亦不改变已有论文 PDF/图表数值。
