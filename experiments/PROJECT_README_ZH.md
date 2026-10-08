# EARA WWW2027：独立实验代码包

本包是当前可执行核心的完整源码交付，不是正式实验结果，也不是论文全部扩展能力已完成的声明。原 AuthorKit27_2 与论文图表未修改。详细对照见 `experiments/CODE_ALIGNMENT_ZH.md`；机器可读状态见 `CAPABILITIES.json`。

## 安装与离线验收

建议 Python 3.12、macOS/Linux。Windows 用户使用 WSL；进程锁使用 fcntl。无需 APPID 即可执行以下验收：

```sh
cd /你的路径/EARA_WWW2027_Code_latest
python3.12 -m venv .venv
.venv/bin/python -m pip install -r experiments/requirements.txt
.venv/bin/python -m experiments.eara capabilities
.venv/bin/python -m unittest experiments.test_controller_contract -v
.venv/bin/python experiments/validate_framework.py
```

验证使用虚构问题、真实小型 BM25s 索引和 MockTransport；不会调用付费模型。报告写入 `tmp/experiments/`，不应填入论文表格。

## 源码入口

```text
experiments/eara/__main__.py       命令行
experiments/eara/agents.py         CRP、CAC 和各基线适配
experiments/eara/retrieval.py      BM25s、dense、weighted RRF
experiments/eara/data.py           数据规范化、独立语料和冻结划分
experiments/eara/gateway.py        模型网关、20 路上限、预算和日志
experiments/eara/runner.py         网格、续跑、隔离正式与调试运行
experiments/eara/audit.py          模型预标注、盲审、校准
experiments/eara/metrics.py        指标和配对比较
experiments/eara/protocol.py       明确拒绝不支持的协议设置
experiments/configs/paper.yaml    当前运行配置
experiments/annotation_ui.html    人工标注界面
```

## 真实模型与数据

```sh
export FRIDAY_CONFIG="/你自己的凭据配置.yaml"
.venv/bin/python -m experiments.eara doctor
```

凭据配置中的 `models` 按别名映射至 `base_url`、`model`、`api_key`；可参照 `credentials.example.yaml`。凭据不在本包中，不要把真实值写入公开代码或命令参数。doctor 检查配置/文件，不发起模型请求，也不能证明端点真的可用。

本地模型配置使用 `EARA_GENERATOR_BASE_URL`、`EARA_GENERATOR_REVISION`、`EARA_SMALL_BASE_URL`、`EARA_SMALL_REVISION` 等环境变量。部署说明见 `experiments/README_ZH.md` 和 `serve_qwen.sh`。训练型基线必须使用对应训练后的 checkpoint，不可用同一个普通聊天模型改名替代。

语料与数据准备见 `experiments/DATA_SOURCES.md`。全部数据路径默认相对本项目根目录，与作者原计算机目录无关。BM25s 为主比较；dense 另安装 `experiments/requirements-dense.txt` 并取得权重。

正式执行前须完成：模型身份冻结、真实全量语料、五数据集划分、独立人审、开发阈值选择及闭环复核。`--mode paper` 保留必要的阻止条件，禁止为出结果跳过。

## 运行保护

- `run` 未给 `--confirm-api-use` 只输出计划，不调用模型。
- 20 为全局在途模型请求上限，不是每个 APPID 各 20。
- 请求总数另有预算；限流时停止该路由，不用换键绕过限流。
- 外部网络/模型失败与方法主动拒答分开；日志保留重试成本。
- candidate_samples=5 尚未实现，将明确失败，不会默默当作 1。

## 已知未完成项

中文扩展、五样本敏感性、证据扰动执行器、人工改写质量统计、非劣效统计尚未集成；TIR/R1 身份待确认；S2G-RAG/ChainRAG 尚待实现；各基线需要正式忠实性审核。完整源码不等于论文所有实验已完成。

## 打包核验

`SOURCE_MANIFEST.json` 给出本包每个交付文件的 SHA256（不含清单自身）。ZIP 与文件夹交付源码按此清单核对。运行产生的缓存、索引、凭据、模型权重和实验日志不进入源码包。

本包不含旧模型权重/训练代码，不包含真实人工标签，也未替换 Overleaf ZIP。`validation/` 中是离线验收与旧代码问题复现记录。
