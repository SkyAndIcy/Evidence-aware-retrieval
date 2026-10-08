# 数据来源与导入边界

本文件记录数据获取入口；不把基准自带的 gold/supporting context 作为检索库。

| 数据 | 官方入口 / 文件 | 导入说明 |
|---|---|---|
| DPR 2018 Wikipedia passages | https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz | TSV/GZip，列 id/text/title。使用 `download-corpus`、`import-corpus`、`index-bm25`。 |
| HotpotQA | https://hotpotqa.github.io/ | 使用公开有答案的开发数据作为采样池；导入 question/answer/ID，不导入 context/supporting_facts。 |
| 2WikiMultihopQA | https://github.com/Alab-NII/2wikimultihop | 使用官方有答案的数据文件；如要纳入实体别名，先依据官方 id_aliases.json 把别名合入 answer_aliases，保留处理记录。 |
| MuSiQue-Ans | https://github.com/StonyBrookNLP/musique | 官方 `download_data.sh` 指向数据压缩包；不执行下载来的脚本。导入 JSONL；排除明确标记 answerable=false 的条目。 |
| Bamboogle | https://github.com/ofirpress/self-ask/blob/main/datasets/bamboogle.md | 作者提供公开电子表格，可导出 TSV/CSV，保留 Question/Answer 列；校验是否仍为计划的 125 题。 |
| MoreHopQA | https://github.com/Alab-NII/morehopqa | 官方数据在 datasets/files/morehopqa_final.json；另有作者的 https://huggingface.co/datasets/alabnii/morehopqa 。使用扩展问题，而非旧的原问题。 |

## MoreHopQA 衍生题来源映射

为了阻止与 HotpotQA/2Wiki/MuSiQue 原题的泄漏，导入时传入一份审核过的 JSON 映射：键为源样本 ID，值为来源问题的 dataset-qualified ID 列表。例如 `{"sample-id": ["hotpotqa:source-id"]}`。确实没有可映射父题的样本应显式记录空列表，而不是默认缺失视为无关联。

当前采用保守规则：父题在其他主体数据池中出现时排除该衍生题，同时进行规范化问题文本去重。去重后如果不足 1,000 条，程序报错，不补重复样本凑数。此时应在正式运行前修订目标数量，或制定并冻结另一份经审核的分组划分方案；不能运行后为了分数好看改变规则。

## 评分口径

当前实现统一的英文规范化 EM/词级 F1，支持 answer_aliases；它不等同于每个原基准完整官方评分（例如 supporting-facts 或实体 ID 指标）。结果文件记录源码版本。正式报告要说明采用的统一答案匹配规则，必要时另接官方 evaluator 复核。

`normalize-dataset` 不会从 supporting paragraphs 反向扩建检索库，不会向模型传入 reference answers。下载、导入、划分和索引分别记录校验和。原始数据的访问权限、许可证和引用义务仍须遵循各作者发布说明。

## 实现依据

- BM25s 官方项目：https://github.com/xhluca/bm25s
- DPR 官方项目：https://github.com/facebookresearch/DPR
- BGE-M3 dense encoder：https://github.com/FlagOpen/FlagEmbedding/tree/master/research/BGE_M3
- Search-o1 的标签式检索与 reason-in-documents 设计：https://github.com/sunnynexus/Search-o1
- Search-R1 原始 checkpoint/运行方式：https://github.com/PeterGriffinJin/Search-R1

本框架的 JSON 接口基线与 Search-o1 固定骨干适配不是对这些仓库逐行复刻；训练型 checkpoint 的部署和原生提示模板仍需逐一验证。
