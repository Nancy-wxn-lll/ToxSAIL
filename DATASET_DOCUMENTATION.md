## 适用范围与证据边界

本文覆盖当前项目使用的八个毒性数据集。TDC 名称对应 `src/data/admet_benchmark.py`，本地文件由 `src/data/datasets.py` 优先读取。下述来源文献是 TDC 官方列出的上游来源

官方任务入口：[TDC Toxicity](https://tdcommons.ai/single_pred_tasks/tox/)。所有任务所属 PyTDC 类均为 `tdc.single_pred.Tox`。本项目使用本地生成的 split。

## 数据集说明总表

| 项目名称 | TDC 任务名称 | 本地文件 | 当前记录数 | 标签列 | 任务/单位 |
| --- | --- | --- | ---: | --- | --- | --- |
| hERG | `hERG` | `herg.tab` | 655 | `Y` | 二分类，无量纲 |
| AMES | `AMES` | `ames.tab` | 7278 | `Y` | 二分类，无量纲 |
| DILI | `DILI` | `dili.tab` | 475 | `Y` | 二分类，无量纲 |
| LD50 | `LD50_Zhu` | `ld50_zhu.tab` | 7385 | `Y` | 回归；官方标注 `log(1/(mol/kg))` |
| Tox21 | `Tox21` | `tox21.tab` | 7831 | 12 个 assay 列，见下文 | 多任务二分类，无量纲 |
| ClinTox | `ClinTox` | `clintox.tab` | 1478 | `Y` | 单标签二分类，无量纲 |
| Skin-Reaction | `Skin_Reaction` | `skin_reaction.tab` | 404 | `Y` | 二分类，无量纲 |
| Carcinogens | `Carcinogens_Lagunin` | `carcinogens_lagunin.tab` | 280 | `Y` | 二分类，无量纲 |

记录数指 TSV 数据行数，不是去重分子数或有效标签总数。

## 1. hERG

- **TDC 任务**：`Tox(name="hERG")`，不是 `hERG_Karim` 或 `hERG_Central`。
- **原始来源**：Wang S et al. *ADMET Evaluation in Drug Discovery. 16. Predicting hERG Blockers by Combining Multiple Pharmacophores and Machine Learning Approaches*. Molecular Pharmaceutics, 2016. [DOI: 10.1021/acs.molpharmaceut.6b00471](https://doi.org/10.1021/acs.molpharmaceut.6b00471)。
- **标签定义**：`Y=1` 表示 hERG 阻断剂，`Y=0` 表示非阻断剂。
- **单位及限制**：本地标签为无量纲类别，不是 IC50。没有恢复本缓存逐条实验阈值，不能套用其他 hERG 数据集的 10 μM 阈值。

## 2. AMES

- **TDC 任务**：`Tox(name="AMES")`。
- **原始来源**：Xu C et al. *In Silico Prediction of Chemical Ames Mutagenicity*. Journal of Chemical Information and Modeling, 2012. [DOI: 10.1021/ci300400a](https://doi.org/10.1021/ci300400a)。
- **标签定义**：`Y=1` 为 Ames 致突变阳性，`Y=0` 为阴性。
- **单位及限制**：无量纲二分类标签；不是致癌率，也不是人体毒性概率。当前缓存没有完整的菌株、代谢活化条件与剂量字段。

## 3. DILI

- **TDC 任务**：`Tox(name="DILI")`。
- **原始来源**：TDC 指向 FDA National Center for Toxicological Research 汇集的数据及 Xu Y et al. *Deep Learning for Drug-Induced Liver Injury*. Journal of Chemical Information and Modeling, 2015. [DOI: 10.1021/acs.jcim.5b00238](https://doi.org/10.1021/acs.jcim.5b00238)。
- **标签定义**：`Y=1` 为药物诱导肝损伤阳性类别，`Y=0` 为阴性类别。
- **单位及限制**：无量纲；不能解释为具体发生率、临床严重程度或剂量依赖的风险。本地未保留原始判定依据的逐药物说明。

## 4. LD50

- **TDC 任务**：`Tox(name="LD50_Zhu")`。
- **原始来源**：Zhu H et al. *Quantitative Structure–Activity Relationship Modeling of Rat Acute Toxicity by Oral Exposure*. Chemical Research in Toxicology, 2009. [DOI: 10.1021/tx900189p](https://doi.org/10.1021/tx900189p)。端点为大鼠经口急性毒性。
- **标签定义**：连续变量 `Y`，对应半数致死剂量的对数倒数表示。
- **单位**：[TDC ADMET 官方数据总表](https://tdcommons.ai/benchmark/admet_group/overview/)明确列为 `log(1/(mol/kg))`，不能直接写作 mg/kg。官网该表未明确对数底数，本文不补造转换细节。

## 5. Tox21

- **TDC 任务**：`Tox(name="Tox21", label_name=<assay>)`；项目本地将 12 个 assay 合并为多标签表。
- **原始来源**：Tox21 Challenge 的核受体与应激反应通路筛选数据。[TDC 任务说明](https://tdcommons.ai/single_pred_tasks/tox/)；[Tox21 Challenge 原始专题](https://www.frontiersin.org/research-topics/2954/tox21-challenge-to-build-predictive-models-of-nuclear-receptor-and-stress-response-pathways-as-mediated-by-exposure-to-environmental-toxicants-and-drugs/magazine)。
- **标签定义**：每个 assay 独立编码，1 为活性阳性，0 为阴性。通路活性阳性不等价于已证实人体临床毒性。
- **标签列及顺序**：`NR-AR`, `NR-AR-LBD`, `NR-AhR`, `NR-Aromatase`, `NR-ER`, `NR-ER-LBD`, `NR-PPAR-gamma`, `SR-ARE`, `SR-ATAD5`, `SR-HSE`, `SR-MMP`, `SR-p53`。
- **单位/缺失值**：类别标签无量纲；空白为缺失，不得当作阴性，训练和评价应按 assay 屏蔽缺失项。7831 是分子记录数，不是每个 assay 的有效标签数。

## 6. ClinTox

- **TDC 任务**：`Tox(name="ClinTox")`。
- **原始来源**：Gayvert KM, Madhukar NS, Elemento O. *A Data-Driven Approach to Predicting Successes and Failures of Clinical Trials*. Cell Chemical Biology, 2016. [原始全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5074862/)，[DOI: 10.1016/j.chembiol.2016.07.023](https://doi.org/10.1016/j.chembiol.2016.07.023)。来源研究比较获批药物与因毒性而在临床试验中失败的药物。
- **标签定义**：按 TDC 毒性任务定义，1 为临床毒性阳性/因毒性失败，0 为阴性对照。
- **本地重要限制**：当前文件只有一个 `Y`，并非同时包含 `FDA_APPROVED` 和 `CT_TOX` 两个任务。
- **单位**：无量纲二分类标签，不是临床不良事件发生率。

## 7. Skin-Reaction

- **TDC 任务**：`Tox(name="Skin_Reaction")`。
- **原始来源**：Alves VM et al. *Predicting Chemically-Induced Skin Reactions. Part I: QSAR Models of Skin Sensitization and Their Application to Identify Potentially Hazardous Compounds*. Toxicology and Applied Pharmacology, 2015. [DOI: 10.1016/j.taap.2014.12.014](https://doi.org/10.1016/j.taap.2014.12.014)，[原始全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4546933/)。TDC 指向 ICCVAM 相关的局部淋巴结试验资料。
- **标签定义**：`Y=1` 为皮肤致敏阳性，`Y=0` 为阴性。
- **单位及限制**：无量纲类别；端点是 skin sensitization，不能笼统改称皮肤刺激性，也不能将标签当作原始 EC3 浓度。

## 8. Carcinogens

- **TDC 任务**：`Tox(name="Carcinogens_Lagunin")`。
- **原始来源**：Lagunin A et al. *Computer-Aided Prediction of Rodent Carcinogenicity by PASS and CISOC-PSCT*. QSAR & Combinatorial Science, 2009. [DOI: 10.1002/qsar.200860192](https://doi.org/10.1002/qsar.200860192)。原文介绍的外部集合来自 ISS Carcinogens Data Bank 和 Prestwick Chemical Library。
- **标签定义**：`Y=1` 为致癌性阳性类别，`Y=0` 为阴性类别。
- **单位及限制**：无量纲。
