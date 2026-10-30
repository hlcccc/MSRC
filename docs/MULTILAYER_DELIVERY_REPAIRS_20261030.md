# 本次交付修复与验证范围

本次固定版本为`acceptance-multilayer-delivery-20261030`，对应新方案指标1.11—1.13。旧发布标签、原始四版本文档、SingGuard回滚标签、冻结的11个多层算法文件及历史实验结果均保留，不将旧成绩登记为本次新实验。

## 实际修复

1. 有拟合版使用公开参数所属的逐字节冻结采集入口`collect_evidence_fitted.py`；无拟合版保留现用`collect_evidence.py`。参数源码合同仍逐项严格检查，未替换参数、未跳过校验、未在测试侧重新拟合。实际执行入口及哈希记入每次事实性`protocol.json`。参数准备及源码配对检查提前到大规模模型和数据下载前。
2. 环境步骤检查Python 3.10、固定包版本、依赖关系、模型类、视觉算子、真实OCR推理和CUDA运算。缺少Python 3.10/venv时，可安装哈希固定的项目内CPython；缺少OpenCV依赖的Linux动态库时，通过APT下载并解包到项目虚拟环境，不修改全局软件、不调用sudo。分词及OCR后端按实际验收环境补充固定版本。
3. 完整但不达标的联合报告保留结果并以返回码3退出；材料不全或审计失败以非零返回。失败重查时同时重写HTML和JSON为INCOMPLETE，避免残留旧的通过页面。
4. 最终步骤真实生成`signal_report.html`、`calibration_report.html`、`warning_report.html`三个独立报告，以及`signal_summary.csv`和`warning_by_source.csv`。总报告提供链接，`outputs.json`记录文件位置与文件哈希；报告生成时检查承诺引用的证据实际存在。

## 报告位置

两版仅运行编号不同。以下均位于`runs/<运行编号>/`之下：

| 用途 | 位置 | 生成阶段 |
| --- | --- | --- |
| 逐样本信号统计与融合输出 | `factual/signal_usage/<模型>_<数据集>.csv`和`.jsonl`，共六组 | 事实性apply或fit完成 |
| 六组ECE及相对提升 | `factual/summary.csv`及`factual/report.json` | 事实性apply或fit完成 |
| 增强五分支结果 | `multilayer/experiments/multilayer_001/report.json` | 增强测试及答案核验完成 |
| 逐token分数、判定与放行记录 | `multilayer/experiments/multilayer_001/records/test/` | 增强测试 |
| 答案验证原始输出和标签 | `multilayer/experiments/multilayer_001/records/answers/` | 答案核验 |
| 三指标总报告 | `multilayer/joint/report.html` | 最终汇总 |
| 功能演示报告与六组信号数 | `multilayer/joint/signal_report.html`及`signal_summary.csv` | 最终汇总 |
| 校准性能报告 | `multilayer/joint/calibration_report.html` | 最终汇总 |
| 生成中预警报告与四来源统计 | `multilayer/joint/warning_report.html`及`warning_by_source.csv` | 最终汇总 |
| 汇总及完整位置清单 | `multilayer/joint/summary.json`及`outputs.json` | 最终汇总 |

## 已验证及不能扩大的结论

- 重新下载69,150,087字节的公开参数包，核对归档SHA256、包内9个文件及24项实际参数源码指纹。修复后的有拟合采集入口全部匹配；原采集入口与包不一致的检查仍会拒绝。
- 在隔离Linux源码目录运行项目回归测试，612项通过、2项按原有条件跳过；另有130项多层模块测试通过。通过与失败报告使用明确的人工单元测试数据，验证文件生成、链接目标和返回码，不是本次新增实验成绩。
- 在既有验收Python环境实际运行CPU环境检查，19项固定依赖、pip check、视觉算子、三个被测模型类及OCR推理通过。另用公开下载并校验的CPython归档，在隔离目录实际安装私有Python 3.10.19。此项不等于在空机器重新安装全部GPU依赖。
- 本轮未重新执行六组全量GPU采集，也未重新完成两版各3,360条增强测试；不能据此宣称新固定入口已经各自空机全量通过。历史97.6190%仍仅是历史参考，以新运行报告为准。
- CPU诊断只打印`ENV_CPU_OK_NOT_GPU_ACCEPTANCE`，不能当作`ENV_OK`。CUDA检查、模型显存容量及完整推理仍须在实际测试硬件上通过。
- 原方案所列RTX 5090与已实际核验的A100不是同一硬件条件，本轮没有验证5090。当前参考依赖为PyTorch 2.7.1默认CUDA 12.6构建；不兼容时环境检查必须失败，不把该硬件配置写成已经验证。
- 已拟合版冻结采集器不实现后续新增的共享GPU占空比控制，因此不能加载共享调度策略冒充同一流程；使用未加载该共享策略的执行环境。无拟合版的现用调度功能保留。

两份第七章Word保留原参考文档的字体、样式和蓝色正文，仅包含我方三项指标及所需操作。截图标注为历史日志/历史实验示例，不作为新运行成绩。原参考方案文件不覆盖。
