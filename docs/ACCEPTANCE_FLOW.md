# 验收测试流程（第三方从零执行）

本文是《课题二 测试方案》第 7 章"4）具体测试流程"对应的操作说明。五步，每步一条命令；每一步正常结束再执行下一步。

`scripts/acceptance_report.py`仅用于历史证据缓存分析，不重新处理数据或运行模型，不是本流程的替代入口。它的历史单信号基线与下列固定八信号流程不同，不得混用结果。

新启动的流程默认不设置固定显存占比，按推理需要申请显存，不会预先占满整卡。独立采集脚本仍可显式设置`--gpu-memory-fraction`，默认`0`表示不限制。共享服务器上加载前仍检查目标GPU；须取得资源使用授权，有其他作业时等待或协商，不抢占。已启动的冻结运行保持原参数，不为解除上限中断或重启；采用新代码时另建运行编号。

前提：一台装有 NVIDIA 驱动的 Linux GPU 服务器，已安装 `git` 与 Python 3.10，磁盘可用空间不少于120 GB。六个模型角色实际复用五份独立权重，另须预留环境、缓存、图像和本次生成证据的空间。

```bash
# （1）下载代码并安装环境
git clone https://github.com/hlcccc/MSRC.git msrc && cd msrc
bash scripts/01_setup_envs.sh --execute

# （2）准备并校验模型与数据
bash scripts/02_prepare_assets_data.sh --execute --package /path/to/msrc_assets_20261018.tar.gz
bash scripts/02_prepare_assets_data.sh --execute --download-models --hf-endpoint https://hf-mirror.com
bash scripts/02_prepare_assets_data.sh --check

# （3）指标 2.1 / 2.2：重新生成回答、采集八类信号、拟合校准器
bash scripts/03_collect_factual.sh --run-id thirdparty_001 --gpus 0 --execute

# （4）指标 2.3：生成前护栏打分、冻结阈值、测试集判定
bash scripts/04_score_safety.sh --run-id thirdparty_001 --gpus 0 --execute

# （5）汇总指标并生成测试报告
bash scripts/05_score_report.sh --run-id thirdparty_001 --execute
# 完成后打开： runs/thirdparty_001/report/report.html
```

## 每一步做什么

### 从已下载的原始数据重新处理

已有权重与原始数据的机器可用下列命令替代第（2）步。它不使用旧输入划分、旧回答、旧证据或旧风险分；重新解析官方标注、解码39个图像变体parquet、生成输入JSONL并校验图像。第（3）（4）步再独立重新划分。

```bash
bash scripts/02_prepare_assets_data.sh --execute \
  --sources configs/raw_sources.local.json --local-models configs/models.local.json
bash scripts/02_prepare_assets_data.sh --check
```

`raw_sources.local.json` 的六项路径如下。相对路径相对于此配置文件。

```json
{
  "pope_annotations": "/path/POPE/annotations",
  "pope_images": "/path/COCO/images",
  "textvqa_annotations": "/path/TextVQA_0.5.1_val.json",
  "textvqa_images": "/path/TextVQA/images",
  "mmsafety_parquet": "/path/MM-SafetyBench/data",
  "benign_annotations": "/path/matched_benign.jsonl"
}
```

POPE目录含`coco_pope_random.json`、`coco_pope_popular.json`、`coco_pope_adversarial.json`，图像文件名保留官方COCO名称。安全基准使用13类的TYPO、SD、SD_TYPO三个图像变体（5,040条，不含Text_only），另含自建同构良性样本1,680条及对应图像。自建数据不是公开下载项，须随交付材料提供；良性原始JSONL逐条含`id/question/image/phrase/tier/label/category`，标签为0。

`models.local.json` 是`llava/qwen/internvl/guard/generator/judge`到本机权重目录的映射。`generator`与`qwen`可指向同一份Qwen2.5-VL权重。权重只读复用，不改写原目录；准备阶段记录所有模型文件SHA256至`assets/models.lock.json`，预检再次核验。网络下载也按实际解析出的提交版本下载，并记录文件指纹，不能只凭仓库名认定模型相同。

预检的`environment.json`记录完整`pip freeze`。复用交付环境锁文件时，第（1）步可指定`MSRC_REQUIREMENTS=/path/acceptance-environment.lock.txt bash scripts/01_setup_envs.sh --execute`；仍会检查核心依赖与规定版本一致。源码应采用交付的冻结版本，不能把之后更新的仓库主分支与既有实验结果混用。

数据包可以先解包、后下载模型；未就绪项分别打印`MODELS_PENDING`或`DATA_PENDING`，不冒充完整就绪。最后的`--check`必须通过。新数据与图像使用真实文件和相对路径；`MANIFEST.sha256`覆盖数据、图像与本次保存的原始标注/parquet，不能只校验几份JSONL。交付包中的`model_expectations.json`不含本机模型路径，保存冻结版本与文件指纹；网络下载若与这些指纹不符会停止，不会静默用新权重替换旧实验模型。

交付包还须包含`model_metadata/`与`model_metadata_manifest.json`。复用权重时会自动导出本次实际使用的配置、分词器及聊天模板；这些小文件可能含已有的本地兼容修改，不能假定只下载公开权重即可完全一致。下载后仅覆盖清单批准且SHA256相符的非权重小文件，再逐项校验整套模型；不覆盖已有只读原权重、不修改权重参数。已有模型指纹但尚未导出这些文件时，可执行`.venv/bin/python -m pipeline.prepare_assets export-model-metadata`后随数据包交付。

| 步骤 | 命令 | 自动完成的事 | 成功标志 |
|---|---|---|---|
| （1） | `scripts/01_setup_envs.sh --execute` | 建 `.venv`，按 `requirements-acceptance.txt` 安装固定版本依赖（torch 2.7.1 / transformers 5.13.0 / numpy 2.2.6 等），校验 GPU | `ENV_OK` |
| （2） | `scripts/02_prepare_assets_data.sh --execute --package …` | 解包交付包里的 POPE/TextVQA 原始输入、MM-SafetyBench 全部三个图像变体与自建良性样本，写 `configs/paths.json` | `ASSETS_OK` / `CONFIG_PATHS_OK` |
| （2） | `… --download-models --hf-endpoint …` | 为六个角色下载五份独立权重（生成器与 Qwen 基座复用），包括 `inclusionAI/SingGuard-4b` | 每个独立仓库 `DOWNLOAD …` 后落盘 |
| （3） | `scripts/03_collect_factual.sh --run-id thirdparty_001 --gpus 0 --execute` | 先预检（资产、依赖、显卡占用），再对三个模型 × 两套数据重新生成回答，六格采集完整后计算八类候选信号，逐格在训练集拟合、测试集评估 | `FACTUAL_COMPLETED thirdparty_001` |
| （4） | `scripts/04_score_safety.sh --run-id thirdparty_001 --gpus 0 --execute` | 护栏模型在生成回答前逐条打分，训练集按准确率最大选阈值并冻结，测试集算提前预警准确率，再对实际输出做答案核验（只作补充记录） | `SAFETY_COMPLETED thirdparty_001` |
| （5） | `scripts/05_score_report.sh --run-id thirdparty_001 --execute` | 核验六格全量、拟合范围、逐样本结果、护栏全量推理和测试集门控核验记录，按指定考核线判定 | `RUN_COMPLETED`表示完整，`ACCEPTANCE_PASS`才表示通过 |

## 目录与产物

```
runs/thirdparty_001/
  environment.json          解释器、依赖版本、显卡、资产指纹
  factual/                  protocol.json、inputs/、evidence/、logs/
    report.json             六格的 ECE、提升率、实际参与信号数、AUROC、分箱敏感性
    signal_usage/*.jsonl    逐样本信号名称、取值、是否参与融合、融合权重
    calibrators/*.json      融合校准器参数（只在训练集上拟合）
  safety/
    inputs.jsonl            6,720 条；按官方原始问题 ID 分组后训练/测试各 3,360 条
    fresh_scores.jsonl      护栏模型逐样本风险分与推理起止时间
    operating_point.json    冻结阈值与选择依据
    report.json             测试集准确率、混淆矩阵、按来源分层
    delivery.jsonl          生成前门控后的实际输出与时间顺序
    answer_judgements.jsonl 答案核验结论（补充记录）
  report/report.html        测试报告
  summary.json              判定汇总
```

## 几条必须遵守的口径

- **只用本次生成的数据**。第（3）（4）步不接受历史证据、历史风险分或缓存回放；输出目录必须不存在，存在即报错退出。
- **参数只在训练集上定**。阈值、标准化统计量、缺失填充值、信号可用性判定都只用训练集；测试集只评估一次。
- **读数现场采集**。证据文件由本次推理直接产生，随运行编号归档；`runs/<run-id>/` 可以整体拷贝给委托方复核。
- **不剔除失败样本**。某格缺失或推理异常时补测后统一统计，不得只报告通过的格。
- **换一次运行就换运行编号**。第（3）—（5）步的 `thirdparty_001` 统一替换为新编号，旧结果原样保留。
- **完整不等于达标**。报告命令退出码0＝完整且全部达标；1＝完整但存在未达标指标；2＝缺项、冒烟或记录核验失败，不作通过结论。三种情况均保留报告及原因。完成时口径会独立按4种/20%/85%判定，不复用里程碑通过标记。
- **本轮固定方法**。8个候选信号，128-token上限、4改写视角、3重采样、3核验探针，种子20260920；方法前基线为未校准的`1-平均token概率`，ECE为15个等宽风险箱。完整信号注册表共12类，不代表本轮使用12类；实际参与数按逐样本可用、训练侧非恒定且非零融合权重统计，不直接记为8。不得根据测试结果改基线、换划分、删失败格或调整采样预算。
- **时序证据的范围**。生成前门控及三时间戳来自受控本地输出流程，不能称为已验证线上HTTP服务或流式产品接口。
- **共享GPU不抢占**。每次加载模型前检查指定显卡；有其他作业时打印进程属主与PID并等待，不换用未获准的显卡，不终止其他进程。整轮按指定GPU顺序执行。

## 被测模型与辅助回答的隔离

八个候选信号固定为`self_consistency`、`resample_consistency`、`typed_verification`、`cross_model_agreement`、`grounding_check`、`object_support`、`sequence_confidence`、`output_entropy`。

每一格由新的采集子进程加载冻结基础模型，按同一原始图像和问题生成主回答；默认不加载第二模型，也不将标签、其他模型回答或校准器结果送入主回答提示词。主流程的辅助关系固定为LLaVA←Qwen、Qwen←LLaVA、InternVL←Qwen。

辅助回答在六格采集完成后的CPU评估阶段只读接入，只比较同题主回答文本。该过程不更新基础模型权重、不改写任何一方的原始回答。每格的融合系数、标准化统计量与缺失填充值分别仅由该格训练集拟合，不在模型或数据集间共享。

同一份Qwen主回答同时用于Qwen被测格和LLaVA的辅助信号，因此两格共享部分证据、结果可能相关；这不等于先用LLaVA的测试结果训练了Qwen，也不能声称执行了两次物理独立的Qwen推理。`tests/test_role_isolation.py`用CPU合成样本检查输入隔离、只读接入及改变评估顺序不改变结果；这些回归测试不是GPU性能达标证据。
