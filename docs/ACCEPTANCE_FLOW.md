# 验收测试流程（第三方从零执行）

本文是《课题二 测试方案》第 7 章"4）具体测试流程"对应的操作说明。五步，每步一条命令；每一步正常结束再执行下一步。

新启动的验收流程不再设置45%的进程显存上限，按推理需要申请显存，不会预先占满整卡。独立采集脚本仍可显式设置`--gpu-memory-fraction`，默认`0`表示不限制。共享GPU上须确认空闲资源并取得使用授权；此改动不会改变已启动进程的上限，也不修改既有冻结运行，使用新版本时应另建运行编号。

前提：一台装有 NVIDIA 驱动的 Linux GPU 服务器，已安装 `git` 与 Python 3.10，磁盘可用空间不少于 120 GB（六个检查点约 70 GB，证据与报告约 30 GB）。

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

| 步骤 | 命令 | 自动完成的事 | 成功标志 |
|---|---|---|---|
| （1） | `scripts/01_setup_envs.sh --execute` | 建 `.venv`，按 `requirements-acceptance.txt` 安装固定版本依赖（torch 2.7.1 / transformers 5.13.0 / numpy 2.2.6 等），校验 GPU | `ENV_OK` |
| （2） | `scripts/02_prepare_assets_data.sh --execute --package …` | 解包交付包里的 POPE/TextVQA 原始输入、MM-SafetyBench 精选子集与自建良性样本，写 `configs/paths.json` | `ASSETS_OK` / `CONFIG_PATHS_OK` |
| （2） | `… --download-models --hf-endpoint …` | 按 `configs/paths.example.json` 的仓库名下载六个检查点（三基座 + 护栏 + 生成器 + 答案核验） | 每个仓库 `DOWNLOAD …` 后落盘 |
| （3） | `scripts/03_collect_factual.sh --run-id thirdparty_001 --gpus 0 --execute` | 先预检（资产、依赖、显卡占用），再对三个模型 × 两套数据重新生成回答并现场采集八类信号，最后一个模型一组完成后在训练集拟合、测试集评估 | `FACTUAL_COMPLETED thirdparty_001` |
| （4） | `scripts/04_score_safety.sh --run-id thirdparty_001 --gpus 0 --execute` | 护栏模型在生成回答前逐条打分，训练集按准确率最大选阈值并冻结，测试集算提前预警准确率，再对实际输出做答案核验（只作补充记录） | `SAFETY_COMPLETED thirdparty_001` |
| （5） | `scripts/05_score_report.sh --run-id thirdparty_001 --execute` | 汇总 2.1/2.2/2.3，按里程碑或完成时口径判定，生成 HTML 报告与 `summary.json` | `RUN_COMPLETED thirdparty_001` |

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
