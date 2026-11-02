# 课题二快速测试流程

本流程采用“全量历史证据复核＋现场抽测”，不重新训练或拟合。经测试方确认采用后执行；如要求全量重新推理，仍使用原测试流程，不以本版代替。

## 1 取得代码和测试材料

在Linux服务器任意工作目录执行：

```bash
git clone --single-branch \
  --branch codex/fast-review-20261102 \
  https://github.com/hlcccc/MSRC.git MSRC_fast
cd MSRC_fast
mkdir -p delivery
```

将本版交付的 `MSRC-fast-evidence-20261102.tar.gz` 放入 `delivery/`。网盘发布时使用独立目录 `MSRC_fast_review_20261102`，不与原交付包混放；分享地址见随包的《交付说明》。代码也可从同一交付目录下载ZIP解压取得。

材料包含：完整逐样本回答和信号证据、固定划分、已拟合参数、内容安全逐token判定记录、答案核验记录，以及现场抽测图像。全量证据复核不需要下载基础模型和全量原始图像。

## 2 安装环境并复核全量指标（指标1.11、1.12、1.13）

```bash
bash scripts/fast_01_setup.sh
bash scripts/fast_02_review.sh
cat runs/fast_001/review/report.md
```

成功提示：`FAST_REVIEW_ENV_OK`、`EVIDENCE_OK`、`REVIEW_COMPLETED NUMERIC_PASS`。

脚本核验交付材料，重新提取历史证据中的八类信号、加载原校准器并计算六组结果；同时检查内容安全全量测试记录及混淆矩阵，不重新拟合，不进行全量模型推理。

复核报告位置：`runs/fast_001/review/report.html`，用浏览器直接打开；命令行数值表见同目录的 `report.md` 及各指标MD报告。

同目录下各指标报告：
指标1.11 不确定性量化建模：`signal_report.html`、`signal_report.md`，查看不确定性信号数。
指标1.12 不确定性来源分析：`calibration_report.html`、`calibration_report.md`，查看六组校准性能提升及平均值。
指标1.13 多模态生成内容风险评估：`warning_report.html`、`warning_report.md`，查看高风险内容提前预警准确率。

逐样本信号明细在 `runs/fast_001/review/samples/`；数值汇总在 `runs/fast_001/review/summary.json`。HTML报告不依赖外部网页。

## 3 准备现场抽测环境和官方模型

```bash
bash scripts/01_setup_envs.sh --execute
bash scripts/02_prepare_assets_data.sh --execute \
  --download-models --skip-optional-roles
```

成功提示：`ENV_OK`、`MODEL_PATHS_OK`。脚本按固定版本下载LLaVA、Qwen、InternVL及答案核验模型，并核验文件；本步骤不需要准备全量数据，出现 `DATA_PENDING` 不影响本版现场抽测。

如从本版网盘下载了 `models/` 与 `env/`，用随包《交付说明》中的离线命令代替这两条命令，不必再联网下载。模型与原版为同一官方版本，不上传到GitHub。

## 4 运行现场抽测（指标1.11、1.12、1.13的功能检查）

```bash
bash scripts/fast_03_demo.sh
cat runs/fast_001/demo/demo_report.md
```

成功提示：`FRESH_SAMPLE_DEMO_COMPLETE`。

使用预先按样本ID固定选取的POPE和TextVQA各3题，三个模型分别重新推理，共18条模型记录；加载原校准器展示回答、置信度及实际参与信号。内容安全在四类来源各抽2题，共8题，重新生成并运行多层预警。抽测不拟合、不读取历史回答，也不以小样本准确率代替全量指标。

现场报告：`runs/fast_001/demo/demo_report.md`；事实性明细：`factual/demo_results.json`；内容安全明细：`safety/sample_01.json`至`sample_08.json`（均位于该demo目录）。

## 5 生成并查看总报告

```bash
bash scripts/fast_04_finish.sh --method-approved
```

仅在测试方确认采用本流程后使用 `--method-approved`。未确认时去掉该参数，程序保留“待确认”的状态；现场抽测未完成时不能生成通过结论。

总报告位置：`runs/fast_001/report.html`，用浏览器直接打开；文字版为同目录的 `report.md`。总报告汇总全量复核与现场抽测的状态，各指标数值见第2步分报告。

`FAST_REVIEW_PASS`表示本版数值要求与现场功能检查均通过，不表示完成了全量重新推理。

再次测评时，先执行 `export MSRC_FAST_RUN=fast_002`，随后重复第2至5步；新结果统一保存到 `runs/fast_002/`，不覆盖前次结果。
