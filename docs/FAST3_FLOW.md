# 7. 测试流程

指标1.11不确定性量化建模、1.12不确定性来源分析和1.13多模态生成内容风险评估具体测试流程：

## 1 检查已有环境与材料

使用本次通用版代码包附带的 check_env.py。已有本版本项目时，先进入实际 MSRC 目录检查；空目录或缺少脚本时，先按第 2 步仅取得小代码包，不先下载全部环境、模型和数据。以下路径为示例。

```bash
cd ~/MSRC_fast_local3x5090_20261103/MSRC &&
python3 -B check_env.py
```

ENV_REUSE_OK 表示现有 Python 及依赖通过检查，跳过环境下载和安装；REUSE_OK 表示该项校验通过。仅补齐 NEED_DOWNLOAD 项；NEED_PREPARE 表示需用本地材料准备，REPAIR 表示先处理报错。未检查的项不能视为通过。

## 2 按检查结果补齐材料和环境

代码及依赖补丁：https://pan.baidu.com/s/1pA-8ULIL2ITGZu0IsOmDWA　提取码：q2nd

模型、数据及已有参数：https://pan.baidu.com/s/1P9IswryNxUndKQlqLV6M0Q　提取码：b3kw

仅下载脚本列出的缺失或损坏文件；先确认已有模型的实际路径，不因路径错误重下模型。材料与 MSRC 并列放置。首次从代码分享的 checkfirst_20261105 子目录只取 MSRC-general-checkfirst-code-20261105.zip 及对应 SHA256SUMS.txt，放到材料目录后运行下列命令，返回第 1 步检查；不要覆盖正在运行的项目。

```bash
cd ~/MSRC_fast_local3x5090_20261103 &&
sha256sum -c SHA256SUMS.txt &&
unzip MSRC-general-checkfirst-code-20261105.zip &&
cd MSRC
```

仅在出现 NEED_ENV_INSTALL 时，补齐脚本列出的离线安装材料并执行；已有可用 CPython 3.10 时，不需要离线 Python 压缩包。环境已经合格则跳过本命令。

```bash
bash install_env.sh
```

安装入口复用原通用版安装脚本，只安装项目私有环境。其末尾 ENV_OK 检查需要已分配的 GPU；没有 GPU 或驱动异常时先处理，不反复重装。安装成功后返回第 1 步检查。

## 3 核验并准备本地材料

已有环境合格、缺失材料补齐后执行。已准备好的模型、数据及参数只核验，不重复下载或解压；本命令不联网下载、不训练、不生成测试答案。

```bash
python3 -B check_env.py --prepare &&
source activate.sh
```

须出现 ENV_REUSE_OK、MATERIALS_OK 和 CPU_PREPARATION_OK_NOT_GPU_ACCEPTANCE。出现 REPAIR、MATERIALS_INCOMPLETE 或其他错误时停止。此处通过不代表 GPU 或指标测试通过。

材料不在项目上级目录时，检查和准备命令均加 --materials-dir 实际材料根目录；模型路径需另行指定时加 --local-models 实际JSON路径。新终端或新作业须重新检查并 source activate.sh，保证后续使用同一个已检查的 Python。

## 4 运行指标测试（指标1.11、1.12、1.13）

先取得正式测试所需的计算资源，在实际计算节点执行完整环境检查。出现 ENV_OK 且正常退出后，再运行下面的原测试命令。

```bash
"$MSRC_PYTHON" -B check_env.py --gpu --environment-only
```

运行下列命令，测试并合并结果。

```bash
bash scripts/run_fast3_local.sh \
  --run-id local3_fitted_001 --gpus 0,1,2
```

流程完成提示：RUN_COMPLETED。报告中SCOPED_PASS表示本次测试范围达标，SCOPED_FAIL表示未达标，INCOMPLETE表示未完成。

## 5 生成并查看测试报告

测试完成后自动生成报告，可运行下列命令查看。

```bash
cat runs/local3_fitted_001/fast3/joint/report.md
cat runs/local3_fitted_001/fast3/joint/signal_report.md
cat runs/local3_fitted_001/fast3/joint/calibration_report.md
cat runs/local3_fitted_001/fast3/joint/warning_report.md
```

总报告位置（项目根目录下，用浏览器打开）：

runs/local3_fitted_001/fast3/joint/report.html
或
runs/local3_fitted_001/fast3/joint/report.md

同目录下总报告及各指标报告：
总报告：report.html、report.md。
指标1.11 不确定性量化建模：signal_report.html、signal_report.md，查看不确定性信号数。
指标1.12 不确定性来源分析：calibration_report.html、calibration_report.md，查看校准性能提升。
指标1.13 多模态生成内容风险评估：warning_report.html、warning_report.md，查看高风险内容提前预警准确率。
