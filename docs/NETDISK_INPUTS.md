# 网盘材料说明

共用材料：prepared_data文件夹、MSRC-code-netdisk-20261030.zip、SHA256SUMS.txt和两个版本的测试流程。prepared_data内含已处理数据包的三个分卷及parts.json，脚本自动合并、导入，不需手动操作。fitted_only文件夹仅供有拟合版下载；无拟合版现场重新训练、拟合。

数据包包含处理后的问题、图片、参考标注、自建良性样本及固定训练/测试划分。POPE为4,500/4,500条，TextVQA为2,504/2,496条，内容安全为3,360/3,360条。安全训练侧再固定为fit 2,688条、calibration 672条；同一原始问题、同图同问输入不跨集。来源与许可见包内prepared_NOTICE.md。

导入命令为bash scripts/02_prepare_assets_data.sh --execute --prepared-package ../prepared_data --download-models。脚本校验图片、标注和固定划分，不再生成图片或重建数据集。软件依赖与官方模型仍由脚本下载，不是完全离线包。运行时的输入快照仍核对固定划分；不根据测试结果重划数据。

导入后的划分文件在assets/partitions/：factual/pope_train.jsonl、pope_test.jsonl、textvqa_train.jsonl、textvqa_test.jsonl；safety/train.jsonl、test.jsonl。summary.json列出数量、身份校验值及跨集检查结果。数据包不含模型预测、指标结果或已拟合参数；参考答案属于数据集标注。

完整原始数据重建方式（从官方来源下载并重新处理）见项目scripts目录中的同一批脚本。

完整步骤见[有拟合版](FLOW_SAFETY_MULTILAYER_FITTED.md)或[无拟合版](FLOW_SAFETY_MULTILAYER_NO_FIT.md)。

不使用网盘数据包时，直接用交付包内的MSRC项目目录，改为从官方来源下载原始数据后现场处理：

```bash
cd MSRC
bash scripts/01_setup_envs.sh --execute
bash scripts/02_prepare_assets_data.sh \
  --execute --download-data --download-models
bash scripts/02_prepare_assets_data.sh --check
```

该方式需自行下载官方原始数据，耗时明显更长；网盘数据包仍是推荐路径。

有拟合版再执行bash scripts/02_prepare_fitted.sh --execute及--check。随后继续对应流程第4—8步。总报告自动生成在runs/<运行编号>/multilayer/joint/report.md和同目录report.html。
