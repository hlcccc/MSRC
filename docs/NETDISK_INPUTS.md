# 网盘材料说明（离线整包）

离线交付包：[MSRC_offline_20261030](https://pan.baidu.com/s/1w2iBAjIj0YfFniU2kFts_g)，提取码：tc5l。

该文件夹是**单链接完整包**，两个版本共用：代码与脚本、已处理数据（固定训练/测试划分）、项目已拟合参数（有拟合版用）、四套官方基座模型、离线依赖包、许可证与三张校验表。下载后按流程安装运行，全程不需要访问HuggingFace、GitHub或PyPI。

早先按版本分开的两个目录（MSRC_fitted / MSRC_no_fit）保留作历史版本，新流程一律使用本整包。

## 目录内容

| 路径 | 内容 |
| --- | --- |
| `MSRC-code-netdisk-20261030.zip` | 完整项目代码与全部脚本（解压得到 `MSRC/`） |
| `prepared_data/` | 已处理数据三个分卷 + `parts.json`，脚本自动合并导入 |
| `fitted_only/` | 项目已拟合参数（66 MB），仅有拟合版使用；无拟合版现场重新拟合 |
| `models/` | 四套官方基座模型（固定 revision，逐文件 SHA256）、`local-models.json`、`SHA256SUMS.txt`、镜像下载脚本 |
| `env/` | 固定版本依赖的 Linux/cp310 wheels、固定版本 CPython、`SHA256SUMS.txt` |
| `LICENSES/` + `MODELS_NOTICE.md` | 各模型许可全文与再分发声明 |
| `docs/` | 两版测试流程（Word 与 Markdown） |

## 三张校验表

```bash
cd ~/MSRC_offline_20261030
sha256sum -c SHA256SUMS.txt          # 代码、数据分卷、参数包、文档
sha256sum -c models/SHA256SUMS.txt   # 四套模型全部文件
sha256sum -c env/SHA256SUMS.txt      # 依赖 wheels 与 CPython 压缩包
```

## 数据与模型说明

数据包包含处理后的问题、图片、参考标注、自建良性样本及固定训练/测试划分。POPE为4,500/4,500条，TextVQA为2,504/2,496条，内容安全为3,360/3,360条；安全训练侧再固定为fit 2,688条、calibration 672条，同一原始问题、同图同问输入不跨集。来源与许可见包内`prepared_NOTICE.md`。

模型均为**上游官方发布的基座权重**，本项目未做训练、微调或量化；Qwen同时承担事实性被测模型和内容安全生成模型，包内只存一份（`local-models.json` 中 `qwen` 与 `generator` 指向同一目录）。原`SingGuard-4b`不随包分发：新多层流程以PlugGuard头作预警，准备脚本已支持在缺少该角色时正常工作；如需联网自取旧角色，可用`--download-models --skip-optional-roles`跳过它。

导入后的划分文件在`assets/partitions/`：`factual/pope_train.jsonl`、`pope_test.jsonl`、`textvqa_train.jsonl`、`textvqa_test.jsonl`；`safety/train.jsonl`、`test.jsonl`。`summary.json`列出数量、身份校验值及跨集检查结果。数据包不含模型预测、指标结果或已拟合参数；参考答案属于数据集标注。

## 完整步骤

见[有拟合版](FLOW_SAFETY_MULTILAYER_FITTED.md)或[无拟合版](FLOW_SAFETY_MULTILAYER_NO_FIT.md)。

## 不使用网盘模型包时

仍可用同一项目从官方来源自行下载（耗时长、依赖国外网络）：

```bash
cd MSRC
bash scripts/01_setup_envs.sh --execute
bash scripts/02_prepare_assets_data.sh --execute --download-data --download-models --skip-optional-roles
bash scripts/02_prepare_assets_data.sh --check
```

也可以只用`models/fetch_models_mirror.py`从国内镜像（ModelScope）重取模型：

```bash
python3 models/fetch_models_mirror.py --dest ./models --models qwen,internvl,judge,llava
```

有拟合版再执行`bash scripts/02_prepare_fitted.sh --execute`及`--check`。随后继续对应流程第4—8步。总报告自动生成在`runs/<运行编号>/multilayer/joint/report.md`和同目录`report.html`。
