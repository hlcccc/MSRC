# 快速版交付说明

## 独立版本

- GitHub分支：`codex/fast-review-20261102`。
- 代码地址：https://github.com/hlcccc/MSRC/tree/codex/fast-review-20261102
- 新工作目录：`MSRC_Fast_Review`。原 `MSRC_Multilayer_Delivery` 及原分支不改。
- 网盘新建 `MSRC_fast_review_20261102`，不要覆盖、删除、重命名任何原交付目录。

## 网盘放什么

交给测试人的目录至少包含：

```text
MSRC_fast_review_20261102/
  MSRC-fast-review-code-20261102.zip
  MSRC-fast-evidence-20261102.tar.gz
  SHA256SUMS.txt
  课题二快速测试流程.docx
  课题二快速测试流程.md
  交付说明.md
```

另可直接在网盘内复制原交付中已经验证的 `models/` 与 `env/` 到本目录，供现场抽测离线安装使用。模型版本和文件内容不变，尤其保留RTX 5090所需的cu128依赖，不以cu126代替。不要复制旧的 `runs/`、报告或部分采集结果来冒充本次输出。

`MSRC-fast-evidence-20261102.tar.gz` 是本版特意交付的历史原始证据，不是基础模型权重，也不是只有汇总数字的结果表。它与现场新推理输出分开保存。压缩包包含少量基准样本与模型回答，分享前应遵循原数据集使用条件；不要提交到GitHub。

## 离线命令（已同时下载models和env时）

代码解压后进入 `MSRC/`，证据压缩包放入 `MSRC/delivery/`，`models/`、`env/` 与 `MSRC/` 同级。离线时按下面顺序完整执行，用本组命令代替主流程的第2至5步：

```bash
bash scripts/01_setup_envs.sh --execute \
  --offline-wheels ../env/wheels \
  --offline-python ../env/python/cpython-3.10.19+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
MSRC_BASE_PYTHON="$PWD/.venv/bin/python" \
  bash scripts/fast_01_setup.sh --offline-wheels ../env/wheels
bash scripts/fast_02_review.sh
bash scripts/02_prepare_assets_data.sh --execute \
  --local-models ../models/local-models.json
bash scripts/fast_03_demo.sh
bash scripts/fast_04_finish.sh --method-approved
```

最后一条同样仅在测试方确认采用快速测评方式后使用。复核环境只需NumPy和官方评分器的tqdm依赖；只有同时交付完整env和models时才可称为全程离线。

## 发布检查

上传后重新下载代码ZIP、证据压缩包和校验表，逐个计算SHA256；校验表必须为LF换行。确认模型目录映射仍为相对路径；在Linux全新目录解压并按测试流程执行。最后创建本目录的新分享链接，把真实链接加入随包《交付说明》，再把链接发给测试人。链接未生成前不要假称已经发布到网盘。

## 测评口径

事实性历史证据来自 `safety_no_fit_fresh_20261024_002`，多层内容安全证据来自 `multilayer_001`；两者不是同一次新联合实验。内容安全基准此前已被观察，本版不据此声称获得独立盲测泛化结果。安全时间戳复核限于历史本地输出缓冲，不等于新服务器HTTP对外输出时间，也不等于标注了首个有害token。
