# OFA 可配置项（与 DefectFlywheel 对接）

## 命令行参数（已支持）

- **`--data_path`**：数据集根目录（如 DefectFlywheel 的 `datasets/ZJU-Leaper-Group5-MVTec_dev` 或某类别父路径）。
- **`--blip_model_path`**：BlipDiffusion 模型目录（需含 `model_index.json` 或为其子目录；若为 HF 缓存结构，代码会自动解析）。推荐在项目根执行 `python scripts/download_blipdiffusion.py` 从 Hugging Face 下载到 `ofa/blipdiffusion_model/`。若未设置，可从环境变量 `BLIP_MODEL_PATH` 读取。
- **`--save_path`**：结果与日志保存目录。
- **`--checkpoint_path`**：检查点目录。
- **`--image_size`**：输入图像缩放尺寸。**建议在 ZJU 模式下使用 `image_size=224`**，以与原 OFA 实现中 16×16 patch grid 的假设保持一致；当 `--dataset zju` 时，main_zju 会自动将 `image_size` 设为 224，无需手动指定。

## 环境变量（供 DefectFlywheel 脚本传入）

- **`BLIP_MODEL_PATH`**：BlipDiffusion 模型路径；若未传 `--blip_model_path`，main_zju 将优先使用此环境变量。
- **`DEFECT_FLYWHEEL_NORMAL_LIST`** / **`DEFECT_FLYWHEEL_SYNTHETIC_LIST`**：协同训练时，从 DefectFlywheel 清单加载数据时使用（由 co_train 或封装脚本设置）。

## 与 DefectFlywheel 的对接约定

1. **导出 .npy**：DefectFlywheel 的 `scripts/ofa_export_npy.py` 调用 OFA 前向，输出目录与 `mine_hard --anomaly_map_dir` 一致，文件名与正常图 basename 一致（`{basename}.npy`）。
2. **训练数据源**：当使用 DefectFlywheel 生成结果时，通过数据模式或环境变量指定两份列表路径，OFA 训练从该列表加载图像，不再使用「patch_embedding 相加+noise」构造伪异常。
