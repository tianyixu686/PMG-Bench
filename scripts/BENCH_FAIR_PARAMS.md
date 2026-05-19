# UserPref 四方法公平对比：训练 / 推理参数（对齐 DreamBench 思路）

统一前提：**SD1.5 同一权重**、`512×512`、**test 集样本索引**与 `processed_dataset/test.json` 一致；历史图筛选与 IP-Adapter 基线一致：**`preference_score ≥ 4.0`，最多 10 张**（由各方法数据构造脚本或推理脚本保证）。

| 方法 | 训练 | 推理（统一） |
|------|------|--------------|
| **Textual Inversion** | `lr=5e-4`，`max_train_steps=1000`（`sbatch_userpref_bench_ti_allusers.sh`）。`train_batch_size=4`（显存不足时用 `2`），`history_policy=threshold`，`history_threshold=4.0`，`num_vectors=8`，`image_size=512` | `num_inference_steps=100`，`guidance_scale=7.5`，`negative_prompt` 与 DreamBooth 脚本默认一致 |
| **DreamBooth LoRA** | `learning_rate=5e-5`，`max_train_steps=500`，`train_batch_size=1`，`gradient_accumulation_steps=4`（等效 batch 4），`history_policy=threshold`，`history_threshold=4.0`，`rank=4`，`fp16` | 同上 |
| **IP-Adapter** | 无训练 | **已有** `generation_results.json`；新实验请与上表推理一致（本仓库 sbatch 仍指向旧参数时，评测以**生成图文件**为准，重跑推理可改 `num_inference_steps=100` / `guidance_scale=7.5`） |
| **PMG** | 全量训练见 `sbatch_userpref_pmg_train_infer.sh`（与仓库一致） | `USERPREF_PMG_INFER.py`：`image_size=512`，`seed=42`，单张生成；若需与上表完全一致，可在该脚本中增加与 SD 相同的 scheduler 步数（当前由 PMG 内部 pipeline 决定） |

说明：**TI 的 DreamBench 表为 batch=4**；若 `train_batch_size` 大于 manifest 条数，训练脚本会自动 clamp。DreamBooth 表为 **LoRA：lr=5e-5, steps=500, eff batch=4**。

输出目录约定（便于 `run_eval_benchmark.py` 扫描）：

- TI / DreamBooth / PMG：`outputs/.../<run_name>/` 下递归存在 `sample_XXXX/gen_0.jpg`（**不要**再按 user 分子目录放 infer，否则需多次合并）。
- IP-Adapter：`.../generation_results.json`。

评测与出图：训练推理完成后运行 `scripts/sbatch_userpref_bench_eval_allmethods.sh`（或其中命令），生成 3.2 维度表、HPS/FID 表、散点、雷达与案例拼图。

## 已在本环境提交的 Slurm 作业（示例）

以下使用同一 `RUN_NAME=bench_fair_20260506`，便于 TI / DreamBooth / 评测输出落在同一目录：

- Textual Inversion 全用户：`sbatch` job **39974** → `scripts/sbatch_userpref_bench_ti_allusers.sh`
- DreamBooth 全用户：`sbatch` job **39975** → `scripts/sbatch_userpref_bench_dreambooth_allusers.sh`
- 评测 + 论文图（在 39974、39975 成功后）：`sbatch` job **39976** → `scripts/sbatch_userpref_bench_eval_allmethods.sh`（`--dependency=afterok:39974:39975`）

输出目录：

- 指标 JSON：`outputs/experiments/bench_metrics/bench_fair_20260506/`
- 图表与表：`outputs/experiments/bench_paper_figs/bench_fair_20260506/`

若需重跑 TI 且跳过已存在 embedding，在训练命令中加 `--overwrite`（需改 sbatch 或本地运行）。

**务必**：两次 `sbatch` 训练前在同一 shell 执行 `export RUN_NAME=bench_fair_20260506`（或你的统一名字），否则评测脚本默认按提交日 `date` 可能找不到 TI/DB 的 eval 文件。
