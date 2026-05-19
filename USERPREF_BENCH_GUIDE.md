# UserPref v1 评测与复现说明

本文件描述 **UserPref v1** 公平对比实验、目录约定与论文结果汇总。  
数据集 schema、构建步骤仍以 [`data/userpref_v1/README.md`](data/userpref_v1/README.md) 为准（请勿与本文混用）。

---

## 实验概览

在 SD1.5、`512×512` 下对四种方法公平对比：

| 方法 | 训练脚本（Slurm） | 说明 |
|------|-------------------|------|
| Textual Inversion | `scripts/sbatch_userpref_bench_ti_allusers.sh` | `max_train_steps=1000`，见 `scripts/BENCH_FAIR_PARAMS.md` |
| DreamBooth LoRA | `scripts/sbatch_userpref_bench_dreambooth_allusers.sh` | 同上 |
| IP-Adapter | 无训练 | 推理生成 `generation_results.json` |
| PMG | `scripts/sbatch_userpref_pmg_train_infer.sh` | 见 method 目录 |

训练/推理完成后统一评测：`scripts/sbatch_userpref_bench_eval_allmethods.sh`。

---

## 目录约定

| 路径 | 是否提交 Git | 用途 |
|------|--------------|------|
| `data/userpref_v1/` | 仅模板与说明 | 完整数据本地构建（见 `.gitignore`） |
| `data/examples/` | ✅ | 少量示例图与 `test_cases.sample.json` |
| `outputs/` | ❌ | 训练 checkpoint、逐样本生成图 |
| `results/` | ✅ | 论文级指标 CSV/JSON 与关键图 |
| `logs/` | ❌ | Slurm 日志 |

---

## 复现命令（摘要）

```bash
# 1. 数据（详见 data/userpref_v1/README.md）
cp data/userpref_v1/sources.example.json data/userpref_v1/sources.json
python tools/build_userpref_v1_dataset.py

# 2. 评测（需已有生成目录）
python -m experiments.exp3_bench_metrics.run_eval_benchmark \
  --test_json data/userpref_v1/processed_dataset/test.json \
  --input outputs/userpref_v1_pmg_infer/userpref_pmg_exp1 \
  --output_json outputs/experiments/exp3_bench_metrics/pmg_eval.json

# 3. 查看仓库内已汇总结果
cat results/bench_fair_20260506/final_metrics.csv
```

更多子命令见 `experiments/USAGE.txt`。

---

## Main Results（bench_fair_20260506）

Test 集 **162** 条样本；对应论文 **Table 4.x / Fig 4.5–4.8**。

| Method | CLIP-I ↑ | CLIP-T ↑ | DINO-I ↑ | LPIPS ↓ | HPSv2 ↑ |
|--------|----------|----------|----------|---------|---------|
| DreamBooth LoRA | **0.7458±0.1004** | **0.3047±0.0298** | 0.4221±0.1553 | 0.7789±0.0616 | **0.2531±0.0362** |
| PMG | 0.7339±0.1048 | 0.3065±0.0278 | **0.4264±0.1550** | 0.7773±0.0668 | 0.2488±0.0361 |
| Textual Inversion | 0.6905±0.1050 | 0.2709±0.0425 | 0.3343±0.1550 | 0.7792±0.0582 | 0.2255±0.0391 |
| IP-Adapter | 0.6468±0.1210 | 0.2345±0.0600 | 0.3158±0.1531 | 0.8111±0.0557 | 0.2097±0.0522 |

图表路径：`results/bench_fair_20260506/figures/`。

---

## 相关文件

- 公平超参表：`scripts/BENCH_FAIR_PARAMS.md`
- 实验代码：`experiments/`
- 根目录安装与结构：`README.md`
