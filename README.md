# PMG-Bench

面向个性化图像生成的评测基准与实验框架，支持 FLICKR-AES / POG / SER30K 等历史数据集，以及主推的 **UserPref v1** 用户偏好 benchmark。

| 文档 | 内容 |
|------|------|
| **[`data/userpref_v1/README.md`](data/userpref_v1/README.md)** | UserPref v1 数据 schema、目录约定、`build_userpref_v1_dataset.py` |
| **[`USERPREF_BENCH_GUIDE.md`](USERPREF_BENCH_GUIDE.md)** | 四方法公平对比、复现命令、`results/` 论文指标与图表 |
| [`experiments/USAGE.txt`](experiments/USAGE.txt) | 实验一 / 实验三 / 指标冲突 命令行 |
| [`scripts/BENCH_FAIR_PARAMS.md`](scripts/BENCH_FAIR_PARAMS.md) | TI / DreamBooth / PMG 公平训练推理参数 |

---

## Installation

```bash
git clone https://github.com/tianyixu686/PMG-Bench.git
cd PMG-Bench
pip install -r requirements.txt
# 可选：pip install -r experiments/exp3_bench_metrics/requirements_eval_extras.txt
```

Python 3.9–3.10、CUDA 11.8+、PyTorch ≥ 2.0、`runwayml/stable-diffusion-v1-5`。完整 conda 环境见 `method/environment.yaml`。

---

## Project Structure

```
PMG-Bench/
├── data/userpref_v1/    # 数据集说明（README）与 sources.example.json
├── data/examples/       # 少量示例（会提交）
├── experiments/         # 评测与分析
├── method/              # TI / DreamBooth / IP-Adapter / PMG 等
├── scripts/             # Slurm 批处理
├── tools/               # 数据构建与评测工具
├── results/             # 可提交的汇总指标与论文图
└── outputs/             # 本地生成物（.gitignore，不提交）
```

---

## License

MIT（各原始数据集遵循其来源许可）。
