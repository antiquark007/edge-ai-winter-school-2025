import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Paths
# =========================
CSV_PATH = Path("benchmarks/lenet5_benchmark.csv")
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

# =========================
# Load Data
# =========================
df = pd.read_csv(CSV_PATH)
df = df.sort_values("batch_size")

# =========================
# 1. Batch Size vs Epoch Time
# =========================
plt.figure()
plt.plot(df.batch_size, df.epoch_time_sec, marker="o")
plt.xlabel("Batch Size")
plt.ylabel("Epoch Time (seconds)")
plt.title("Training Time vs Batch Size")
plt.grid(True)
plt.savefig(OUT_DIR / "epoch_time_vs_batch_size.png")
plt.close()

# =========================
# 2. Batch Size vs Training Throughput
# =========================
plt.figure()
plt.plot(df.batch_size, df.train_throughput_samples_sec, marker="o")
plt.xlabel("Batch Size")
plt.ylabel("Training Throughput (samples/sec)")
plt.title("Training Throughput vs Batch Size")
plt.grid(True)
plt.savefig(OUT_DIR / "train_throughput_vs_batch_size.png")
plt.close()

# =========================
# 3. Batch Size vs Inference Latency
# =========================
plt.figure()
plt.plot(df.batch_size, df.inference_latency_ms, marker="o")
plt.xlabel("Batch Size")
plt.ylabel("Inference Latency (ms)")
plt.title("Single-Sample Inference Latency vs Batch Size")
plt.grid(True)
plt.savefig(OUT_DIR / "latency_vs_batch_size.png")
plt.close()

# =========================
# 4. Batch Size vs Inference Throughput
# =========================
plt.figure()
plt.plot(df.batch_size, df.inference_throughput_samples_sec, marker="o")
plt.xlabel("Batch Size")
plt.ylabel("Inference Throughput (samples/sec)")
plt.title("Inference Throughput vs Batch Size")
plt.grid(True)
plt.savefig(OUT_DIR / "inference_throughput_vs_batch_size.png")
plt.close()

# =========================
# 5. Batch Size vs Accuracy
# =========================
plt.figure()
plt.plot(df.batch_size, df.accuracy, marker="o")
plt.xlabel("Batch Size")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Batch Size")
plt.ylim(0.9, 1.0)
plt.grid(True)
plt.savefig(OUT_DIR / "accuracy_vs_batch_size.png")
plt.close()

# =========================
# 6. Batch Size vs GPU Memory
# =========================
plt.figure()
plt.plot(df.batch_size, df.max_gpu_mem_mb, marker="o")
plt.xlabel("Batch Size")
plt.ylabel("Peak GPU Memory (MB)")
plt.title("GPU Memory Usage vs Batch Size")
plt.grid(True)
plt.savefig(OUT_DIR / "gpu_memory_vs_batch_size.png")
plt.close()

# =========================
# 7. Training Throughput vs GPU Memory
# =========================
plt.figure()
plt.scatter(df.max_gpu_mem_mb, df.train_throughput_samples_sec)
plt.xlabel("GPU Memory (MB)")
plt.ylabel("Training Throughput (samples/sec)")
plt.title("Training Throughput vs GPU Memory")
plt.grid(True)
plt.savefig(OUT_DIR / "throughput_vs_memory.png")
plt.close()

# =========================
# 8. Inference Latency vs Throughput
# =========================
plt.figure()
plt.scatter(
    df.inference_latency_ms,
    df.inference_throughput_samples_sec
)
plt.xlabel("Inference Latency (ms)")
plt.ylabel("Inference Throughput (samples/sec)")
plt.title("Latency vs Throughput Trade-off")
plt.grid(True)
plt.savefig(OUT_DIR / "latency_vs_throughput.png")
plt.close()

# =========================
# 9. Normalized Speedup vs Batch Size
# =========================
base_throughput = df.train_throughput_samples_sec.iloc[0]
speedup = df.train_throughput_samples_sec / base_throughput

plt.figure()
plt.plot(df.batch_size, speedup, marker="o")
plt.xlabel("Batch Size")
plt.ylabel("Normalized Speedup")
plt.title("Training Speedup vs Batch Size")
plt.grid(True)
plt.savefig(OUT_DIR / "speedup_vs_batch_size.png")
plt.close()

# =========================
# 10. Throughput Efficiency
# =========================
efficiency = df.train_throughput_samples_sec / df.batch_size

plt.figure()
plt.plot(df.batch_size, efficiency, marker="o")
plt.xlabel("Batch Size")
plt.ylabel("Throughput per Sample")
plt.title("Training Efficiency vs Batch Size")
plt.grid(True)
plt.savefig(OUT_DIR / "efficiency_vs_batch_size.png")
plt.close()

print(f"All plots saved to: {OUT_DIR.resolve()}")
