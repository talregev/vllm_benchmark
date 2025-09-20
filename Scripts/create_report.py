import json, os, glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load benchmark results
files = sorted(glob.glob("benchmark-results/*.json"),
               key=lambda path: int(os.path.splitext(os.path.basename(path))[0]))

records = []
for path in files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    concurrency = int(os.path.splitext(os.path.basename(path))[0])
    records.append({
        "concurrency": concurrency,
        "throughput": data["throughput"],  # tokens/sec
        "ttft_ms":    data["ttft_ms"],     # ms
        "itl_ms":     data["itl_ms"],      # ms
        "e2e_ms":     data["e2e_ms"],      # ms
    })

df = pd.DataFrame(records).sort_values("concurrency")

sns.set_theme(style="whitegrid", context="talk")
plots = [
    ("throughput", "Throughput",          "tokens/sec", "throughput.png"),
    ("ttft_ms",    "Time To First Token", "ms",         "ttft_ms.png"),
    ("itl_ms",     "Inter Token Latency", "ms",         "itl_ms.png"),
    ("e2e_ms",     "End-to-End Latency",  "ms",         "e2e_ms.png"),
]

for metric, title, ylabel, outfile in plots:
    ax = sns.barplot(data=df, x="concurrency", y=metric)
    ax.set_title(title)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel(ylabel)
    for bar in ax.patches:
        value = bar.get_height()
        ax.annotate(f"{value:.2f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", xytext=(0, 5), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.clf()

print("Saved:", [outfile for _, _, _, outfile in plots])
