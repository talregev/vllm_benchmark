import json, os, glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load benchmark results
files = glob.glob("benchmark-results/*.json")

records = []
for path in files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    name = os.path.splitext(os.path.basename(path))[0]
    concurrency = int(''.join(filter(str.isdigit, name)))
    metrics = data["benchmarks"][0]["metrics"]
    records.append({
        "concurrency": concurrency,
        "throughput":  metrics["tokens_per_second"]["successful"]["mean"],      # tokens/sec
        "ttft_ms":     metrics["time_to_first_token_ms"]["successful"]["mean"], # ms
        "itl_ms":      metrics["inter_token_latency_ms"]["successful"]["mean"], # ms
        "e2e_ms":      metrics["request_latency"]["successful"]["mean"],        # ms
    })

df = pd.DataFrame(records).sort_values("concurrency")

sns.set_theme(style="whitegrid", context="talk")
plots = [
    ("throughput", "Throughput",          "tokens/sec", "throughput.png"),
    ("ttft_ms",    "Time To First Token", "ms",         "ttft_ms.png"),
    ("itl_ms",     "Inter Token Latency", "ms",         "itl_ms.png"),
    ("e2e_ms",     "End-to-End Latency",  "ms",         "e2e_ms.png"),
]

report_folder = "reports"
os.makedirs(report_folder, exist_ok=True)
for metric, title, ylabel, outfile in plots:
    ax = sns.barplot(data=df, x="concurrency", y=metric)
    ax.set_title(title, pad=10)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel(ylabel)
    # Extend y-axis limit by 15%
    ymax = df[metric].max()
    ax.set_ylim(0, ymax * 1.15)
    for bar in ax.patches:
        value = bar.get_height()
        ax.annotate(f"{value:.2f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", xytext=(0, 5), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(f"{os.path.join(report_folder,outfile)}", dpi=150)
    plt.clf()

print("Saved:", [outfile for _, _, _, outfile in plots])
