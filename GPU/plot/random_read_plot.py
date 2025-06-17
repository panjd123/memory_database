import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("gpu_read_bw.csv")
df = df.sort_values(by="DataSizeMB")
df["DataSizeMB_str"] = df["DataSizeMB"].astype(str)

sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

ax = sns.lineplot(data=df, x="DataSizeMB_str", y="BandwidthGBps", hue="TestType", marker="o")

plt.title("GPU Read Bandwidth vs Data Size")
plt.xlabel("Data Size (MB)")
plt.ylabel("Bandwidth (GB/s)")
plt.legend(title="Test Type")

plt.xticks(rotation=45)

# 给每个点添加带宽数值标签
for line in ax.get_lines():
    x_data = line.get_xdata()
    y_data = line.get_ydata()
    for x, y in zip(x_data, y_data):
        label = f"{y:.3f}"
        ax.text(x, y, label, fontsize=8, ha='left', va='bottom', rotation=30)

plt.tight_layout()
plt.savefig("gpu_read_bandwidth_annotated.png", dpi=300)
plt.show()
