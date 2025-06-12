import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("merged_benchmark_results.csv")
df["TestCase"] = df["TestCase"].astype(str)  # 确保 TestCase 列是字符串类型
df["TestCase"] = pd.Categorical(df["TestCase"], ordered=True)
plt.figure(figsize=(16, 10))
sns.lineplot(data=df, x="TestCase", y="Time", hue="Method", style="Method", markers=True, linewidth=2, markersize=8)
# plt.yscale("log")
plt.savefig("benchmark_results.png")

# order=["3.1", "3.2", "3.3", "4.1", "4.2", "2.1", "2.2", "2.3", "4.3", "3.4"]