import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("SSB_benchmark_merged.csv")
df["TestCase"] = df["TestCase"].astype(str)  # 确保 TestCase 列是字符串类型
# df["TestCase"] = pd.Categorical(df["TestCase"],categories=["3.1", "3.2", "3.3", "4.1", "4.2", "2.1", "2.2", "2.3", "4.3", "3.4"], ordered=True)
# df["TestCase"] = pd.Categorical(df["TestCase"],categories=["3.1", "4.1", "4.2", "4.3", "2.1", "2.2", "2.3", "3.2", "3.3", "3.4"], ordered=True)
df["TestCase"] = pd.Categorical(df["TestCase"],
                                categories=["1.0", "0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1",
                                            "0.01", "0.001", "0.0001", "1e-05", "1e-06", "1e-07", "1e-08"], ordered=True)
df = df[["column" in s for s in df["Method"]]]
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="TestCase", y="Time", hue="Method", style="Method", markers=True, linewidth=2, markersize=8)
plt.tight_layout()
# plt.yscale("log")
plt.savefig("benchmark_results.png")

#  