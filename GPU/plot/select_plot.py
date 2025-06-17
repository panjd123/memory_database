import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("SSB_benchmark_merged.csv")
df["TestCase"] = df["TestCase"].astype(str)  # 确保 TestCase 列是字符串类型
print(df["TestCase"].unique())
df["TestCase"] = pd.Categorical(df["TestCase"],
                                categories=["1.0", "0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1",
                                            "0.01", "0.001", "0.0001", "1e-05", "1e-06", "1e-07", "1e-08"], ordered=True)
df1 = df[df["Method"] == "columnwise_dynamic_vector"]
df2 = df[df["Method"] == "columnwise_static_vector"]
df_join = pd.merge(df1, df2, on=["TestCase", "code"], suffixes=("_dynamic", "_static"))
df_join["Speedup"] = df_join["Time_dynamic"] / df_join["Time_static"]
# plt.figure(figsize=(16, 10))
# sns.lineplot(data=df, x="TestCase", y="Time", hue="Method", style="Method", markers=True, linewidth=2, markersize=8)
# plt.yscale("log")
# plt.savefig("benchmark_results.png")
# print(df_join["TestCase"].unique())
fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(data=df_join, x="TestCase", y="Speedup", marker="o", ax=ax)
for line in ax.get_lines():
    x_data = line.get_xdata()
    y_data = line.get_ydata()
    for x, y in zip(x_data, y_data):
        label = f"{y:.3f}"
        ax.text(x, y, label, fontsize=8, ha='left', va='bottom')
plt.xlabel("ratio")
plt.xticks(rotation=30)
plt.title("Speedup of columnwise_static_vector over columnwise_dynamic_vector")
plt.savefig("benchmark_results_speedup.png")