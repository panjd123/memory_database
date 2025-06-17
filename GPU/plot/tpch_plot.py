import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("TPCH_benchmark_merged.csv")
df["Mline/ms"] = 600 / df["Time"]
print(df)

fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(data=df, x="TestCase", y="Mline/ms", hue="Method", ci=None, palette="muted", ax=ax)
sns.despine()
plt.xlabel("SF")
plt.legend(title="Method", loc="upper right")
plt.savefig("tpch.png", dpi=300, bbox_inches="tight")
