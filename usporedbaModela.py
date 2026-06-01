import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    "Model": ["GraphSAGE", "PinSage", "NGCF", "LightGCN"],
    "Recall@20": [0.0104, 0.0250, 0.0162, 0.0180],
    "NDCG@20": [0.0084, 0.0197, 0.0132, 0.0142],
    "Najbolja epoha": [1000, 1900, 2200, 2600]
}

df = pd.DataFrame(data)

df = df.sort_values(by="Recall@20", ascending=False)

print(df)

sns.set(style="whitegrid")

plt.figure()
ax = sns.barplot(x="Model", y="Recall@20", data=df)
plt.title("Usporedba modela prema Recall@20")
plt.xlabel("Model")
plt.ylabel("Recall@20")

for i, v in enumerate(df["Recall@20"]):
    ax.text(i, v + 0.0005, f"{v:.3f}", ha='center')

plt.tight_layout()
plt.show()

plt.figure()
ax = sns.barplot(x="Model", y="NDCG@20", data=df)
plt.title("Usporedba modela prema NDCG@20")
plt.xlabel("Model")
plt.ylabel("NDCG@20")

for i, v in enumerate(df["NDCG@20"]):
    ax.text(i, v + 0.0005, f"{v:.3f}", ha='center')

plt.tight_layout()
plt.show()

df_melted = df.melt(id_vars="Model",
                    value_vars=["Recall@20", "NDCG@20"],
                    var_name="Metrika",
                    value_name="Vrijednost")

plt.figure()
ax = sns.barplot(x="Model", y="Vrijednost", hue="Metrika", data=df_melted)

plt.title("Usporedba modela prema evaluacijskim metrikama")
plt.xlabel("Model")
plt.ylabel("Vrijednost metrike")
plt.legend(title="Metrika")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')

tablica = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc='center'
)

plt.title("Rezultati modela")
plt.show()
