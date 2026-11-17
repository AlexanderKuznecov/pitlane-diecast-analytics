import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 10)

# Етап 6. Клиентска сегментация (K-means) и бизнес анализ

sales = pd.read_csv("data/clean/sales.csv", parse_dates=["order_date"])
sales = sales[sales["status"] == "Delivered"].copy()
sales["cost_total"] = sales["quantity"] * sales["cost"]
sales["profit"] = sales["line_total"] - sales["cost_total"]

# 1. Клиентски характеристики
customers = sales.groupby("customer_id").agg(frequency=("order_id", "nunique"),
                                             total_spent=("line_total", "sum")).reset_index()
customers["avg_order_value"] = customers["total_spent"] / customers["frequency"]
features = ["frequency", "total_spent", "avg_order_value"]

print("-------------------------------")
print("1. Клиентски характеристики:")
print(customers[features].describe().round(2))

# стандартизация - K-means смята разстояния, а стойностите са в различен мащаб
X = StandardScaler().fit_transform(customers[features])

# 2. Избор на брой клъстери - метод на лакътя
inertia = []
for k in range(2, 9):
    inertia.append(KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_)

print("-------------------------------")
print("2. Inertia за k = 2..8:")
print([round(i, 1) for i in inertia])

plt.plot(range(2, 9), inertia, marker="o")
plt.title("Метод на лакътя")
plt.xlabel("Брой клъстери k")
plt.ylabel("Inertia")
plt.show()

# 3. K-means
k = 4
customers["cluster"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)

# именуваме сегментите според средните разходи (от най-малко към най-много)
order = customers.groupby("cluster")["total_spent"].mean().sort_values().index
names = ["Occasional buyers", "Rare big spenders", "Loyal fans", "VIP collectors"]
customers["segment"] = customers["cluster"].map(dict(zip(order, names)))

segments = customers.groupby("segment")[features].mean().round(2)
segments["customers"] = customers["segment"].value_counts()
segments["revenue_share_%"] = (customers.groupby("segment")["total_spent"].sum()
                               / customers["total_spent"].sum() * 100).round(1)

print("-------------------------------")
print("3. Сегменти:")
print(segments.sort_values("revenue_share_%", ascending=False))

sns.scatterplot(data=customers, x="frequency", y="total_spent", hue="segment", s=70)
plt.title("Клиентски сегменти (K-means)")
plt.xlabel("Брой поръчки")
plt.ylabel("Общо похарчено (EUR)")
plt.legend(title="Сегмент")
plt.show()

# 4. Приходи, себестойност и печалба по категории
categories = sales.groupby("category").agg(revenue=("line_total", "sum"), cost=("cost_total", "sum"),
                                           profit=("profit", "sum"))
categories["margin_%"] = categories["profit"] / categories["revenue"] * 100
categories = categories.sort_values("profit", ascending=False).round(2)

print("-------------------------------")
print("4. Приходи, себестойност и печалба по категории:")
print(categories)
print("Общо приходи и печалба:")
print(round(sales["line_total"].sum(), 2))
print(round(sales["profit"].sum(), 2))

print("-------------------------------")
print("5. Топ 5 продукта по печалба:")
print(sales.groupby("product_name")["profit"].sum().sort_values(ascending=False).head(5).round(2))

# 5. Клиентско поведение: какво купува всеки сегмент
sales = sales.merge(customers[["customer_id", "segment"]], on="customer_id")
behaviour = pd.crosstab(sales["segment"], sales["category"], values=sales["line_total"],
                        aggfunc="sum", normalize="index") * 100

print("-------------------------------")
print("6. Какво купува всеки сегмент (% от разходите му):")
print(behaviour.round(1))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
categories[["revenue", "profit"]].plot(kind="barh", ax=axes[0])
axes[0].set_title("Приходи и печалба по категории")
axes[0].set_xlabel("EUR")
axes[0].set_ylabel("")
axes[0].legend(["Приходи", "Печалба"])
sns.heatmap(behaviour, annot=True, fmt=".0f", cmap="Oranges", ax=axes[1])
axes[1].set_title("Какво купува всеки сегмент (%)")
axes[1].set_xlabel("Категория")
axes[1].set_ylabel("Сегмент")
plt.tight_layout()
plt.show()

# 6. Интерпретация
top = segments["revenue_share_%"].idxmax()
print("-------------------------------")
print("7. Интерпретация:")
print(f"{top}: {segments.loc[top, 'customers']} клиенти носят {segments.loc[top, 'revenue_share_%']}% от приходите.")
print(f"Най-печеливша категория е {categories.index[0]}, а най-висок марж има {categories['margin_%'].idxmax()}.")

customers.to_csv("data/clean/customer_segments.csv", index=False)
