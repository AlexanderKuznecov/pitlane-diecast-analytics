import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform, binom, pearsonr, spearmanr


# Етап 3. Статистически анализ: описателна статистика, разпределения,
# корелации(нормално, равномерно и биномно)


np.random.seed(42)

sales = pd.read_csv("data/clean/sales.csv", parse_dates=["order_date"])
customers = pd.read_csv("data/clean/customers.csv")

# стойност на всяка поръчка
orders = sales.groupby("order_id").agg(order_value=("line_total", "sum"),
                                       n_products=("product_id", "count"),
                                       quantity=("quantity", "sum"),
                                       age=("age", "first")).reset_index()

# 1. Описателна статистика
stats_table = pd.DataFrame()
for name, col in {"Цена": sales["unit_price"], "Количество": sales["quantity"],
                  "Стойност на поръчка": orders["order_value"]}.items():
    stats_table[name] = [col.mean(), col.median(), col.mode()[0], col.var(), col.std(),
                         col.quantile(0.25), col.quantile(0.5), col.quantile(0.75), col.min(), col.max()]
stats_table.index = ["Средна", "Медиана", "Мода", "Дисперсия", "Стд. отклонение",
                     "Q1", "Q2", "Q3", "Минимум", "Максимум"]

print("-------------------------------")
print("1. Описателна статистика:")
print(stats_table.round(2))

# 2. Разпределение на цени, количества и стойности на поръчки
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(sales["unit_price"], bins=20, ax=axes[0], color="darkorange")
axes[0].set_title("Цени")
axes[0].set_xlabel("Цена (EUR)")
axes[0].set_ylabel("Брой")
sns.countplot(x=sales["quantity"], ax=axes[1], color="gray")
axes[1].set_title("Количества")
axes[1].set_xlabel("Количество")
axes[1].set_ylabel("Брой")
sns.histplot(orders["order_value"], bins=30, kde=True, ax=axes[2], color="teal")
axes[2].set_title("Стойност на поръчка")
axes[2].set_xlabel("Стойност (EUR)")
axes[2].set_ylabel("Брой")
plt.tight_layout()
plt.show()

print("-------------------------------")
print("2. Асиметрия на стойността на поръчките:")
print(round(orders["order_value"].skew(), 2))

# 3. Нормално, равномерно и биномно разпределение
daily = sales.groupby("order_date")["line_total"].sum()
mu, sigma = norm.fit(daily)
return_rate = (sales.drop_duplicates("order_id")["status"] == "Returned").mean()
simulated_returns = np.random.binomial(100, return_rate, 1000)

print("-------------------------------")
print("3. Нормално (дневен приход): средна и стд. отклонение")
print(round(mu, 2))
print(round(sigma, 2))
print("Биномно: дял върнати поръчки")
print(round(return_rate, 3))
print("Вероятност за повече от 10 върнати от 100:")
print(round(1 - binom.cdf(10, 100, return_rate), 4))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
x = np.linspace(daily.min(), daily.max(), 200)
axes[0].hist(daily, bins=25, density=True, color="lightgray")
axes[0].plot(x, norm.pdf(x, mu, sigma), color="darkorange")
axes[0].set_title("Нормално: дневен приход")
axes[0].set_xlabel("Приход (EUR)")

x = np.linspace(10, 65, 200)
axes[1].hist(customers["age"], bins=11, density=True, color="lightgray")
axes[1].plot(x, uniform.pdf(x, 16, 44), color="darkorange")
axes[1].set_title("Равномерно: възраст на клиентите")
axes[1].set_xlabel("Възраст")

k = np.arange(0, 16)
axes[2].hist(simulated_returns, bins=np.arange(-0.5, 16), density=True, color="lightgray")
axes[2].plot(k, binom.pmf(k, 100, return_rate), "o-", color="darkorange")
axes[2].set_title("Биномно: върнати от 100 поръчки")
axes[2].set_xlabel("Брой върнати")
plt.tight_layout()
plt.show()

# 4. Pearson и Spearman корелации
r1, p1 = pearsonr(orders["n_products"], orders["order_value"])
s1, _ = spearmanr(orders["n_products"], orders["order_value"])
r2, p2 = pearsonr(orders["age"], orders["order_value"])
s2, _ = spearmanr(orders["age"], orders["order_value"])

print("-------------------------------")
print("4. Брой продукти и стойност (Pearson, p, Spearman):")
print(round(r1, 3))
print(round(p1, 4))
print(round(s1, 3))
print("Възраст и стойност (Pearson, p, Spearman):")
print(round(r2, 3))
print(round(p2, 4))
print(round(s2, 3))

# 5. Корелационна матрица, heatmap и scatter
cols = ["order_value", "n_products", "quantity", "age"]
labels = ["Стойност", "Брой продукти", "Количество", "Възраст"]
corr = orders[cols].corr()
corr.index = labels
corr.columns = labels

print("-------------------------------")
print("5. Корелационна матрица (Pearson):")
print(corr.round(2))

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0])
axes[0].set_title("Корелационна матрица")
sns.scatterplot(data=orders, x="n_products", y="order_value", alpha=0.4, ax=axes[1])
axes[1].set_title("Брой продукти и стойност")
axes[1].set_xlabel("Брой продукти")
axes[1].set_ylabel("Стойност (EUR)")
sns.scatterplot(data=orders, x="age", y="order_value", alpha=0.4, ax=axes[2], color="darkorange")
axes[2].set_title("Възраст и стойност")
axes[2].set_xlabel("Възраст")
axes[2].set_ylabel("Стойност (EUR)")
plt.tight_layout()
plt.show()

# 6. Интерпретация
strength = "слаба" if abs(r1) < 0.3 else "умерена" if abs(r1) < 0.7 else "силна"
print("-------------------------------")
print("6. Интерпретация:")
print(f"Брой продукти и стойност: r = {r1:.2f}, p = {p1:.4f} -> {strength} връзка.")
if p2 < 0.05:
    print(f"Възраст и стойност: r = {r2:.2f}, p = {p2:.2f} -> статистически значима, но много слаба връзка.")
else:
    print(f"Възраст и стойност: r = {r2:.2f}, p = {p2:.2f} -> няма значима връзка (p > 0.05).")
print("Корелацията показва само, че двете стойности се променят заедно, а не че едната причинява другата.")
