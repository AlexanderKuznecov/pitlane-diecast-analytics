import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder

customers = pd.read_csv("data/customers.csv")
products = pd.read_csv("data/products.csv")
orders = pd.read_csv("data/orders.csv")
order_items = pd.read_csv("data/order_items.csv")
reviews = pd.read_csv("data/reviews.csv")

before = order_items.copy()
before_customers = customers.copy()
before_reviews = reviews.copy()

# Етап 2. Предварителна обработка: дубликати, типове, липсващи стойности, outliers

print("-------------------------------")
print("1. Преди почистването (липсващи / дубликати):")
for name, df in {"Клиенти": customers, "Поръчки": orders, "Детайли": order_items, "Мнения": reviews}.items():
    print(name)
    print(df.isnull().sum().sum())
    print(df.duplicated().sum())

# 2. Дубликати
customers = customers.drop_duplicates()
orders = orders.drop_duplicates()

# 3. Типове на данните: датите са текст, а в поръчките има и друг формат
iso = pd.to_datetime(orders["order_date"], format="%Y-%m-%d", errors="coerce")
other = pd.to_datetime(orders["order_date"], format="%d.%m.%Y", errors="coerce")
orders["order_date"] = iso.fillna(other)
customers["registration_date"] = pd.to_datetime(customers["registration_date"])
reviews["review_date"] = pd.to_datetime(reviews["review_date"])

# 4. Несъответствия: "  sofia" и "VARNA" -> "Sofia" и "Varna"
customers["city"] = customers["city"].str.strip().str.title()

# 5. Невалидни и липсващи стойности
customers.loc[(customers["age"] < 16) | (customers["age"] > 100), "age"] = np.nan
customers["age"] = customers["age"].fillna(customers["age"].median()).astype(int)
orders["payment_method"] = orders["payment_method"].fillna("Unknown")
reviews = reviews.dropna(subset=["review_text"])
reviews = reviews[reviews["rating"].between(1, 5)]

print("-------------------------------")
print("2. Типове след корекцията:")
print(orders.dtypes)

print("-------------------------------")
print("3. Градове след корекцията:")
print(customers["city"].value_counts())
print("-------------------------------")
print("3б. Възраст и оценки преди и след корекцията (минимум и максимум):")
print("Възраст преди:")
print(before_customers["age"].min(), before_customers["age"].max())
print("Възраст след:")
print(customers["age"].min(), customers["age"].max())
print("Оценки преди:")
print(before_reviews["rating"].min(), before_reviews["rating"].max())
print("Оценки след:")
print(reviews["rating"].min(), reviews["rating"].max())

# 6. Отклоняващи се стойности в количеството: IQR и Z-score
Q1 = order_items["quantity"].quantile(0.25)
Q3 = order_items["quantity"].quantile(0.75)
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
iqr_outliers = order_items[order_items["quantity"] > upper]

z = np.abs(stats.zscore(order_items["quantity"]))
z_outliers = order_items[z > 3]

print("-------------------------------")
print("4. Outliers по IQR (горна граница и редове):")
print(upper)
print(iqr_outliers)

print("-------------------------------")
print("5. Outliers по Z-score:")
print(z_outliers)

order_items = order_items[order_items["quantity"] <= upper].copy()

# липсващата стойност на реда се изчислява отново
order_items["line_total"] = (order_items["quantity"] * order_items["unit_price"]).round(2)

# 7. Сравнение преди и след
print("-------------------------------")
print("6. Сравнение преди и след почистването:")
print(before[["quantity", "line_total"]].describe().round(2))
print(order_items[["quantity", "line_total"]].describe().round(2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(x=before["quantity"], ax=axes[0], color="salmon")
axes[0].set_title("Количество преди почистването")
axes[0].set_xlabel("Количество")
sns.boxplot(x=order_items["quantity"], ax=axes[1], color="seagreen")
axes[1].set_title("Количество след почистването")
axes[1].set_xlabel("Количество")
plt.tight_layout()
plt.show()

# 8. Кодиране на категориални характеристики
le = LabelEncoder()
customers["category_code"] = le.fit_transform(customers["customer_category"])
payment_dummies = pd.get_dummies(orders["payment_method"], prefix="pay", dtype=int)

print("-------------------------------")
print("7. Label Encoding (категория на клиента):")
print(customers[["customer_category", "category_code"]].drop_duplicates())

print("-------------------------------")
print("8. One-Hot Encoding (начин на плащане):")
print(payment_dummies.head())

# мащабирането (StandardScaler) е в етап 6, където е необходимо за K-means

# 9. Свързване и запис на почистените данни
sales = order_items.merge(orders, on="order_id")
sales = sales.merge(products, on="product_id")
sales = sales.merge(customers, on="customer_id")

print("-------------------------------")
print("9. Почистена свързана таблица:")
print(sales.shape)
print(sales.isnull().sum().sum())

os.makedirs("data/clean", exist_ok=True)
sales.to_csv("data/clean/sales.csv", index=False)
customers.to_csv("data/clean/customers.csv", index=False)
reviews.to_csv("data/clean/reviews.csv", index=False)
