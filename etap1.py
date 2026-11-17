import os
import numpy as np
import pandas as pd
from faker import Faker

# фиксиран seed
np.random.seed(42)
fake = Faker("en_US")
Faker.seed(42)

# брой записи
N_CUSTOMERS = 120
N_ORDERS = 1200
N_REVIEWS = 350

# А. Генериране на данните (PitLane Diecast: онлайн магазин за Hot Wheels и F1 модели)

# 1. Продукти: идентификатор, име, категория, цена, себестойност
products = pd.DataFrame([
    ["Mainline Nissan Skyline GT-R R34", "Mainline", 3.49, 1.60],
    ["Mainline Toyota Supra", "Mainline", 3.49, 1.60],
    ["Mainline Porsche 911 GT3", "Mainline", 3.99, 1.80],
    ["Mainline Ford Mustang", "Mainline", 3.49, 1.60],
    ["Mainline Lamborghini Huracan", "Mainline", 3.99, 1.80],
    ["Mainline Honda Civic Type R", "Mainline", 3.49, 1.60],
    ["Car Culture Datsun 510", "Premium", 8.99, 4.00],
    ["Car Culture Mazda RX-7", "Premium", 9.49, 4.20],
    ["Car Culture BMW M3 E30", "Premium", 9.99, 4.50],
    ["Car Culture Toyota AE86", "Premium", 9.49, 4.20],
    ["Treasure Hunt Chevrolet Camaro", "Treasure Hunt", 7.99, 3.50],
    ["Super Treasure Hunt Nissan Fairlady Z", "Treasure Hunt", 24.99, 11.00],
    ["Treasure Hunt Dodge Charger", "Treasure Hunt", 7.99, 3.50],
    ["F1 Red Bull Racing 1:64", "F1 Collection", 11.99, 5.50],
    ["F1 Ferrari 1:64", "F1 Collection", 11.99, 5.50],
    ["F1 McLaren 1:64", "F1 Collection", 11.99, 5.50],
    ["F1 Aston Martin 1:64", "F1 Collection", 11.99, 5.50],
    ["F1 Mercedes 1:43 Elite", "F1 Collection", 39.99, 19.00],
    ["Track Builder Stunt Set", "Track Sets", 34.99, 24.50],
    ["Mega Garage Playset", "Track Sets", 59.99, 42.00],
    ["Monster Loop Race Track", "Track Sets", 44.99, 31.50],
    ["Collector Display Case", "Accessories", 24.99, 12.00],
    ["Protector Pack Blister Covers", "Accessories", 8.99, 4.00],
    ["Car Carrying Case", "Accessories", 19.99, 9.50],
], columns=["product_name", "category", "price", "cost"])
products.insert(0, "product_id", range(1, len(products) + 1))

# 2. Клиенти: идентификатор, име, възраст, град, дата на регистрация, категория
cities = ["Sofia", "Plovdiv", "Varna", "Burgas", "Ruse", "Stara Zagora"]
customers = pd.DataFrame({
    "customer_id": range(1, N_CUSTOMERS + 1),
    "name": [fake.name() for _ in range(N_CUSTOMERS)],
    "age": np.random.randint(16, 60, N_CUSTOMERS),
    "city": np.random.choice(cities, N_CUSTOMERS, p=[0.4, 0.2, 0.15, 0.1, 0.08, 0.07]),
    "registration_date": [fake.date_between(pd.Timestamp("2022-01-01"), pd.Timestamp("2025-10-31")) for _ in range(N_CUSTOMERS)],
    "customer_category": np.random.choice(["Collector", "F1 Fan", "Casual", "Gift Buyer"], N_CUSTOMERS, p=[0.25, 0.25, 0.35, 0.15]),
})

# 3. Поръчки: 12 месеца (11.2025 - 10.2026) със сезонност
days = pd.date_range("2025-11-01", "2026-10-31")
month_weight = {11: 1.6, 12: 1.8, 1: 0.6, 2: 0.8, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 0.7, 8: 0.7, 9: 0.9, 10: 1.3}
day_weight = np.array([month_weight[d.month] * (1.3 if d.day_of_week >= 5 else 1) for d in days])
day_weight[(days >= "2026-02-10") & (days <= "2026-02-12")] = 0     # сайтът не работи 3 дни

# колекционерите и феновете на F1 поръчват по-често
customer_weight = customers["customer_category"].map({"Collector": 4, "F1 Fan": 3, "Casual": 1.5, "Gift Buyer": 1})

orders = pd.DataFrame({
    "order_id": range(1, N_ORDERS + 1),
    "customer_id": np.random.choice(customers["customer_id"], N_ORDERS, p=customer_weight / customer_weight.sum()),
    "order_date": np.sort(np.random.choice(days, N_ORDERS, p=day_weight / day_weight.sum())),
    "status": np.random.choice(["Delivered", "Cancelled", "Returned"], N_ORDERS, p=[0.88, 0.07, 0.05]),
    "payment_method": np.random.choice(["Card", "PayPal", "Cash on delivery"], N_ORDERS, p=[0.5, 0.3, 0.2]),
})

# 4. Детайли на поръчките: 1-3 продукта в поръчка, стойност = количество * цена
# всеки тип клиент предпочита различни категории
preference = {
    "Collector": {"Mainline": 1, "Premium": 3, "Treasure Hunt": 3, "F1 Collection": 1.5, "Track Sets": 0.5, "Accessories": 1.5},
    "F1 Fan": {"Mainline": 0.5, "Premium": 0.5, "Treasure Hunt": 0.5, "F1 Collection": 6, "Track Sets": 0.3, "Accessories": 1},
    "Casual": {"Mainline": 5, "Premium": 1, "Treasure Hunt": 0.5, "F1 Collection": 0.5, "Track Sets": 2, "Accessories": 0.7},
    "Gift Buyer": {"Mainline": 2, "Premium": 1, "Treasure Hunt": 0.5, "F1 Collection": 1.5, "Track Sets": 4, "Accessories": 1},
}
customer_type = customers.set_index("customer_id")["customer_category"]

rows = []
for order in orders.itertuples():
    weight = products["category"].map(preference[customer_type[order.customer_id]])
    for product_id in np.random.choice(products["product_id"], np.random.randint(1, 4), replace=False, p=weight / weight.sum()):
        quantity = np.random.choice([1, 1, 1, 2, 3])
        price = products.loc[products["product_id"] == product_id, "price"].values[0]
        rows.append([order.order_id, product_id, quantity, price, round(quantity * price, 2)])
order_items = pd.DataFrame(rows, columns=["order_id", "product_id", "quantity", "unit_price", "line_total"])

# 5. Мнения, текстът зависи от оценката
positive = ["Absolutely loved {p}! Best addition to my collection.", "{p} is amazing, great paint and fast delivery!!!",
            "Fantastic, {p} exceeded my expectations :) https://youtube.com/watch?v=hw123"]
neutral = ["{p} is ok, nothing special.", "Delivery took a week, {p} is as described."]
negative = ["Very disappointed with {p}, the paint was chipped.", "Terrible! {p} arrived in a crushed box and late!!!",
            "{p} is overpriced, waste of money :( www.reddit.com/r/HotWheels"]

rows = []
for i, order in enumerate(orders[orders["status"] == "Delivered"].sample(N_REVIEWS, random_state=42).itertuples()):
    product_id = np.random.choice(order_items.loc[order_items["order_id"] == order.order_id, "product_id"])
    rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.35, 0.3])
    if order.order_date.month == 12:
        rating = max(1, rating - 2)   # забавени доставки по Коледа -> по-ниски оценки
    if products.loc[products["product_id"] == product_id, "category"].values[0] == "Track Sets":
        rating = max(1, rating - 1)   # пистите се чупят -> по-ниски оценки
    texts = positive if rating >= 4 else neutral if rating == 3 else negative
    name = products.loc[products["product_id"] == product_id, "product_name"].values[0]
    review_date = order.order_date + pd.Timedelta(days=int(np.random.randint(2, 15)))
    rows.append([i + 1, order.order_id, order.customer_id, product_id, review_date,
                 np.random.choice(texts).format(p=name), rating])
reviews = pd.DataFrame(rows, columns=["review_id", "order_id", "customer_id", "product_id", "review_date",
                                      "review_text", "rating"])

# 6. Контролирани проблеми в данните
customers["age"] = customers["age"].astype(float)
customers.loc[[3, 15, 40, 77], "age"] = np.nan                         # липсващи стойности
customers.loc[[10, 50], "age"] = [150, -5]                             # отклоняващи се стойности
customers.loc[[5, 25], "city"] = ["  sofia", "VARNA"]                  # неправилен формат
customers = pd.concat([customers, customers.iloc[[1, 2, 3]]])          # дубликати

orders["order_date"] = orders["order_date"].dt.strftime("%Y-%m-%d")
orders.loc[[100, 300, 500, 700], "order_date"] = ["15.12.2025", "03.02.2026", "21.05.2026", "10.08.2026"]
orders.loc[orders.sample(30, random_state=1).index, "payment_method"] = np.nan
orders = pd.concat([orders, orders.iloc[:10]])

order_items.loc[[20, 400, 900], "quantity"] = 50                       # нереални количества
order_items.loc[order_items.sample(15, random_state=2).index, "line_total"] = np.nan

reviews.loc[[7, 70, 170], "review_text"] = np.nan
reviews.loc[[12, 120], "rating"] = [0, 9]

# 7. Запис в CSV
os.makedirs("data", exist_ok=True)
customers.to_csv("data/customers.csv", index=False)
products.to_csv("data/products.csv", index=False)
orders.to_csv("data/orders.csv", index=False)
order_items.to_csv("data/order_items.csv", index=False)
reviews.to_csv("data/reviews.csv", index=False)

# Б. Зареждане и проверка
customers = pd.read_csv("data/customers.csv")
products = pd.read_csv("data/products.csv")
orders = pd.read_csv("data/orders.csv")
order_items = pd.read_csv("data/order_items.csv")
reviews = pd.read_csv("data/reviews.csv")

tables = {"Клиенти": customers, "Продукти": products, "Поръчки": orders,
          "Детайли": order_items, "Мнения": reviews}

print("-------------------------------")
print("1. Размери на таблиците:")
for name, df in tables.items():
    print(name)
    print(df.shape)

print("-------------------------------")
print("2. Структура и типове:")
for name, df in tables.items():
    print(name)
    df.info()

print("-------------------------------")
print("3. Основни показатели:")
print(order_items[["quantity", "unit_price", "line_total"]].describe().round(2))
print(customers["age"].describe().round(2))

print("-------------------------------")
print("4. Липсващи стойности и дубликати:")
for name, df in tables.items():
    print(name)
    print(df.isnull().sum().sum())
    print(df.duplicated().sum())

# свързани DataFrame обекти: поръчки + детайли + продукти + клиенти
sales = order_items.merge(orders.drop_duplicates(), on="order_id")
sales = sales.merge(products, on="product_id")
sales = sales.merge(customers.drop_duplicates(), on="customer_id")

print("-------------------------------")
print("5. Свързана таблица:")
print(sales.shape)
print(sales[["order_id", "name", "product_name", "quantity", "line_total", "status"]].head())

print("-------------------------------")
print("6. Съгласуваност на данните:")
print(orders["customer_id"].isin(customers["customer_id"]).all())
print(order_items["product_id"].isin(products["product_id"]).all())
