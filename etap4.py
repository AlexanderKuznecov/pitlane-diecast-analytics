import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

# Етап 4. Времеви редове и прогнозиране: сезонност, 3 базови модела и LSTM

np.random.seed(42)
tf.random.set_seed(42)

sales = pd.read_csv("data/clean/sales.csv", parse_dates=["order_date"])
sales = sales[sales["status"] == "Delivered"]

# 1. Времеви индекс, дневни приходи и брой поръчки
daily = sales.groupby("order_date").agg(revenue=("line_total", "sum"), orders=("order_id", "nunique"))
print("-------------------------------")
print("1. Дни с поръчки:")
print(len(daily))

# липсващи периоди: дни без нито една поръчка -> 0
daily = daily.asfreq("D").fillna(0)
print("Дни след asfreq (цялата година):")
print(len(daily))
print("Дни без нито една поръчка:")
print((daily["orders"] == 0).sum())

weekly = daily.resample("W").sum()
monthly = daily.resample("MS").sum()

print("-------------------------------")
print("2. Месечни приходи и поръчки:")
print(monthly.round(2))

# 2. Тенденции и сезонност
fig, axes = plt.subplots(3, 1, figsize=(13, 10))
axes[0].plot(daily["revenue"], color="lightgray", label="дневен приход")
axes[0].plot(daily["revenue"].rolling(7).mean(), color="darkorange", label="7-дневна средна")
axes[0].set_title("Дневни приходи (тенденция)")
axes[0].legend()
axes[1].plot(weekly["orders"], marker="o", color="teal")
axes[1].set_title("Брой поръчки по седмици")
axes[2].bar(monthly.index.strftime("%m.%Y"), monthly["revenue"], color="darkorange")
axes[2].set_title("Приходи по месеци (сезонност)")
plt.tight_layout()
plt.show()

# 3. Хронологично разделяне 80 / 20
revenue = daily["revenue"]
split = int(len(revenue) * 0.8)
train = revenue.iloc[:split]
test = revenue.iloc[split:]

print("-------------------------------")
print("3. Train и test период:")
print(train.index.min().date(), train.index.max().date())
print(test.index.min().date(), test.index.max().date())

# 4. Прогнози
# последна наблюдавана стойност
naive = revenue.shift(1).iloc[split:]

# пълзяща средна за последните 7 дни
moving_avg = revenue.rolling(7).mean().shift(1).iloc[split:]

# линейна регресия: номер на деня -> приход
t = np.arange(len(revenue)).reshape(-1, 1)
lr = LinearRegression()
lr.fit(t[:split], train)
linear = pd.Series(lr.predict(t[split:]), index=test.index)

# LSTM: от последните 7 дни предсказва следващия
window = 7
scaler = MinMaxScaler()
scaler.fit(train.values.reshape(-1, 1))
scaled = scaler.transform(revenue.values.reshape(-1, 1))

X, y = [], []
for i in range(window, len(scaled)):
    X.append(scaled[i - window:i, 0])
    y.append(scaled[i, 0])
X = np.array(X).reshape(-1, window, 1)
y = np.array(y)

X_train, y_train = X[:split - window], y[:split - window]
X_test = X[split - window:]

model = Sequential([Input(shape=(window, 1)), LSTM(32), Dense(1)])
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

lstm = scaler.inverse_transform(model.predict(X_test, verbose=0)).flatten()
lstm = pd.Series(lstm, index=test.index)

# 5. MAE, MSE, RMSE
results = pd.DataFrame(index=["MAE", "MSE", "RMSE"])
for name, pred in {"Последна стойност": naive, "Пълзяща средна": moving_avg,
                   "Линейна регресия": linear, "LSTM": lstm}.items():
    mse = mean_squared_error(test, pred)
    results[name] = [mean_absolute_error(test, pred), mse, np.sqrt(mse)]
results = results.T.round(2)

print("-------------------------------")
print("4. Сравнение на моделите:")
print(results)
print("Най-добър модел по MAE:")
print(results["MAE"].idxmin())
print("LSTM срещу базовия модел (последна стойност), MAE:")
print(results.loc["LSTM", "MAE"])
print(results.loc["Последна стойност", "MAE"])

plt.figure(figsize=(13, 5))
plt.plot(test, color="black", linewidth=2, label="реални")
plt.plot(naive, alpha=0.5, label="последна стойност")
plt.plot(moving_avg, label="пълзяща средна")
plt.plot(linear, label="линейна регресия")
plt.plot(lstm, label="LSTM")
plt.title("Реални и прогнозирани приходи (тестов период)")
plt.legend()
plt.show()

# запис за dashboard-а
forecast = pd.DataFrame({"real": test, "naive": naive, "moving_avg": moving_avg, "linear": linear, "lstm": lstm})
forecast.to_csv("data/clean/forecast.csv")
results.to_csv("data/clean/forecast_metrics.csv")
