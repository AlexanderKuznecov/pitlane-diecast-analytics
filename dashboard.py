# Етап 7 и 8. Интерактивно табло (Streamlit) и изводи с LLM
import os
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()
st.set_page_config(page_title="PitLane Diecast - анализ", layout="wide")

# ============================================================
# Зареждане на резултатите от етапи 2-6
# ============================================================
sales = pd.read_csv("data/clean/sales.csv", parse_dates=["order_date"])
segments = pd.read_csv("data/clean/customer_segments.csv")
reviews = pd.read_csv("data/clean/reviews_sentiment.csv", parse_dates=["review_date"])
forecast = pd.read_csv("data/clean/forecast.csv", parse_dates=["order_date"], index_col="order_date")
metrics = pd.read_csv("data/clean/forecast_metrics.csv", index_col=0)

sales = sales[sales["status"] == "Delivered"]
sales["profit"] = sales["line_total"] - sales["quantity"] * sales["cost"]
sales = sales.merge(segments[["customer_id", "segment"]], on="customer_id")
reviews = reviews.merge(segments[["customer_id", "segment"]], on="customer_id")

st.title("PitLane Diecast: анализ на продажбите на Hot Wheels и F1 модели")

# ============================================================
# Етап 7. Филтри
# ============================================================
st.sidebar.header("Филтри")
period = st.sidebar.date_input("Период", value=(sales["order_date"].min(), sales["order_date"].max()))
if len(period) < 2:
    st.stop()   # докато не е избрана и крайна дата
start, end = period
categories = st.sidebar.multiselect("Категория", sales["category"].unique(), default=sales["category"].unique())
chosen = st.sidebar.multiselect("Сегмент", segments["segment"].unique(), default=segments["segment"].unique())

data = sales[sales["order_date"].between(pd.Timestamp(start), pd.Timestamp(end))
             & sales["category"].isin(categories) & sales["segment"].isin(chosen)]
rev = reviews[reviews["review_date"].between(pd.Timestamp(start), pd.Timestamp(end)) & reviews["segment"].isin(chosen)]

if data.empty:
    st.warning("Няма данни за избраните филтри.")
    st.stop()

# ============================================================
# KPI
# ============================================================
revenue = data["line_total"].sum()
profit = data["profit"].sum()
n_orders = data["order_id"].nunique()

# ръст: последните 30 дни от периода спрямо предишните 30
last_day = data["order_date"].max()
last_30 = data[data["order_date"] > last_day - pd.Timedelta(days=30)]["line_total"].sum()
prev_30 = data[(data["order_date"] <= last_day - pd.Timedelta(days=30))
               & (data["order_date"] > last_day - pd.Timedelta(days=60))]["line_total"].sum()
growth = (last_30 - prev_30) / prev_30 * 100 if prev_30 > 0 else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Общи приходи", f"{revenue:,.0f} €")
c2.metric("Печалба", f"{profit:,.0f} €", f"марж {profit / revenue * 100:.1f}%", delta_color="off")
c3.metric("Поръчки", n_orders)
c4.metric("Клиенти", data["customer_id"].nunique())
c5.metric("Средна поръчка", f"{revenue / n_orders:.2f} €")
c6.metric("Ръст (последни 30 дни)", f"{growth:+.1f}%")

# ============================================================
# Графики
# ============================================================
left, right = st.columns(2)

monthly = data.groupby(data["order_date"].dt.to_period("M").astype(str))[["line_total", "profit"]].sum()
fig = px.line(monthly, markers=True, title="Приходи и печалба по месеци",
              labels={"order_date": "Месец", "value": "EUR", "variable": ""})
fig.for_each_trace(lambda t: t.update(name={"line_total": "Приходи", "profit": "Печалба"}[t.name]))
left.plotly_chart(fig)

by_cat = data.groupby("category")[["line_total", "profit"]].sum().sort_values("profit")
fig = px.bar(by_cat, orientation="h", barmode="group", title="Приходи и печалба по категории",
             labels={"category": "", "value": "EUR", "variable": ""})
fig.for_each_trace(lambda t: t.update(name={"line_total": "Приходи", "profit": "Печалба"}[t.name]))
right.plotly_chart(fig)

left, right = st.columns(2)

seg_count = segments[segments["customer_id"].isin(data["customer_id"])]["segment"].value_counts()
left.plotly_chart(px.pie(names=seg_count.index, values=seg_count.values, hole=0.4,
                         title="Разпределение на клиентските сегменти"))

bg = {"Positive": "Положително", "Neutral": "Неутрално", "Negative": "Отрицателно"}
sentiment = rev["sentiment"].map(bg).value_counts()
right.plotly_chart(px.pie(names=sentiment.index, values=sentiment.values, hole=0.4,
                          title=f"Sentiment на мненията ({len(rev)} мнения)",
                          color=sentiment.index, color_discrete_map={"Положително": "seagreen",
                                                                     "Неутрално": "gray", "Отрицателно": "firebrick"}))

fig = px.line(forecast.rename(columns={"real": "Реални", "naive": "Последна стойност", "moving_avg": "Пълзяща средна",
                                       "linear": "Линейна регресия", "lstm": "LSTM"}),
              title="Реални и прогнозирани продажби (тестов период)",
              labels={"order_date": "Дата", "value": "Дневен приход (EUR)", "variable": ""})
fig.update_traces(line=dict(color="black", width=3), selector=dict(name="Реални"))
st.plotly_chart(fig)
st.dataframe(metrics)

# ============================================================
# Етап 8. Изводи и препоръки с LLM (Google Gemini)
# ============================================================
st.header("Изводи и препоръки (Gemini)")

seg_revenue = data.groupby("segment")["line_total"].sum().sort_values(ascending=False)
cat_profit = data.groupby("category")["profit"].sum().sort_values(ascending=False)
sent_share = (rev["sentiment"].map(bg).value_counts(normalize=True) * 100).round(1)
month_revenue = data.groupby(data["order_date"].dt.to_period("M").astype(str))["line_total"].sum()
cat_margin = (cat_profit / data.groupby("category")["line_total"].sum() * 100).round(1)
negative_by_month = (rev.groupby("month")["sentiment"].apply(lambda x: (x == "Negative").mean() * 100)).round(1)

# сегментите: брой клиенти, поръчки, средна поръчка и дял от приходите
seg_table = segments[segments["customer_id"].isin(data["customer_id"])].groupby("segment").agg(
    клиенти=("customer_id", "count"), средно_поръчки=("frequency", "mean"), средна_поръчка_eur=("avg_order_value", "mean"))
seg_table["дял_приходи_%"] = (seg_revenue / seg_revenue.sum() * 100)
seg_table = seg_table.round(1)

# най-слабо оценени продукти (поне 8 мнения)
names = sales.drop_duplicates("product_id").set_index("product_id")["product_name"]
ratings = rev.groupby(rev["product_id"].map(names))["rating"].agg(["mean", "count"])
weak_products = ratings[ratings["count"] >= 8]["mean"].sort_values().head(3).round(2)

summary = {
    "приходи_eur": round(revenue, 2),
    "печалба_eur": round(profit, 2),
    "поръчки": int(n_orders),
    "ръст_последни_30_дни_%": round(growth, 1),
    "печалба_по_категории": cat_profit.round(2).to_dict(),
    "приходи_по_сегменти": seg_revenue.round(2).to_dict(),
    "приходи_по_месеци": month_revenue.round(2).to_dict(),
    "марж_по_категории_%": cat_margin.to_dict(),
    "сегменти": seg_table.to_dict(orient="index"),
    "sentiment_%": sent_share.to_dict(),
    "отрицателни_мнения_по_месеци_%": negative_by_month.to_dict(),
    "най_слабо_оценени_продукти": weak_products.to_dict(),
    "грешка_на_прогнозните_модели_mae": metrics["MAE"].to_dict(),
}

# реалните изводи, с които проверяваме отговора на модела
facts = {
    "Най-печеливша категория": cat_profit.index[0],
    "Сегмент с най-много приходи": seg_revenue.index[0],
    "Най-силен месец": month_revenue.idxmax(),
    "Най-слаб месец": month_revenue.idxmin(),
    # при кратък период може да няма достатъчно мнения
    "Преобладаващ sentiment": sent_share.idxmax() if len(sent_share) > 0 else "няма мнения",
    "Месец с най-много отрицателни мнения": negative_by_month.idxmax() if len(negative_by_month) > 0 else "няма мнения",
    "Най-слабо оценен продукт": weak_products.index[0] if len(weak_products) > 0 else "няма достатъчно мнения",
    "Най-точен прогнозен модел": metrics["MAE"].idxmin(),
}

# изводите от изчисленията се показват винаги, дори ако LLM-ът не отговори
check = pd.DataFrame({"Реална стойност": facts.values()}, index=facts.keys())
st.subheader("Изводи от изчисленията")
st.table(check)

with st.expander("Какви данни се изпращат към модела"):
    st.json(summary)

st.caption("Към модела се изпращат само изчислените показатели от таблото, без лични данни на клиентите. "
           "Моделът не смята нищо сам, а само интерпретира числата. След отговора проверяваме дали основните "
           "му изводи съвпадат с реално изчислените стойности.")

if st.button("Генерирай изводи и препоръки"):
    prompt = ("Ти си бизнес анализатор на онлайн магазин за Hot Wheels и F1 умалени модели. "
              "Ето изчислените резултати:\n" + json.dumps(summary, ensure_ascii=False) +
              "\n\nНапиши отговора изцяло на български (до 350 думи) в два раздела:\n"
              "1. Изводи - 5 точки, всяка с конкретно число от данните: най-печеливша категория и марж, "
              "сегментът с най-много приходи и кой сегмент заслужава допълнително внимание, най-силен и най-слаб "
              "месец (кога има спад), какво показват мненията (преобладаващ sentiment, месецът с най-много "
              "отрицателни мнения, най-слабо оцененият продукт), най-точният прогнозен модел.\n"
              "2. Препоръки - 4 конкретни бизнес действия, всяко свързано с някой от изводите.\n"
              "Използвай само тези числа и не измисляй нови. Имената на категориите, сегментите, продуктите, "
              "месеците (във формат 2025-12) и моделите пиши точно както са в данните.")
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("Липсва GEMINI_API_KEY във файла .env!")
            st.stop()

        client = genai.Client(api_key=api_key)
        with st.spinner("Моделът анализира..."):
            response = client.models.generate_content_stream(
                model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
                contents=prompt,
            )
            answer = st.write_stream(chunk.text for chunk in response if chunk.text)

        # за sentiment търсим по корен на думата ("положи" хваща и "положителни")
        found = []
        for name, value in facts.items():
            word = value.lower()[:6] if name == "Преобладаващ sentiment" else value.lower()
            if value.startswith("няма"):
                found.append("- няма какво да се провери")
            else:
                found.append("✅ да" if word in answer.lower() else "❌ не")
        check["Съвпада ли с LLM"] = found
        st.subheader("Проверка на отговора спрямо изчисленията")
        st.table(check)
    except Exception as e:
        st.error(f"Грешка при заявката към Gemini: {e}")