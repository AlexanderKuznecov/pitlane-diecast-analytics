import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


# Етап 5. Анализ на клиентските мнения: почистване на текста, честоти, TF-IDF, VADER

nltk.download("vader_lexicon", quiet=True)

reviews = pd.read_csv("data/clean/reviews.csv", parse_dates=["review_date"])
products = pd.read_csv("data/products.csv")


# 1. Почистване: малки букви, без URL адреси, без цифри и символи
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(text.split())


# имената на продуктите не носят мнение
product_words = set(clean_text(" ".join(products["product_name"])).split())
stop_words = set(ENGLISH_STOP_WORDS) | product_words

reviews["clean_text"] = reviews["review_text"].apply(clean_text)
reviews["tokens"] = reviews["clean_text"].apply(lambda t: [w for w in t.split() if w not in stop_words])

print("-------------------------------")
print("1. Пример за почистване и токенизация:")
example = reviews[reviews["review_text"].str.contains("http|www")].iloc[0]
print(example["review_text"])
print(example["clean_text"])
print(example["tokens"])

# 2. Честоти на думи и n-грамове
word_freq = Counter(w for tokens in reviews["tokens"] for w in tokens).most_common(10)
words = pd.DataFrame(word_freq, columns=["word", "count"])

joined = reviews["tokens"].apply(" ".join)
cv = CountVectorizer(ngram_range=(2, 2))
bigram_matrix = cv.fit_transform(joined)
bigrams = pd.Series(bigram_matrix.sum(axis=0).A1, index=cv.get_feature_names_out()).sort_values(ascending=False).head(10)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(joined)
tfidf_words = pd.Series(tfidf_matrix.mean(axis=0).A1, index=tfidf.get_feature_names_out()).sort_values(ascending=False).head(10)

print("-------------------------------")
print("2. Най-чести думи:")
print(words)

print("-------------------------------")
print("3. Най-чести фрази (биграми, CountVectorizer):")
print(bigrams)

print("-------------------------------")
print("4. Най-важни думи по TF-IDF:")
print(tfidf_words.round(3))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].barh(words["word"], words["count"], color="darkorange")
axes[0].set_title("Най-чести думи")
axes[0].set_xlabel("Брой")
axes[1].barh(bigrams.index, bigrams.values, color="teal")
axes[1].set_title("Най-чести фрази")
axes[1].set_xlabel("Брой")
axes[2].barh(tfidf_words.index, tfidf_words.values, color="gray")
axes[2].set_title("TF-IDF")
axes[2].set_xlabel("Средна TF-IDF стойност")
plt.tight_layout()
plt.show()

# 3. Sentiment analysis с VADER
sia = SentimentIntensityAnalyzer()
reviews["compound"] = reviews["review_text"].apply(lambda t: sia.polarity_scores(t)["compound"])
reviews["sentiment"] = "Neutral"
reviews.loc[reviews["compound"] >= 0.05, "sentiment"] = "Positive"
reviews.loc[reviews["compound"] <= -0.05, "sentiment"] = "Negative"

shares = reviews["sentiment"].value_counts(normalize=True) * 100
reviews["month"] = reviews["review_date"].dt.to_period("M").astype(str)
by_month = pd.crosstab(reviews["month"], reviews["sentiment"], normalize="index") * 100

print("-------------------------------")
print("5. Дялове на sentiment категориите (%):")
print(shares.round(1))

print("-------------------------------")
print("6. Sentiment по месеци (%):")
print(by_month.round(1))

bg = {"Positive": "Положително", "Neutral": "Неутрално", "Negative": "Отрицателно"}
colors = {"Positive": "seagreen", "Neutral": "gray", "Negative": "firebrick"}

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
axes[0].pie(shares, labels=[bg[s] for s in shares.index], autopct="%.1f%%", colors=[colors[s] for s in shares.index])
axes[0].set_title("Дялове на мненията")
by_month.rename(columns=bg).plot(kind="bar", stacked=True, ax=axes[1],
                                 color=[colors[s] for s in by_month.columns])
axes[1].set_title("Sentiment по месеци")
axes[1].set_xlabel("Месец")
axes[1].set_ylabel("%")
axes[1].legend(title="")
plt.tight_layout()
plt.show()

# 4. Проверка спрямо оценката на клиента (4-5 положително, 3 неутрално, 1-2 отрицателно)
reviews["rating_label"] = "Neutral"
reviews.loc[reviews["rating"] >= 4, "rating_label"] = "Positive"
reviews.loc[reviews["rating"] <= 2, "rating_label"] = "Negative"
agreement = (reviews["sentiment"] == reviews["rating_label"]).mean() * 100

print("-------------------------------")
print("7. Съвпадение на VADER с оценката на клиента (%):")
print(round(agreement, 1))
print(pd.crosstab(reviews["rating_label"], reviews["sentiment"]))

# 5. Интерпретация и ограничения
worst_month = by_month["Negative"].idxmax()
print("-------------------------------")
print("8. Интерпретация и ограничения:")
print(f"Положителни са {shares['Positive']:.1f}%, а отрицателни {shares['Negative']:.1f}% от мненията.")
print(f"Най-много отрицателни мнения има през {worst_month} ({by_month['Negative'].max():.1f}%).")
if worst_month in ["2025-12", "2026-01"]:
    print("Това са мненията за поръчките от коледния сезон (забавени доставки).")
print(f"VADER съвпада с оценката на клиента в {agreement:.1f}% от случаите.")
print("Ограничения: VADER работи с речник на думите, не разбира сарказъм и контекст,")
print("а неутралните мнения често ги отчита като положителни или отрицателни.")

reviews.drop(columns=["tokens"]).to_csv("data/clean/reviews_sentiment.csv", index=False)
