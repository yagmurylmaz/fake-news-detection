import pandas as pd

true = pd.read_csv("dataset/True.csv")
true["label"] = "REAL"
true["text"] = true["title"].fillna("") + " " + true["text"].fillna("")

# news.csv henuz birlestirilmemisse (label yok) sahte haber kaynagi olarak kullan
news = pd.read_csv("dataset/news.csv")
if "label" not in news.columns:
    fake = news.copy()
    fake["label"] = "FAKE"
    fake["text"] = fake["title"].fillna("") + " " + fake["text"].fillna("")
    df = pd.concat([fake[["text", "label"]], true[["text", "label"]]], ignore_index=True)
else:
    # Zaten birlestirilmis dosyada tekrar calistirildiysa True.csv'den duzelt
    true_texts = set(true["text"])
    fake = news[~news["text"].isin(true_texts)][["text"]].drop_duplicates()
    fake["label"] = "FAKE"
    df = pd.concat([fake, true[["text", "label"]]], ignore_index=True)

df = df[["text", "label"]]
df.to_csv("dataset/news.csv", index=False)

print(df.head())
print("Toplam satir:", len(df))
print(df["label"].value_counts())
