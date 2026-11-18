import pandas as pd
import urllib.request
import zipfile
import io
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

response = urllib.request.urlopen(url)
z = zipfile.ZipFile(io.BytesIO(response.read()))

with z.open("SMSSpamCollection") as file:
    data = pd.read_csv(
        file,
        sep="\t",
        header=None,
        names=["label", "message"]
    )

X = data["message"]
y = data["label"]

model = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)

model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained successfully!")