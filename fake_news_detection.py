# Fake News Detection System
# By: Nasina Sriram

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load Dataset
true_news = pd.read_csv("D:/naman digital/True.csv")
fake_news = pd.read_csv("D:/naman digital/Fake.csv")

true_news["label"] = 1   # Real
fake_news["label"] = 0   # Fake

data = pd.concat([true_news, fake_news], axis=0)
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle

# Step 3: Clean Data
data = data.drop(["subject", "date"], axis=1)
data["content"] = data["title"] + " " + data["text"]
data = data.dropna()

# Step 4: Split Data
X = data["content"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature Extraction
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Model Building
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

# Step 7: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 8: Test with a Custom News
sample = ["The Prime Minister announced new healthcare reforms today."]
sample_vec = vectorizer.transform(sample)
pred = model.predict(sample_vec)

if pred == 1:
    print("✅ Real News")
else:
    print("🚨 Fake News")
