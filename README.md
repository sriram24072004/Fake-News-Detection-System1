# 🧠 Fake News Detection System

## 📋 Project Overview
The **Fake News Detection System** is a Machine Learning project that classifies news articles as *Real* or *Fake* using Natural Language Processing (NLP) techniques.  
The model analyzes the text of news headlines and content to identify misleading or false information.

---

## 🎯 Objective
To build an automated system that can detect and flag fake news using text-based data analysis and supervised machine learning.

---

## ⚙️ Technologies Used
- **Programming Language:** Python  
- **Libraries:**  
  - Pandas  
  - NumPy  
  - Scikit-learn  
  - NLTK  
  - Matplotlib / Seaborn  
- **Algorithm:** Logistic Regression (can be extended to Random Forest or BERT)  
- **Environment:** Jupyter Notebook / Google Colab / VS Code

---

## 🧩 Dataset
Dataset used: **Fake and Real News Dataset** from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

| File Name | Description |
|------------|--------------|
| `True.csv` | Contains real news articles |
| `Fake.csv` | Contains fake news articles |

Each file includes:
- `title`: The headline of the news article  
- `text`: The main content of the article  
- `subject`: Category (politics, world news, etc.)  
- `date`: Date of publication  

---

## 🚀 Implementation Steps

1. **Import Libraries**  
   Import necessary Python libraries for data preprocessing and model training.

2. **Load Dataset**  
   Combine `True.csv` and `Fake.csv` into a single dataframe.

3. **Data Cleaning**  
   Remove unnecessary columns, handle missing values, and merge `title` + `text`.

4. **Text Preprocessing**  
   - Tokenization  
   - Stopword removal  
   - Lemmatization (optional)

5. **Feature Extraction**  
   Convert text data into numeric form using **TF-IDF Vectorizer**.

6. **Model Building**  
   Train a **Logistic Regression** classifier to distinguish between real and fake news.

7. **Evaluation**  
   Measure model performance using accuracy, precision, recall, F1-score, and confusion matrix.

8. **Prediction**  
   Test the model with custom news headlines or articles.

---

## 📊 Example Output

Accuracy: 0.9867483296213808

Classification Report:
               precision    recall  f1-score   support

           0       0.99      0.99      0.99      4673
           1       0.99      0.99      0.99      4307

    accuracy                           0.99      8980
   macro avg       0.99      0.99      0.99      8980
weighted avg       0.99      0.99      0.99      8980
