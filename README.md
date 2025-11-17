📌 TikTok Viral Predictor — Machine Learning Mini Project

A simplified ML project demonstrating feature engineering, model training, and prediction workflow.

🌟 Overview

This project is a machine learning mini-project designed to predict whether a TikTok video has the potential to go viral based on basic engagement features such as views, likes, comments, and hashtag count.

It was created as part of my Professional Portfolio Artifact #3 for the AIML-500 Machine Learning Fundamentals course at IWU.
The goal is to demonstrate my ability to:

Apply ML concepts

Build a simple prediction pipeline

Communicate technical ideas clearly

Showcase growing competency in AI/ML development

🚀 Features

✔ Basic dataset preprocessing
✔ Exploratory data analysis (EDA)
✔ Feature engineering (hashtags_count, engagement_rate, etc.)
✔ Logistic Regression classifier
✔ Accuracy / F1-Score evaluation
✔ Clean code structure & modular workflow
✔ Easy to extend with real data

📂 Project Structure
📁 tiktok-viral-predictor/
│── data/
│   └── sample_tiktok_data.csv
│── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
│── notebooks/
│   └── exploratory-analysis.ipynb
│── README.md
│── requirements.txt

🔧 How It Works
1️⃣ Data Preprocessing

Clean missing values

Convert numerical types

Create derived features

Normalize inputs

Sample code:

df["engagement_rate"] = (df["likes"] + df["comments"]) / df["views"]

2️⃣ Model Training

We use Logistic Regression as a simple baseline classifier.

model = LogisticRegression()
model.fit(X_train, y_train)

3️⃣ Prediction

Input example:

sample = {
    "views": 12000,
    "likes": 1500,
    "comments": 120,
    "hashtags_count": 4
}


Output:

Prediction: VIRAL (1)

📊 Model Performance
Metric	Score
Accuracy	0.87
F1-Score	0.84

(Note: Sample metrics based on demonstration dataset.)

🧩 Why This Artifact Matters

This project showcases core ML skills:

✔ Understanding real-world problem framing
✔ Building a complete ML pipeline
✔ Interpreting model metrics
✔ Explaining ML work to a general audience
✔ Translating course concepts into practical output

It represents my progress in transitioning toward more advanced ML and AI engineering projects.

🛠 Tech Stack

Python

Pandas

Scikit-learn

NumPy

Matplotlib / Seaborn (optional)

📘 How to Run (Optional for Grading)
pip install -r requirements.txt
python src/train_model.py
python src/predict.py

📝 Reflection (for IWU Portfolio)

This project helped me understand how ML models are structured in real workflows. Before creating this artifact, I felt overwhelmed when looking at full ML pipelines. Breaking the project into small steps—preprocessing, feature engineering, training, evaluating—helped me gain clarity and confidence.

I learned that even a simple model can provide valuable insights when the data is well-prepared and the problem is defined clearly. This artifact demonstrates my ability to learn quickly, adapt ML concepts to practical use cases, and communicate technical work effectively to different audiences.
