✈️ Airline Sentiment Analysis – NLP & Classification Project
This project performs end-to-end sentiment analysis on airline-related tweets using classical Natural Language Processing (NLP) and a Naive Bayes classifier. The objective is to clean, analyze, and classify tweets into positive, neutral, or negative sentiments based on customer feedback.

🧠 Objective
  •	Analyze public sentiment towards different airlines using Twitter data
  •	Clean and preprocess noisy text (mentions, links, emojis, etc.)
  •	Build a TF-IDF-based text classifier to predict sentiments
  •	Evaluate model performance using confusion matrix and accuracy

📌 Key Features
  •	Random tweet sampling for insight into raw sentiment distributions
  •	Custom text cleaning pipeline:
  o	Removes mentions, URLs, punctuation, emojis, airline names
  o	Strips stopwords and irrelevant tokens
  •	Common word frequency analysis per sentiment
  •	TF-IDF vectorization of cleaned text
  •	Multinomial Naive Bayes model for classification
  •	Confusion matrix and accuracy-based evaluation

🛠️ Tech Stack
•	Language: Python
•	Libraries:
o	pandas, numpy, re, string, collections
o	nltk (stopwords)
o	scikit-learn for preprocessing, modeling, and evaluation

