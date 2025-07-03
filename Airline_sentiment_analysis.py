# Airline Sentiment Analysis - In-Class Project

import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

# Step 1: Read the training data using pandas and select only sentiment and text columns
train_df = pd.read_csv(r" ")
train_data = train_df[['airline_sentiment', 'text']].copy()

# Step 2: Observe randomly generated 10 tweets for each sentiment
sentiments = train_data['airline_sentiment'].unique()

for sentiment in sentiments:
    sentiment_tweets = train_data[train_data['airline_sentiment'] == sentiment]['text'].sample(10, random_state=42)
    print(f"\n{sentiment.upper()} SENTIMENT - 10 Random Tweets:")
    for i, tweet in enumerate(sentiment_tweets, 1):
        print(f"{i}. {tweet}")
    
    # Observe patterns
    references = sum(1 for tweet in sentiment_tweets if '@' in tweet)
    links = sum(1 for tweet in sentiment_tweets if 'http' in tweet.lower())
    punctuations = sum(1 for tweet in sentiment_tweets if any(p in tweet for p in string.punctuation))
    emoticons = sum(1 for tweet in sentiment_tweets if any(e in tweet for e in ['😊', '😢', '😡', '😀', '😞', '👍', '👎', ':)', ':(', ':D']))
    
    print(f"Observations: References(@): {references}, Links: {links}, Punctuations: {punctuations}, Emoticons: {emoticons}")

# Step 3: Function to clean all observed tokens from tweet text
def clean_tweet(text):
    if pd.isna(text):
        return ""
    
    text = str(text)
    # Remove references (@username)
    text = re.sub(r'@\w+', '', text)
    # Remove links
    text = re.sub(r'http\S+|https\S+', '', text)
    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove emoticons
    emoticons = ['😊', '😢', '😡', '😀', '😞', '👍', '👎', ':)', ':(', ':D', ':-)', ':-(']
    for emoticon in emoticons:
        text = text.replace(emoticon, '')
    # Remove extra whitespaces and convert to lowercase
    text = ' '.join(text.split()).lower()
    return text

# Apply cleaning and save in new column
train_data['cleaned_text'] = train_data['text'].apply(clean_tweet)

# Step 4: List down most common 15 words for each sentiment
def get_most_common_words(df, sentiment_col, text_col, sentiment, n_words=15):
    sentiment_text = df[df[sentiment_col] == sentiment][text_col]
    all_words = []
    for text in sentiment_text:
        if pd.notna(text):
            all_words.extend(str(text).split())
    return Counter(all_words).most_common(n_words)

print("\n\nMost common 15 words for each sentiment (after cleaning):")
for sentiment in sentiments:
    common_words = get_most_common_words(train_data, 'airline_sentiment', 'cleaned_text', sentiment)
    print(f"\n{sentiment.upper()}:")
    for word, count in common_words:
        print(f"{word}: {count}")

# Step 5: Remove stopwords and save in new column
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    if pd.isna(text):
        return ""
    words = str(text).split()
    return ' '.join([word for word in words if word.lower() not in stop_words])

train_data['text_no_stopwords'] = train_data['cleaned_text'].apply(remove_stopwords)

print("\n\nMost common 15 words after removing stopwords:")
for sentiment in sentiments:
    common_words = get_most_common_words(train_data, 'airline_sentiment', 'text_no_stopwords', sentiment)
    print(f"\n{sentiment.upper()}:")
    for word, count in common_words:
        print(f"{word}: {count}")

# Step 6: Remove airline and flight related words
airline_words = ['americanair', 'united', 'delta', 'southwestair', 'jetblue', 'virginamerica', 'usairways', 'flight', 'plane']

def remove_airline_words(text):
    if pd.isna(text):
        return ""
    words = str(text).split()
    return ' '.join([word for word in words if word.lower() not in airline_words])

train_data['final_text'] = train_data['text_no_stopwords'].apply(remove_airline_words)

print("\n\nMost common 15 words after removing airline/flight words:")
for sentiment in sentiments:
    common_words = get_most_common_words(train_data, 'airline_sentiment', 'final_text', sentiment)
    print(f"\n{sentiment.upper()}:")
    for word, count in common_words:
        print(f"{word}: {count}")

print("\nObservations:")
print("- Removing stopwords revealed more meaningful sentiment-specific words")
print("- Removing airline names helps focus on customer experience words")
print("- Positive tweets show appreciation words, negative show complaint words")

# Step 7: Encode sentiments using Label Encoder
label_encoder = LabelEncoder()
train_data['sentiment_encoded'] = label_encoder.fit_transform(train_data['airline_sentiment'])

# Step 8: Vectorize the text column
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data['final_text'])
y_train = train_data['sentiment_encoded']

# Step 9: Prepare multiclass classification model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 10: Read test data and carry out same operations
test_df = pd.read_csv(r" ")
test_data = test_df[['airline_sentiment', 'text']].copy()

# Apply same cleaning operations
test_data['cleaned_text'] = test_data['text'].apply(clean_tweet)
test_data['text_no_stopwords'] = test_data['cleaned_text'].apply(remove_stopwords)
test_data['final_text'] = test_data['text_no_stopwords'].apply(remove_airline_words)

# Encode test sentiments
test_data['sentiment_encoded'] = label_encoder.transform(test_data['airline_sentiment'])

# Vectorize test text
X_test = vectorizer.transform(test_data['final_text'])
y_test = test_data['sentiment_encoded']

# Step 11: Predict sentiments for test data
y_pred = model.predict(X_test)

# Step 12: Print and explain confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\n\nConfusion Matrix:")
print(cm)

# Create labeled confusion matrix
cm_df = pd.DataFrame(cm, 
                     index=[f'Actual {label}' for label in label_encoder.classes_],
                     columns=[f'Predicted {label}' for label in label_encoder.classes_])
print("\nLabeled Confusion Matrix:")
print(cm_df)

print("\nConfusion Matrix Explanation:")
print("- Diagonal elements show correct predictions")
print("- Off-diagonal elements show misclassifications")
print("- Each row represents actual sentiment, each column represents predicted sentiment")

# Step 13: Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")