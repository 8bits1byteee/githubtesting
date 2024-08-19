import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Step 1: Read the Data
data = pd.read_csv('C:\\Users\\Client\\Desktop\\socialmedia-disaster-tweets-DFE-3.csv', encoding='ISO-8859-1')

# Step 2: Preprocess the Data
data = data.dropna(subset=['text'])

# Define disaster-related keywords and categories
disaster_keywords = {
    'earthquake': ['earthquake'],
    'flood': ['flood'],
    'wildfire': ['wildfire'],
    'hurricane': ['hurricane'],
    'tornado': ['tornado'],
    'tsunami': ['tsunami']
}

severity_keywords = {
    'high': ['severe', 'catastrophic', 'extreme'],
    'medium': ['moderate', 'significant', 'intense'],
    'low': ['minor', 'small', 'light']
}

def categorize_disaster(text):
    """
    Categorize the tweet based on disaster-related keywords.
    """
    text = text.lower()
    for disaster, keywords in disaster_keywords.items():
        if any(keyword in text for keyword in keywords):
            return disaster
    return 'unknown'  # If no keywords match, classify as 'unknown'

def determine_severity(text):
    """
    Determine the severity of the disaster based on keywords.
    """
    text = text.lower()
    for severity, keywords in severity_keywords.items():
        if any(keyword in text for keyword in keywords):
            return severity
    return 'unknown'  # If no keywords match, classify as 'unknown'

# Apply categorization and severity determination
data['disaster_type'] = data['text'].apply(categorize_disaster)
data['severity'] = data['text'].apply(determine_severity)

# Filter out 'unknown' categories for training
train_data = data[data['disaster_type'] != 'unknown']

# Features and labels
X = train_data['text']
y = train_data['disaster_type']

# Split the data into training, evaluation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train a Model
model = Pipeline([
    ('vectorizer', CountVectorizer()),  # Convert text to numerical features
    ('classifier', MultinomialNB())  # Naive Bayes classifier
])

model.fit(X_train, y_train)

# Evaluate the Model on the evaluation set
y_eval_pred = model.predict(X_eval)
print("Evaluation Set Performance:")
print(classification_report(y_eval, y_eval_pred))

# Final Evaluation on the Test Set
y_test_pred = model.predict(X_test)
print("Test Set Performance:")
print(classification_report(y_test, y_test_pred))

# Apply the model to relevant tweets
relevant_tweets = data[data['disaster_type'] != 'unknown'].copy()
relevant_tweets['predicted_disaster'] = model.predict(relevant_tweets['text'])

# Adding severity information
relevant_tweets['predicted_severity'] = relevant_tweets['text'].apply(determine_severity)

# Optional: Save the classified tweets to a new CSV file
relevant_tweets.to_csv('classified_relevant_tweets_with_severity.csv', index=False)

print(f'Number of relevant tweets: {len(relevant_tweets)}')
print('Sample of classified tweets with severity:')
print(relevant_tweets[['text', 'predicted_disaster', 'predicted_severity']].head())
