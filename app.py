import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify
import joblib



# Step 1: Data Loading and Preprocessing

# Load the dataset
df = pd.read_csv("sample_data.csv")

# Preprocessing (if needed)
# ...
df['text'] = df['text'].fillna('')
df['text'] = df['text'].astype(str)
df = df[df['text'] != '']


# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Step 2: Model Training

# Vectorize the text using TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)

train_labels = train_labels.fillna('')
# Train a logistic regression model

train_labels = train_labels.astype(str)
test_labels = test_labels.astype(str)

model = LogisticRegression()
model.fit(X_train, train_labels)

# Save the trained model and vectorizer
joblib.dump(model, "text_classifier_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")

# Step 3: Model Evaluation

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy}")

classification_report_result = classification_report(test_labels, predictions,zero_division=1)
print("Classification Report:")
print(classification_report_result)


app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("text_classifier_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']

    # Vectorize the input text
    text_vectorized = vectorizer.transform(pd.Series(text))

    # Make predictions
    prediction = model.predict(text_vectorized)

    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
