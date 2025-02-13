import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
import matplotlib.pyplot as plt

# Parameters
num_users = 1000
num_items = 500
num_interactions = 10000

# Generate users
def generate_users(num_users):
    users = []
    for user_id in range(1, num_users + 1):
        age = random.randint(18, 60)
        gender = random.choice(["Male", "Female", "Other"])
        location = random.choice(["Urban", "Suburban", "Rural"])
        users.append([user_id, age, gender, location])
    return pd.DataFrame(users, columns=["user_id", "age", "gender", "location"])

# Generate items
def generate_items(num_items):
    categories = ["Shirts", "Dresses", "Pants", "Shoes", "Accessories", "Outerwear", "Activewear", "Swimwear", "Underwear", "Bags"]
    colors = ["Red", "Blue", "Green", "Black", "White", "Yellow", "Pink", "Gray", "Purple", "Brown"]
    brands = ["Nike", "Adidas", "Zara", "H&M", "Gucci", "Prada", "Levi's", "Uniqlo", "Chanel", "Louis Vuitton"]
    items = []
    for item_id in range(1, num_items + 1):
        category = random.choice(categories)
        color = random.choice(colors)
        price = round(random.uniform(10, 1000), 2)
        brand = random.choice(brands)
        items.append([item_id, category, color, price, brand])
    return pd.DataFrame(items, columns=["item_id", "category", "color", "price", "brand"])

# Generate user-item interactions
def generate_interactions(num_users, num_items, num_interactions):
    interactions = []
    for _ in range(num_interactions):
        user_id = random.randint(1, num_users)
        item_id = random.randint(1, num_items)
        interaction_type = random.choice(["view", "click", "purchase"])
        timestamp = pd.Timestamp.now() - pd.to_timedelta(random.randint(0, 30 * 24 * 60 * 60), unit='s')
        interactions.append([user_id, item_id, interaction_type, timestamp])
    return pd.DataFrame(interactions, columns=["user_id", "item_id", "interaction_type", "timestamp"])

# Generate datasets
users_df = generate_users(num_users)
items_df = generate_items(num_items)
interactions_df = generate_interactions(num_users, num_items, num_interactions)

# Merge datasets for modeling
data = interactions_df.merge(users_df, on="user_id").merge(items_df, on="item_id")
data["interaction_type"] = data["interaction_type"].map({"view": 0, "click": 1, "purchase": 2})

# Features and Labels
X = pd.get_dummies(data.drop(columns=["interaction_type", "timestamp"]))
y = data["interaction_type"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Prepare data for LSTM
sequence_length = 5
sequences = []
labels = []
for user_id in data["user_id"].unique():
    user_data = data[data["user_id"] == user_id].sort_values("timestamp")
    user_sequences = user_data["item_id"].values
    for i in range(len(user_sequences) - sequence_length):
        sequences.append(user_sequences[i:i + sequence_length])
        labels.append(user_data["interaction_type"].values[i + sequence_length])
sequences = np.array(sequences)
labels = np.array(labels)

# Split LSTM data
X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# LSTM Model
lstm_model = Sequential([
    Embedding(input_dim=num_items + 1, output_dim=50, input_length=sequence_length),
    LSTM(128, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: view, click, purchase
])
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_lstm_train, y_lstm_train, epochs=5, batch_size=32, validation_split=0.2)

# Predictions
lstm_predictions = np.argmax(lstm_model.predict(X_lstm_test), axis=1)

# Evaluation
rf_accuracy = accuracy_score(y_test, rf_predictions)
lstm_accuracy = accuracy_score(y_lstm_test, lstm_predictions)

print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"LSTM Accuracy: {lstm_accuracy:.2f}")

# Visualization code
plt.figure(figsize=(10, 6))
plt.bar(["Random Forest", "LSTM"], [rf_accuracy, lstm_accuracy], color=["blue", "orange"])
plt.title("Comparative Accuracy of Random Forest and LSTM")
plt.ylabel("Accuracy")
plt.show()