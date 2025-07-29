import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

# ========== TRAINING SECTION ==========

# Load data
df = pd.read_csv("flight_data.csv")

# Select features
features = ['source_city', 'destination_city', 'class', 'days_left']
target = 'price'
df = df[features + [target]]

# Encode categories
le_source = LabelEncoder()
le_dest = LabelEncoder()
le_class = LabelEncoder()

df['source_city'] = le_source.fit_transform(df['source_city'])
df['destination_city'] = le_dest.fit_transform(df['destination_city'])
df['class'] = le_class.fit_transform(df['class'])

# Save encoders
with open('encoders.pkl', 'wb') as f:
    pickle.dump((le_source, le_dest, le_class), f)

# Prepare data
X = df[features]
y = df[target]

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

import matplotlib.pyplot as plt
import seaborn as sns

# === Feature Importance Plot ===
importances = model.feature_importances_
feature_names = ['Source City', 'Destination City', 'Class', 'Days Left']
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feature_names, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# === Prediction vs Actual Scatter Plot ===
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Flight Prices')
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

# === Residual Plot ===
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True, color='coral')
plt.title('Residual Distribution')
plt.xlabel('Prediction Error (Residual)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("residuals.png")
plt.show()

# === Histogram of Predicted Prices ===
plt.figure(figsize=(8, 5))
sns.histplot(y_pred, bins=30, kde=True, color='steelblue')
plt.title('Distribution of Predicted Prices')
plt.xlabel('Predicted Price (₹)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("predicted_distribution.png")
plt.show()


print("\n✅ Model trained and evaluated:")
print(f"📊 MAE       = ₹{mae:.2f}")
print(f"📊 RMSE      = ₹{rmse:.2f}")
print(f"📊 R² Score  = {r2:.4f}")

# Save model
with open('flight_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# ========== PREDICTION SECTION ==========

# Load model and encoders
with open('flight_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    le_source, le_dest, le_class = pickle.load(f)

from fuzzywuzzy import process

# Unique values from training data
all_source_cities = le_source.classes_
all_dest_cities = le_dest.classes_
all_classes = le_class.classes_

def match_input(user_input, choices, label):
    match, score = process.extractOne(user_input, choices)
    if score < 60:
        print(f"❌ '{user_input}' doesn't match any known {label}. Try again.")
        exit()
    print(f"🔍 Interpreted '{user_input}' as '{match}' (match score: {score})")
    return match

# User Input with fuzzy match
raw_source = input("\nEnter source city: ").strip()
raw_dest = input("Enter destination city: ").strip()
raw_class = input("Enter class (Economy/Business): ").strip()
days_left = int(input("Enter days left for flight: "))

# Match using fuzzy logic
source = match_input(raw_source, all_source_cities, "source city")
dest = match_input(raw_dest, all_dest_cities, "destination city")
cls = match_input(raw_class, all_classes, "class")

# Encode input
source_encoded = le_source.transform([source])[0]
dest_encoded = le_dest.transform([dest])[0]
class_encoded = le_class.transform([cls])[0]


# Encode input
try:
    source_encoded = le_source.transform([source])[0]
    dest_encoded = le_dest.transform([dest])[0]
    class_encoded = le_class.transform([cls])[0]
except ValueError:
    print("❌ Input value not seen during training.")
    exit()

# Predict
features = [[source_encoded, dest_encoded, class_encoded, days_left]]
price = model.predict(features)[0]
print(f"\n💰 Predicted Flight Price: ₹{int(price)}")
