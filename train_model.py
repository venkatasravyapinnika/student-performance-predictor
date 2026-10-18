from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# 1. Define project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "student-mat.csv"
MODEL_FILE = BASE_DIR / "student_model.pkl"


# 2. Load the dataset
df = pd.read_csv(DATA_FILE, sep=";")
print("Dataset loaded:", df.shape)


# 3. Select features and target
features = [
    "G1",
    "G2",
    "studytime",
    "failures",
    "absences",
    "age"
]

X = df[features].copy()
y = df["G3"]


# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 5. Train the model
model = LinearRegression()
model.fit(X_train, y_train)


# 6. Evaluate the model
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))


# 7. Save the trained model
joblib.dump(model, MODEL_FILE)

print("Model saved successfully:", MODEL_FILE)