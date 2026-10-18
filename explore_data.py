
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "student-mat.csv"

df = pd.read_csv(INPUT_FILE, sep=";")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print(df.columns)

for col in df.columns:
    missing_count = df[col].isnull().sum()

    if missing_count > 0:
        print(col, ":", missing_count, "missing values")
    else:
        print(col, ": No missing values")

print(df.describe())
print(df['G3'].mean())
print(df['G3'].max())
print(df['G3'].min())

print("\nCorrelation with final grade (G3):")

correlation = df.corr(numeric_only=True)["G3"].sort_values(
    ascending=False
)

print(correlation)

print('\nRemoving the duplicte values:')
duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    print("Duplicate rows:", duplicate_count)
else:
    print("No duplicate rows")

print("\nData types:")
print(df.dtypes)

# Select features (X) and target (y)

features = [
    "G1",
    "G2",
    "studytime",
    "failures",
    "absences",
    "age"
    
]

X = df[features]
y = df["G3"]

print("Selected features:")
print(X.head())

print("\nTarget (G3):")
print(y.head())

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

X["schoolsup"] = X["schoolsup"].map({"yes": 1, "no": 0})
X["famsup"] = X["famsup"].map({"yes": 1, "no": 0})
print(X[["schoolsup", "famsup"]].head())
print(X.dtypes)

print(X.isnull().sum())
print(y.isnull().sum())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted grades:", y_pred[:10])
print("Actual grades:", y_test.iloc[:10].values)

from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

import pandas as pd

results = pd.DataFrame({
    "Actual Grade": y_test.values,
    "Predicted Grade": y_pred.round(2)
})

print(results)

results["Error"] = (
    results["Actual Grade"] - results["Predicted Grade"]
)

print(results.head(10))

results.to_csv("prediction_results.csv", index=False)

print("Prediction results saved successfully!")