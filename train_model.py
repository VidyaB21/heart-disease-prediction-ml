import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("heart.csv")

# Split features and target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# One-hot encode categorical features
X = pd.get_dummies(X)

# Save column names
columns = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Save everything properly
joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "heart_scaler.pkl")
joblib.dump(columns, "heart_columns.pkl")

print("MODEL SAVED SUCCESSFULLY")