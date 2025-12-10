import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Dataset
df = pd.read_csv("DATASET/titanic.csv")

# 2. Drop unnecessary columns
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# 3. Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 4. Encode categorical columns
label_cols = ['Sex', 'Embarked']
le = LabelEncoder()

for col in label_cols:
    df[col] = le.fit_transform(df[col])

# 5. Define features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 8. Predict
y_pred = model.predict(X_test)

# 9. Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 10. Predict custom passenger example
sample = X.iloc[[0]]  # one passenger
print("Predicted survival for first passenger:", model.predict(sample))
