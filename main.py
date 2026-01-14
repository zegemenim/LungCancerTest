import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns

df = pd.read_csv("SurveyLungCancer.csv")

print(df.head())
print(df.info())
print(df.describe())
df.GENDER = df.GENDER.map({"M": 1, "F": 0})
df.LUNG_CANCER = df.LUNG_CANCER.map({"YES": 1, "NO": 0})
print(df.isnull().sum())
# Remove gaps in end of column names
df.columns = df.columns.str.strip()
print(df.columns)

forChange = [
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL CONSUMING",
    "COUGHING",
    "SHORTNESS OF BREATH",
    "SWALLOWING DIFFICULTY",
    "CHEST PAIN",
]

for col in forChange:
    df[col] = df[col].map({2: 1, 1: 0})
print(df.head())

print(df.corr())
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Best model evaluation
from sklearn.metrics import classification_report, confusion_matrix
from lazypredict.Supervised import LazyClassifier

lazyreg = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = lazyreg.fit(X_train, X_test, y_train, y_test)
print(models)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))
print(df.LUNG_CANCER.value_counts())

# Predicting a new sample
gender = input("Enter Gender (M/F): ")
if gender == "M":
    gender = 1
else:
    gender = 0
age = int(input("Enter Age: "))
smoking = int(input("Do you smoke? (1 for Yes, 0 for No): "))
yellow_fingers = int(input("Do you have yellow fingers? (1 for Yes, 0 for No): "))
anxiety = int(input("Do you have anxiety? (1 for Yes, 0 for No): "))
peer_pressure = int(input("Do you experience peer pressure? (1 for Yes, 0 for No): "))
chronic_disease = int(input("Do you have a chronic disease? (1 for Yes, 0 for No): "))
fatigue = int(input("Do you experience fatigue? (1 for Yes, 0 for No): "))
allergy = int(input("Do you have allergies? (1 for Yes, 0 for No): "))
wheezing = int(input("Do you experience wheezing? (1 for Yes, 0 for No): "))
alcohol_consuming = int(input("Do you consume alcohol? (1 for Yes, 0 for No): "))
coughing = int(input("Do you have a coughing issue? (1 for Yes, 0 for No): "))
shortness_of_breath = int(
    input("Do you experience shortness of breath? (1 for Yes, 0 for No): ")
)
swallowing_difficulty = int(
    input("Do you have difficulty swallowing? (1 for Yes, 0 for No): ")
)
chest_pain = int(input("Do you experience chest pain? (1 for Yes, 0 for No): "))
new_sample = np.array(
    [
        gender,
        age,
        smoking,
        yellow_fingers,
        anxiety,
        peer_pressure,
        chronic_disease,
        fatigue,
        allergy,
        wheezing,
        alcohol_consuming,
        coughing,
        shortness_of_breath,
        swallowing_difficulty,
        chest_pain,
    ]
).reshape(1, -1)
prediction = logreg.predict(new_sample)
if prediction[0] == 1:
    print("The model predicts that the person has lung cancer.")
else:
    print("The model predicts that the person does not have lung cancer.")
