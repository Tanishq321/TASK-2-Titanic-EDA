# Titanic Data Cleaning & Exploratory Data Analysis (EDA)
# Task 2

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("train.csv")

print(f"🔹 Dataset Shape: {df.shape}")
print(f"🔹 Columns: {list(df.columns)}")

# -------------------------------
# 2. Data Cleaning
# -------------------------------

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Extract Titles from passenger names
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
df["Title"] = df["Title"].replace({
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"
})

# Keep Sex column as categorical (Male/Female) for clarity
# Also create a numeric column for correlation
df["Sex_Numeric"] = df["Sex"].map({"male": 0, "female": 1})

# Drop unnecessary columns
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

# -------------------------------
# 3. Exploratory Data Analysis (EDA)
# -------------------------------

# 1. Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df, hue="Survived", palette="Set2", legend=False)
plt.title("Survival Count\n(0 = Did Not Survive, 1 = Survived)")
plt.show()

# 2. Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df, palette="pastel")
plt.title("Survival by Gender\nBlue = Did Not Survive, Orange = Survived")
plt.show()

# 3. Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", hue="Survived", data=df, palette="coolwarm")
plt.title("Survival by Passenger Class\nBlue = Did Not Survive, Red = Survived")
plt.show()

# 4. Age Distribution with Survival
plt.figure(figsize=(8,5))
sns.histplot(df, x="Age", hue="Survived", bins=30, kde=True, palette="muted")
plt.title("Age Distribution by Survival\nOrange = Survived, Blue = Did Not Survive")
plt.show()

# 5. Fare Distribution with Survival
plt.figure(figsize=(8,5))
sns.boxplot(x="Survived", y="Fare", data=df, palette="Set3")
plt.title("Fare Distribution by Survival\n(0 = Did Not Survive, 1 = Survived)")
plt.show()

# 6. Survival by Title (social status)
plt.figure(figsize=(10,5))
sns.countplot(x="Title", hue="Survived", data=df, palette="Set1")
plt.title("Survival by Title\nDifferent colors = Survived/Did Not Survive")
plt.xticks(rotation=45)
plt.show()

# 7. Survival by Embarked Port
plt.figure(figsize=(6,4))
sns.countplot(x="Embarked", hue="Survived", data=df, palette="Set2")
plt.title("Survival by Port of Embarkation\nDifferent colors = Survived/Did Not Survive")
plt.show()

# 8. Correlation Heatmap (numeric only)
plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=["int64", "float64"]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Titanic Dataset\n(Values closer to +1 = Strong Positive, -1 = Strong Negative)")
plt.show()

# -------------------------------
# 4. Insights (Print Stats)
# -------------------------------
print("\n🔹 Average Age by Survival:")
print(df.groupby("Survived")["Age"].mean())

print("\n🔹 Survival Rate by Gender:")
print(df.groupby("Sex")["Survived"].mean())

print("\n🔹 Survival Rate by Class:")
print(df.groupby("Pclass")["Survived"].mean())

print("\n🔹 Survival Rate by Embarked Port:")
print(df.groupby("Embarked")["Survived"].mean())
