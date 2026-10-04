
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("plots", exist_ok=True)
df = pd.read_csv("Titanic-Dataset.csv")

print("First 5 rows:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nDataset Information:")
df.info()


print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())
print("\nMedian:")
print(df.median(numeric_only=True))
print("\nStandard Deviation:")
print(df.std(numeric_only=True))

print("\nDuplicate Rows:", df.duplicated().sum())

# Select numeric columns (exclude PassengerId)
numeric_cols = df.select_dtypes(include="number").columns.tolist()
numeric_cols = [col for col in numeric_cols if col != "PassengerId"]

# Histograms
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"plots/{col}_histogram.png")
    plt.show()
    plt.close()

# Boxplots
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(f"plots/{col}_boxplot.png")
    plt.show()
    plt.close()

# Correlation matrix
correlation = df[numeric_cols].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("plots/correlation_matrix.png")
plt.show()
plt.close()

# Pairplot
pair_cols = ["Survived", "Pclass", "Age", "Fare"]
pair_cols = [col for col in pair_cols if col in df.columns]
if len(pair_cols) > 1:
    sns.pairplot(
        df[pair_cols].dropna().sample(
            min(300, len(df[pair_cols].dropna())),
            random_state=42
        ),
        hue="Survived" if "Survived" in pair_cols else None
    )
    plt.savefig("plots/pairplot.png")
    plt.show()
    plt.close()

# Categorical feature analysis
for col in ["Sex", "Pclass", "Embarked", "Survived"]:
    if col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=col)
        plt.title(f"Count of {col}")
        plt.tight_layout()
        plt.savefig(f"plots/{col}_countplot.png")
        plt.show()
        plt.close()

# Skewness
print("\nSkewness:")
print(df[numeric_cols].skew())

# Survival analysis
if "Survived" in df.columns:
    print("\nSurvival Rate:")
    print(df["Survived"].value_counts(normalize=True) * 100)
    if "Sex" in df.columns:
        print("\nSurvival Rate by Gender:")
        print(df.groupby("Sex")["Survived"].mean() * 100)
    if "Pclass" in df.columns:
        print("\nSurvival Rate by Passenger Class:")
        print(df.groupby("Pclass")["Survived"].mean() * 100)

# 1. Identify outliers using IQR
print("\n--- Outlier Detection ---")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} potential outliers")

# 2. Identify survival patterns
print("\n--- Feature-Level Inferences ---")
if "Survived" in df.columns and "Sex" in df.columns:
    gender_survival = df.groupby("Sex")["Survived"].mean() * 100
    print("\nSurvival rate by gender:")
    print(gender_survival.round(2))
    print("Higher survival rate:",
          gender_survival.idxmax())
if "Survived" in df.columns and "Pclass" in df.columns:
    class_survival = df.groupby("Pclass")["Survived"].mean() * 100
    print("\nSurvival rate by passenger class:")
    print(class_survival.round(2))
    print("Class with highest survival rate:",
          class_survival.idxmax())

# 3. Analyze age patterns
if "Age" in df.columns:
    print("\nAge distribution:")
    print(df["Age"].describe())
    if "Survived" in df.columns:
        df["Age_Group"] = pd.cut(
            df["Age"],
            bins=[0, 12, 18, 35, 60, 100],
            labels=["Child", "Teenager", "Adult",
                    "Middle-aged", "Senior"]
        )
        print("\nSurvival rate by age group (%):")
        print(
            (df.groupby("Age_Group", observed=False)["Survived"]
             .mean() * 100).round(2)
        )

# 4. Analyze fare by passenger class
if "Fare" in df.columns and "Pclass" in df.columns:
    print("\nAverage fare by passenger class:")
    print(df.groupby("Pclass")["Fare"].mean().round(2))

# 5. Correlation with survival
if "Survived" in df.columns:
    print("\nCorrelation with survival:")
    print(df[numeric_cols].corr()["Survived"].sort_values(
        ascending=False
    ).round(2))