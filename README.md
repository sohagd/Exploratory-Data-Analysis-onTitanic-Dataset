
# Task 2: Exploratory Data Analysis (EDA) using Python

## 1. Project Overview

The objective is to understand the dataset through statistical analysis, data visualization, pattern recognition, and anomaly detection.
The Titanic dataset is used to explore passenger information, identify relationships between different features, analyze survival patterns and detect potential outliers. The analysis is performed using Python libraries such as Pandas, Matplotlib, and Seaborn.

## 2. Objectives

- Understand the structure and characteristics of the dataset.
- Generate descriptive statistics such as mean, median, standard deviation, and quartiles.
- Identify missing values and duplicate records.
- Visualize numerical and categorical features.
- Analyze relationships and correlations between variables.
- Identify patterns, trends, and potential anomalies.
- Perform basic feature-level analysis and draw inferences from visualizations.

## 3. Dataset

**Dataset:** Titanic Dataset

The dataset contains passenger information from the Titanic, including demographic details, ticket class, fare and survival status.
Some of the important features include:

| Feature | Description |
|---|---|
| PassengerId | Unique passenger identification number |
| Survived | Survival status (0 = No, 1 = Yes) |
| Pclass | Passenger class (1, 2, or 3) |
| Name | Passenger's name |
| Sex | Passenger's gender |
| Age | Passenger's age |
| SibSp | Number of siblings or spouses aboard |
| Parch | Number of parents or children aboard |
| Ticket | Ticket number |
| Fare | Ticket fare |
| Cabin | Cabin number |
| Embarked | Port of embarkation |

## 4. Tools and Technologies

- **Python:** Programming language used for analysis.
- **Pandas:** Data loading, manipulation, and statistical analysis.
- **Matplotlib:** Creating charts and visualizations.
- **Seaborn:** Statistical visualization, correlation heatmaps, and pairplots.


## 5. Workflow

### Step 1: Importing Libraries
Imported Pandas, Matplotlib, Seaborn and OS for data analysis and visualization.

### Step 2: Loading the Dataset
Loaded the Titanic dataset using `read_csv()` and displayed the first five rows.

### Step 3: Dataset Exploration
Used `head()`, `shape`, `info()`, and `describe()` to understand the dataset structure, dimensions, data types and statistics.

### Step 4: Missing Values and Duplicates
Identified missing values using `isnull().sum()` and duplicate records using `duplicated().sum()`. Missing values were retained in the original dataset.

### Step 5: Statistical Analysis
Calculated mean, median, standard deviation, quartiles, and skewness to understand feature distributions and variability.

### Step 6: Data Visualization

Created visualizations to explore the dataset and identify meaningful patterns.
**Histograms**
Histograms were generated for numerical features to understand their distributions, frequency patterns, and possible skewness.
**Boxplots**
Boxplots were created to examine the spread of numerical data, identify potential outliers, and understand the distribution of values.
**Correlation Heatmap**
A correlation matrix was generated using Seaborn to examine relationships between numerical features and identify variables that may be associated with survival.
**Pairplot**
A pairplot was created using selected features such as Survived, Pclass, Age, and Fare to visualize relationships between multiple numerical variables.
**Countplots**
Countplots were generated for categorical features such as Sex, Pclass, Embarked, and Survived to understand their frequency distributions.

All generated visualizations were saved in the `plots` directory.

### Step 7: Pattern and Trend Identification

Explored the dataset to identify patterns and relationships between passenger characteristics and survival status.
The analysis included:
- Comparing survival rates between male and female passengers.
- Examining survival rates across passenger classes.
- Exploring age distribution and survival patterns across age groups.
- Comparing average fares across passenger classes.
- Examining correlations between numerical features.

These analyses help identify associations and trends in the dataset.

### Step 8: Anomaly and Outlier Detection

Used the Interquartile Range (IQR) method to identify potential outliers in numerical features.

The IQR is calculated as:
IQR = Q3 - Q1

Where:
- Q1 is the first quartile.
- Q3 is the third quartile.

The lower and upper boundaries are calculated as:
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR

Values outside these boundaries are flagged as potential outliers. Boxplots are also used to visualize these observations.
Potential outliers were identified for analysis and were not automatically removed.

### Step 9: Feature-Level Inferences

Performed feature-level analysis to understand how individual variables relate to survival.
- **Sex:** Compared survival rates across gender categories.
- **Pclass:** Examined the relationship between passenger class and survival.
- **Age:** Analyzed age distribution and survival rates across age groups.
- **Fare:** Examined fare distribution and compared average fares across passenger classes.
- **Survived:** Analyzed the distribution of survival outcomes.

The results from statistical analysis and visualizations were used to draw basic inferences about the dataset.

## 6. Key Findings

The visualizations and statistical summaries provide a way to investigate:
- The distribution and variability of passenger ages and fares.
- Missing data in selected features.
- Potential outliers in numerical variables.
- Differences in survival rates across gender and passenger classes.
- Relationships between numerical features.
- Possible associations between passenger characteristics and survival.

The exact observations can be interpreted from the generated plots and the statistical results.
The analysis demonstrates how exploratory data analysis can be used to understand data quality, discover relationships, and identify areas that may require further preprocessing before machine learning.