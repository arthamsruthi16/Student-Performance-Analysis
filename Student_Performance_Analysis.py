#---------------------------------STUDENT PERFORMANCE ANALYSIS----------------------------
#---------------------------DATA PREPARATION-----------------------------
#-------EXPLORATORY DATA ANALYSIS--------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engine.outliers import Winsorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#----------------------READ DATA----------------------
std_data = pd.read_csv(r"C:/Excel/student_performance.csv")
print(std_data.shape)
print(std_data.info())
print(std_data.head())
print(std_data.tail())
print(std_data.describe())

#----------------------FIRST MOMENT----------------------
print("Mean Values:\n", std_data[['Age','Marks','Attendance']].mean())
print("Median Values:\n", std_data[['Age','Marks','Attendance']].median())
print("Mode Values:\n", std_data[['Age','Marks','Attendance','Gender']].mode().iloc[0])

#----------------------SECOND MOMENT----------------------
print("Variance:\n", std_data[['Age','Marks','Attendance']].var())
print("Standard Deviation:\n", std_data[['Age','Marks','Attendance']].std())
print("Range (Max-Min):\n", std_data[['Age','Marks','Attendance']].max() - std_data[['Age','Marks','Attendance']].min())

#----------------------THIRD MOMENT----------------------
print("Skewness:\n", std_data[['Age','Marks','Attendance']].skew())

#----------------------FOURTH MOMENT----------------------
print("Kurtosis:\n", std_data[['Age','Marks','Attendance']].kurt())

#----------------------DATA VISUALIZATION----------------------
# Histograms
std_data[['Age','Marks','Attendance']].hist(figsize=(8,6), bins=10, color='skyblue')
plt.show()

# Boxplots
for col in ['Age','Marks','Attendance']:
    sns.boxplot(y=std_data[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Scatter plot (Attendance vs Marks)
plt.figure(figsize=(8,6))
plt.scatter(x=std_data.Attendance, y=std_data.Marks, color="purple", alpha=0.6)
plt.xlabel('Attendance')
plt.ylabel('Marks')
plt.title("Correlation between Marks and Attendance")
plt.show()

# Correlation matrix
print(std_data.corr())
sns.heatmap(std_data.corr(), annot=True, cmap="coolwarm")
plt.show()

#----------------------OUTLIER TREATMENT----------------------
# Winsorization for Age
winsor_age = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['Age'])
std_data['Age'] = winsor_age.fit_transform(std_data[['Age']])

# Winsorization for Attendance
winsor_att = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['Attendance'])
std_data['Attendance'] = winsor_att.fit_transform(std_data[['Attendance']])

# Winsorization for Marks
winsor_marks = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['Marks'])
std_data['Marks'] = winsor_marks.fit_transform(std_data[['Marks']])

#----------------------HANDLING MISSING VALUES----------------------
print("Missing values before imputation:\n", std_data.isna().sum())

# Age -> Mean
mean_imputer = SimpleImputer(strategy='mean')
std_data['Age'] = mean_imputer.fit_transform(std_data[['Age']])

# Attendance -> Median
median_imputer = SimpleImputer(strategy='median')
std_data['Attendance'] = median_imputer.fit_transform(std_data[['Attendance']])

# Gender -> Mode
mode_imputer = SimpleImputer(strategy='most_frequent')
std_data['Gender'] = mode_imputer.fit_transform(std_data[['Gender']])

# Marks -> Median
std_data['Marks'] = median_imputer.fit_transform(std_data[['Marks']])

print("Missing values after imputation:\n", std_data.isna().sum())

#----------------------FEATURE SCALING----------------------
# Drop Student_ID (not useful for prediction)
std_data = std_data.drop(['Student_ID'], axis=1)

# Separate numeric data
numeric_cols = ['Age','Attendance','Marks']

# Standardization
scaler = StandardScaler()
std_scaled = pd.DataFrame(scaler.fit_transform(std_data[numeric_cols]), columns=numeric_cols)

# Normalization
minmax = MinMaxScaler()
std_normalized = pd.DataFrame(minmax.fit_transform(std_data[numeric_cols]), columns=numeric_cols)

# Concatenate back with categorical
scaled_data = pd.concat([std_data[['Gender']], std_scaled], axis=1)

#----------------------MODEL BUILDING----------------------
# Independent (X) and Dependent (y)
X = std_data[['Gender', 'Age', 'Attendance']]
y = std_data['Marks']

# Encode Gender
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

# Train, predict, and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = r2

    # Scatter plot (Actual vs Predicted)
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label="Predicted vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             color='red', linestyle='--', label="Perfect Prediction")
    plt.xlabel("Actual Marks")
    plt.ylabel("Predicted Marks")
    plt.title(f"{name} - Actual vs Predicted")
    plt.legend()
    plt.show()

# Bar chart comparing R² scores
plt.figure(figsize=(6,4))
plt.bar(results.keys(), results.values(), color=['skyblue','orange','green','purple'])
plt.ylabel("R² Score")
plt.title("Model Comparison (R² Score)")
plt.show()

print("Model Performance (R² Scores):\n", results)
