# predictive-maintenance-AI
Predicting equipment failure using machine learning models.

🏭 Project Overview

This project demonstrates a Predictive Maintenance approach using the AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository.
The goal is to predict whether a machine is likely to fail based on sensor readings such as temperature, torque, rotational speed, and tool wear.
Predictive maintenance helps reduce downtime, optimize production, and save costs in industrial manufacturing environments.

🎯 Objectives

Clean and preprocess sensor data
Encode categorical features
Train a classification model to predict machine failure
Evaluate model performance with metrics and visualizations
Interpret results to understand the most influential factors in failures

🧰 Tools & Libraries Used

pandas: Data manipulation and preprocessing
numpy: Numerical computations
scikit-learn:	Model training, evaluation, and data splitting
matplotlib:	Data visualization
Google Colab:	Development environment

📊 Dataset

Source: AI4I 2020 Predictive Maintenance Dataset (UCI Repository)
Size: 10,000 samples × 14 features
Target variable: Machine failure (1 = failure, 0 = normal)
Features include:
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
Type (L, M, H) — machine type

⚙️ Project Workflow
1️⃣ Data Loading and Exploration
data = pd.read_csv('ai4i2020.csv')
print(data.head())
print(data.info())


Checked for missing values and data types

Visualized feature distributions

2️⃣ Preprocessing
data = data.drop(['UDI', 'Product ID'], axis=1)
data = pd.get_dummies(data, columns=['Type'])


Removed irrelevant identifiers

Converted categorical Type into numeric columns

3️⃣ Feature and Target Split
X = data.drop('Machine failure', axis=1)
y = data['Machine failure']

4️⃣ Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

5️⃣ Model Training
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

6️⃣ Evaluation
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=['No Failure', 'Failure'], cmap='Blues')
plt.show()

📈 Results

Accuracy: ~96–98% (depending on random state)

Precision & Recall: High for both classes

Confusion Matrix:
Visualized to show true vs. false predictions

Insights:

High torque and tool wear correlate strongly with machine failures

Type H machines show slightly higher failure risk

🧠 Key Learnings

Data preprocessing (handling IDs, encoding categories)
Using train_test_split for fair evaluation
Applying RandomForestClassifier for classification
Visualizing results with confusion matrices and charts
Understanding class imbalance and its effect on model accuracy
