# Task 3: Customer Purchase Prediction using Decision Tree Classifier

# 📦 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 📊 2. Create Synthetic Dataset
np.random.seed(42)
n_samples = 10000

jobs = ['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar',
        'unemployed', 'entrepreneur', 'student', 'housemaid']
marital_status = ['married', 'single', 'divorced']
education_levels = ['primary', 'secondary', 'tertiary', 'unknown']
housing_loan = ['yes', 'no']
personal_loan = ['yes', 'no']
contact_types = ['cellular', 'telephone']
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
poutcomes = ['success', 'failure', 'nonexistent']

data = {
    'age': np.random.randint(18, 95, n_samples),
    'job': np.random.choice(jobs, n_samples),
    'marital': np.random.choice(marital_status, n_samples),
    'education': np.random.choice(education_levels, n_samples),
    'balance': np.random.randint(-2000, 50000, n_samples),
    'housing': np.random.choice(housing_loan, n_samples),
    'loan': np.random.choice(personal_loan, n_samples),
    'contact': np.random.choice(contact_types, n_samples),
    'day': np.random.randint(1, 31, n_samples),
    'month': np.random.choice(months, n_samples),
    'duration': np.random.randint(10, 4000, n_samples),
    'campaign': np.random.randint(1, 50, n_samples),
    'previous': np.random.randint(0, 10, n_samples),
    'poutcome': np.random.choice(poutcomes, n_samples),
    'purchase': np.random.choice(['yes', 'no'], n_samples, p=[0.15, 0.85])
}

df = pd.DataFrame(data)

# 💾 Save dataset (optional)
df.to_csv("customer_purchase_data.csv", index=False)

# 📋 3. Basic EDA
print(df.head())
print(df['purchase'].value_counts())

# 📊 4. Visualize Class Distribution
sns.countplot(data=df, x='purchase')
plt.title("Class Distribution")
plt.show()

# 🧼 5. Encode Categorical Variables
label_encoders = {}
for column in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# 🧪 6. Split Data
X = df.drop('purchase', axis=1)
y = df['purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌳 7. Train Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# 📈 8. Model Evaluation
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 🌲 9. Visualize Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.title("Decision Tree")
plt.show()
