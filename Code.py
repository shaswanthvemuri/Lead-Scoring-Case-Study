import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset
lead_df = pd.read_csv('Lead Scoring.csv')

# Display basic information about the dataset
print(lead_df.head())
lead_df.info()

# Check for missing values
print(lead_df.isnull().sum())

# Fill missing values for numeric columns with median and categorical columns with mode
for column in lead_df.columns:
    if lead_df[column].dtype == 'object':
        lead_df[column].fillna(lead_df[column].mode()[0], inplace=True)
    else:
        lead_df[column].fillna(lead_df[column].median(), inplace=True)

# Encoding categorical variables (if any)
lead_df = pd.get_dummies(lead_df, drop_first=True)

# Prepare the data
X = lead_df.drop('Converted', axis=1)  # Replace 'Converted' with the actual target column name
y = lead_df['Converted']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Decision Tree Accuracy:', accuracy)

# Visualizations

# 1. Line Graphs: Model performance over different epochs (using dummy data for illustration)
epochs = np.arange(1, 11)
performance = np.random.rand(10) * 0.1 + 0.8  # Dummy data
plt.figure(figsize=(10, 6))
plt.plot(epochs, performance, marker='o')
plt.title('Model Performance over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Performance')
plt.grid(True)
plt.show()

# 2. Bar Charts: Frequency of different lead sources
plt.figure(figsize=(12, 6))
sns.countplot(data=lead_df, x='Lead Source_Olark Chat')
plt.title('Frequency of Different Lead Sources')
plt.xticks(rotation=90)
plt.show()

# 3. Scatter Plots: Relationship between TotalVisits and Total Time Spent on Website
plt.figure(figsize=(10, 6))
sns.scatterplot(data=lead_df, x='TotalVisits', y='Total Time Spent on Website', hue='Converted')
plt.title('Total Visits vs Total Time Spent on Website')
plt.xlabel('Total Visits')
plt.ylabel('Total Time Spent on Website')
plt.show()

# 4. Histograms: Distribution of Total Time Spent on Website
plt.figure(figsize=(10, 6))
sns.histplot(data=lead_df, x='Total Time Spent on Website', bins=20, kde=True)
plt.title('Distribution of Total Time Spent on Website')
plt.xlabel('Total Time Spent on Website')
plt.ylabel('Frequency')
plt.show()

# Train and evaluate K-Nearest Neighbors (KNN) model
training_accuracy = []
test_accuracy = []
for n_neighbors in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    training_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))

print("KNN Training Accuracy for different neighbors:", training_accuracy)
print("KNN Test Accuracy for different neighbors:", test_accuracy)

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
print(f"KNN Training Accuracy: {knn.score(X_train, y_train)}")
print(f"KNN Test Accuracy: {knn.score(X_test, y_test)}")

# Train and evaluate Decision Tree with depth limit
dt1 = DecisionTreeClassifier(random_state=0, max_depth=3)
dt1.fit(X_train, y_train)
print(f"Decision Tree (max_depth=3) Training Accuracy: {dt1.score(X_train, y_train)}")
print(f"Decision Tree (max_depth=3) Test Accuracy: {dt1.score(X_test, y_test)}")

# Train and evaluate Multi-Layer Perceptron (MLP) model
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)
print(f"MLP Training Accuracy: {mlp.score(X_train, y_train)}")
print(f"MLP Test Accuracy: {mlp.score(X_test, y_test)}")

# Scaling the data and evaluating MLP with scaled data
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

mlp1 = MLPClassifier(random_state=0)
mlp1.fit(X_train_scaled, y_train)
print(f"Scaled MLP Training Accuracy: {mlp1.score(X_train_scaled, y_train)}")
print(f"Scaled MLP Test Accuracy: {mlp1.score(X_test_scaled, y_test)}")
