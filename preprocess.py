import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = pd.read_excel('New Microsoft Excel Worksheet.xlsx')


class_labels = data['Class']

label_encoder = LabelEncoder()
data['Class'] = label_encoder.fit_transform(class_labels)


data_array = np.array(data)

# Extract features (X) and target variable (y) from the NumPy array
y = data_array[:, -1] 
X = data_array[:, :-1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a KNN classifier with additional parameters
knn_classifier = KNeighborsClassifier(
    n_neighbors=3,
    weights='uniform',
    algorithm='auto',
    p=2,
)

# Fit the classifier to the training data
knn_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn_classifier.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Additional evaluation metrics including F1 score and recall
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))
