import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report
# convert these categorical labels into numeric values.
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Load your dataset
data = pd.read_excel('New Microsoft Excel Worksheet.xlsx')

# Identify the column containing the categorical class labels
class_labels = data['Class']

#Identify the column containing the categorical class labels. Let's say it's called8 'class_label' in your dataset.
# Initialize and fit the LabelEncoder
# label_encoder is for transform non-numeric labels (e.g., class labels or target variables) into numeric values
label_encoder = LabelEncoder()
data['Class'] = label_encoder.fit_transform(class_labels)

print(data['Class'])


#`np.array(data)`takes the data from the `data` variable and creates a new NumPy array with the same data
data = np.array(data)


#extracts the last column from the data array
y = data[:, -1]
#extracts all columns except the last one from the data array
x = data[:, :-1]


X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)


# Create an instance of the KNeighborsClassifier class
model = KNeighborsClassifier(n_neighbors=5)


#use the fit method of the DecisionTreeClassifier to train the model
model = model.fit(X_train, y_train)


#calls the predict method of your model to make predictions on the X_test data
y_hat = model.predict(X_test)


print(classification_report(y_test, y_hat))
