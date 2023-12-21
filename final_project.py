from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Import data
normal = pd.read_csv('normal.csv')
LL1 = pd.read_csv('LL1.csv')
LL2 = pd.read_csv('LL2.csv')
Str1 = pd.read_csv('Str1_3-Str2_2-LL1.csv')
warning_data_wec = pd.read_csv('Partial_shading_edit.csv')

# Concatenate data
total = pd.concat([normal, LL1, LL2, Str1, warning_data_wec])

features = ['V', 'I', 'P', 'G']
x = total[features]
y1 = total['fault']
y2 = total['no_module_fault']
y3 = total['partial_shading']


# Create and train the KNN classifier
x1train, x1test, y1train, y1test = train_test_split(x, y2, test_size=0.2, random_state=50)
model4 = KNeighborsClassifier()
model4.fit(x1train, y1train)

# Make predictions on the test set
y4pred = model4.predict(x1test)

# Evaluate the performance of the model
accuracy = accuracy_score(y1test, y4pred)
# print("Accuracy:", accuracy)

#test thử với số thật
import numpy as np
# Assuming your four numbers are stored in a list called `numbers`
numbers = [139.09889 , 8.61107 ,  898.10606 , 1197.790279]

# Convert the list to a numpy array and reshape it to match the training data shape
input_data = np.array(numbers).reshape(1, -1)

# Use the trained model to make predictions on the input data
prediction = int( model4.predict(input_data))
if prediction == 0:
            prediction = "Normal State "
elif prediction == 1:
            prediction= "Short circuit fault on one panel state"
elif prediction ==2:
            prediction = "Short circuit fault on two panels state"
elif prediction ==3:
            prediction = "Partial shading state"
print(prediction)