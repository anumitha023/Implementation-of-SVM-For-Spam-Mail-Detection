# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries.

2.Read the CSV file and display data using head().

3.Split the dataset using train_test_split().

4.Calculate predictions and accuracy.

5.Print the outputs.

6.End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: M.R.Anumitha
RegisterNumber: 212223040018 
*/
```
```
import chardet, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn import metrics

# Detect encoding
with open('spam.csv', 'rb') as f:
    print(chardet.detect(f.read(100000)))

# Load data
data = pd.read_csv('spam.csv', encoding='windows-1252')
print(data.head())
print(data.info())
print(data.isnull().sum())

# Split data
x = data['v1'].values   # Labels
y = data['v2'].values   # Messages
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Text vectorization
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

# Train & predict
model = SVC()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Predictions:", y_pred)

# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

```
``
## Output:
![SVM For Spam Mail Detection](sam.png)

ENCODING DETECTED
![Screenshot 2025-05-08 094621](https://github.com/user-attachments/assets/1ef9b6a5-e94d-454a-9598-5473112e50f3)

FIRST FEW ROWS, DATA INFO, MISSING VALUES
![Screenshot 2025-05-08 094635](https://github.com/user-attachments/assets/9361ee7d-9c31-41e3-ac03-5193dd33a452)

PREDICTED LABELS
![Screenshot 2025-05-08 094651](https://github.com/user-attachments/assets/93aacee9-b2b0-443d-aceb-c3a018725c36)

MODEL ACCURACY
![Screenshot 2025-05-08 094657](https://github.com/user-attachments/assets/1164d7f3-c2ca-4909-8aee-26af28e54616)





## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
