# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load Dataset and Create DataFrame: Load the Iris dataset, create a DataFrame, and add a target column for species labels.
2. Define Features and Target: Set x as the feature columns and y as the target (species) column.
3. Split Data into Train and Test Sets: Use train_test_split to divide x and y into training and testing sets (80-20 split).
4. Initialize and Train Model: Initialize an SGDClassifier with specified parameters and fit it to the training data.
5. Make Predictions: Predict target labels for the test set using the trained model.
6. Calculate Accuracy: Compute the accuracy score by comparing predicted and actual values in the test set.
7. Generate Confusion Matrix: Display the confusion matrix to evaluate model performance across different classes.







## Program:
### DATA:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SRINIIDHI SENTHIL
RegisterNumber: 212222230148
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
```
### SGDClassifier:
```
iris = load_iris()
df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
```
```
x=df.drop('target',axis=1)
y=df['target']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(x_train, y_train)
```
### ACCURACY:
```
sgd_clf.fit(x_train, y_train)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```
### CONFUSION MARIX:
```
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```
## Output:
### DATA:
![image](https://github.com/user-attachments/assets/03192351-7bb9-4aa5-a7cd-5d47414c2238)
### SGDClassifier:
![image](https://github.com/user-attachments/assets/8a285b91-d149-4a8e-8de6-ffce187aef97)
### ACCURACY:
![image](https://github.com/user-attachments/assets/c93be364-5fb2-458a-b88c-1dab94d3b108)
### CONFUSION MARIX:
![image](https://github.com/user-attachments/assets/f14d99a2-441f-4525-ba3d-5138908fda9e)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
