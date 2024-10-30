# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the Program.
2. Import the necessary packages.
3. Read the given csv file and display the few contents of the data.
4. Assign the features for x and y respectively.
5. Split the x and y sets into train and test sets.
6. Convert the Alphabetical data to numeric using CountVectorizer.
7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
8. Find the accuracy of the model.
9. End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Yogesh. V
RegisterNumber:  212223230250
*/
```
```
import pandas as pd
data=pd.read_csv("Exp_11_spam.csv",encoding='windows-1252')
data.head()

dat.info()
data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![Screenshot 2024-10-26 195813](https://github.com/user-attachments/assets/ff371d34-9769-43de-96f6-7821466883d3)
### Info:
![Screenshot 2024-10-26 195824](https://github.com/user-attachments/assets/6bc218e6-d608-4c3c-be60-cfb52a6ba513)
### Y-prediction:
![Screenshot 2024-10-26 195836](https://github.com/user-attachments/assets/c56cbb97-199e-4f10-a351-12275f8ae582)
### Accuracy:
![Screenshot 2024-10-26 195844](https://github.com/user-attachments/assets/a02ac7d1-2ac7-447f-848b-e564f5e5d669)
## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
