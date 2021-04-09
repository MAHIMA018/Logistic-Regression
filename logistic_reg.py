import os
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt 

os.chdir(r'C:\Users\...')

dataset = pd.read_csv('Social_Network_Ads.csv')
#print(os.getcwd()) 

X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values
#print(len(X))
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 1/4 , random_state= 0)

#feature scaling
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)

#fitting logistic regression to the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)
#predicting the test results
Y_pred = classifier.predict(X_test)
#print(Y_pred)

#create confusion matrix
cm = confusion_matrix(Y_test,Y_pred)
print(cm)

accuracy = (np.trace(cm)/100)*100
print(accuracy)

#visualising the training set results 
x,y = X_train,Y_train 
x1,x2 = np.meshgrid(np.arange(start= x[:,0].min()-1,stop = x[:,0].max()+1,step=0.01),
np.arange(start = x[:,1].min()-1,stop = x[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
alpha=0.75,cmap=ListedColormap(('blue','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y)):
    plt.scatter(x[y==j,0],x[y==j,1],
    c= ListedColormap(('blue','green'))(i),label = j)
plt.title('Logistic Regression (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()   

#visualising the Logistic Regression for Test Set
x,y = X_test,Y_test
x1,x2 = np.meshgrid(np.arange(start= x[:,0].min()-1,stop = x[:,0].max()+1,step=0.01),
np.arange(start = x[:,1].min()-1,stop = x[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
alpha=0.75,cmap=ListedColormap(('blue','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y)):
    plt.scatter(x[y==j,0],x[y==j,1],
    c= ListedColormap(('blue','green'))(i),label = j)
plt.title('Logistic Regression (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()   
