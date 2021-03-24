import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn import svm


def SplitData(data):
    ds=data.drop(columns='CPU Package Temperature')
    print("Dataset Shape : ",data.shape)
    X=ds.iloc[::].values
    Y=data.iloc[:,1].values
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=100)
    return X_train,X_test,Y_train,Y_test


def train_model_SVM(X_train,Y_train):
    regressor=svm.SVR(kernel='linear')
    regressor.fit(X_train,Y_train)
    return regressor


def result(model,X_test,Y_test):
    Y_pred=model.predict(X_test)
    print("Mean Squared Error : ",mean_squared_error(Y_test,Y_pred))
    print('Variance score: %.2f' % model.score(X_test,Y_test))


if __name__=="__main__":
    data=pd.read_csv("Dataset.csv")
    # print(data)
    X_train,X_test,Y_train,Y_test=SplitData(data)
    # print(X_train.shape,X_test.shape)
    # print(Y_train.shape,Y_test.shape)

    print("\nSVM :")
    model_SVM=train_model_SVM(X_train,Y_train)
    result(model_SVM,X_test,Y_test)
