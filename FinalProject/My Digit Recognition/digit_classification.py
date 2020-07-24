import pandas as pd
import matplotlib.pyplot as plt  
from sklearn import svm
from sklearn import metrics
import joblib
import numpy as np
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import sys
import os

def check(choice):
    ##Get Data from CSV
    script_dir = os.getcwd()
    file = 'dataset6labels.csv'
    dataframe = pd.read_csv(os.path.normcase(os.path.join(script_dir, file)))
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    #print(dataframe) 

    ##Seperate Labels and Features
    X = dataframe.drop(['label'],axis=1)
    Y = dataframe["label"]

    X_train, Y_train = X[0:130], Y[0:130]
    X_test, Y_test = X[130:], Y[130:]
    
    ##Build a Model and Save it
    if choice == "1":
            model = GaussianNB()
    elif choice == "2":
            model = SVC(kernel='linear', C=1)
    elif choice == "3":
            model = SVC(kernel='sigmoid', C=1)     
    elif choice == "4":
            model = SVC(kernel='poly', C=1)     
    elif choice == "5":
            model = SVC(kernel='rbf', C=1, gamma=0.01)       
    else:
            model = KNeighborsClassifier(n_neighbors=10)

    model.fit(X_train,Y_train)

    ##Step5 : Print Accuracy 
    prediction = model.predict(X_test)
    print("Model Score/Accuracy is", metrics.accuracy_score(Y_test, prediction))

def main():

        ans = True

        while ans:

                print("---------------------- MAIN MENU -----------------------")
                print("------------------- DIGIT RECOGNITION ------------------")
                print("Please select the appropriate classification algorithm.")
                choice = input("""
1: Gaussian
2: SVM_Linear
3: SVM_Sigmoid
4: SVM_Poly
5: SVM_Rbf
6: KNeighbors
7: Quit/Log Out
Please enter your choice: """)

                if choice == "1":
                        print("You selected Gaussian Naive Bayes Classifier.")
                        check(choice)
                elif choice == "2":
                        print("You selected SVM Linear Classifier.")
                        check(choice)
                elif choice == "3":
                        print("You selected SVM Sigmoid Classifier.")
                        check(choice)
                elif choice == "4":
                        print("You selected SVM Poly Classifier.")
                        check(choice)
                elif choice == "5":
                        print("You selected SVM Non Linear Classifier.")
                        check(choice)
                elif choice == "6":
                        print("You selected KNeighbors.")
                        check(choice)
                elif choice == "7":  
                        ans = False
                        print("Bye Bye!")
                        sys.exit
                else:
                        print("You must only select either 1,2,3,4,6 or 7.")
                        print("Please try again")
                        main()   
               

if __name__ == "__main__":

    main()