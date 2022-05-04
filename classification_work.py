import importlib
from tabulate import tabulate

import mysklearn.myutils
importlib.reload(mysklearn.myutils)
import mysklearn.myutils as myutils

import mysklearn.mypytable
importlib.reload(mysklearn.mypytable)
from mysklearn.mypytable import MyPyTable 

import mysklearn.myclassifiers
importlib.reload(mysklearn.myclassifiers)
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier

import mysklearn.myclassifier_maya
importlib.reload(mysklearn.myclassifier_maya)
from mysklearn.myclassifier_maya import MyDecisionTreeClassifier, MyRandomForestClassifier

import mysklearn.myevaluation
importlib.reload(mysklearn.myevaluation)
import mysklearn.myevaluation as myevaluation

def classify_data(fname, random_state_val):
    print("-------CLASSIFIYING DATA...-----")
    #loading data
    stroke_data = MyPyTable()
    stroke_data.load_from_file(fname)
    print("attribute names:", stroke_data.column_names)

    #code to group data by class
    stroke_classes, stroke_data_by_class = myutils.group_by(stroke_data.data, stroke_data.column_names, "stroke")
    #stroke_classes = sorted(stroke_classes, reverse=True)
    print("classes:", stroke_classes)
    for partition in stroke_data_by_class:
        print("num instances of class", partition[0][-1], ":", len(partition))

    #make classifiers - KNN, NB, DT, RF
    knn_clf = MyKNeighborsClassifier(n_neighbors=5)
    nb_clf = MyNaiveBayesClassifier()
    tree_clf = MyDecisionTreeClassifier()
    #TODO: random forest
    #forest_clf = MyRandomForestClassifier(N=, M=, F=, random_forest=True)

    #Create X and y data
    X = [inst[:-1] for inst in stroke_data.data]
    y = [inst[-1] for inst in stroke_data.data]

    #k fold cross validation
    knn_y_pred = []
    nb_y_pred = []
    tree_y_pred = []
    #forest_y_pred = []

    y_true = []
    X_train_folds, X_test_folds = myevaluation.kfold_cross_validation(X, n_splits=10, random_state=random_state_val, shuffle=True)
    #print(X_train_folds)
    #print(X_test_folds)
    for fold_index in range(len(X_train_folds)): 
        X_train, y_train, X_test, y_test = myutils.one_fold_splits(X, y, X_train_folds, X_test_folds, fold_index)
        #build y true
        for val in y_test:
            y_true.append(val)
    
        #fit and predict kNN classifier
        knn_clf.fit(X_train, y_train)
        for pred in knn_clf.predict(X_test):
            knn_y_pred.append(pred) 
    
    #fit and predict naive bayes
        nb_clf.fit(X_train, y_train)
        for pred in nb_clf.predict(X_test):
            nb_y_pred.append(pred)
    
    #fit and predict tree    
        tree_clf.fit(X_train, y_train)
        for pred in tree_clf.predict(X_test):
            tree_y_pred.append(pred)
        
        #forest_clf.fit(X_train, y_train)
        #for pred in forest_clf.predict(X_test):
        #    forest_y_pred.append(pred)  
          
    predictions = [knn_y_pred, nb_y_pred, tree_y_pred] #TODO forest results
    titles = ["kNN Classifier (5 neighbors)", "Naive Bayes", "Decision Tree"]

    for i in range(len(predictions)):
        myutils.show_results(y_true, predictions[i], stroke_classes, titles[i])
