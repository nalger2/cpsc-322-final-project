#classification implementation code --- to be cleaned up and pasted into Technical Report

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
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier

import mysklearn.myevaluation
importlib.reload(mysklearn.myevaluation)
import mysklearn.myevaluation as myevaluation

#loading data
stroke_data = MyPyTable()
stroke_data.load_from_file("input_data/stroke-data.csv")
print("attribute names:", stroke_data.column_names)
print("len before removing missing vals", len(stroke_data.data))

#code to remove rows w/missing values (use only once relevant attributes are decided)
#stroke_data.remove_rows_with_missing_values()
#stroke_data.remove_rows_with_missing_values_by_col()
#print("len after removing missing vals", len(stroke_data.data))

#code to group data by class
stroke_classes, stroke_data_by_class = myutils.group_by(stroke_data.data, stroke_data.column_names, "stroke")
print("classes:", stroke_classes)
for partition in stroke_data_by_class:
    print("num instances of class", partition[0][-1], ":", len(partition))

#remove possibly irrelevant columns
cols_to_remove = [7, 6, 5, 0] #residence, work, married, ID
stroke_data.remove_columns(cols_to_remove)
print(stroke_data.column_names)
print(stroke_data.data[0])