"""This file tests the classifiers in myclassifiers.py using desk calculations and
sci-kit-learn libraries. Each test performs multiple tests case.
"""
import numpy as np
from sklearn import naive_bayes
from sklearn.linear_model import LinearRegression
from mysklearn import myevaluation
from mysklearn.myclassifier_maya import MyRandomForestClassifier #for testing

from mysklearn.myclassifiers import MyNaiveBayesClassifier,\
    MyKNeighborsClassifier, MyDecisionTreeClassifier
from mysklearn.mypytable import MyPyTable

interview_X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
interview_y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True",
                         "True", "True", "True", "True", "False"]
degrees_X_train = [
        ["A", "B", "A", "B", "B"],
        ["A", "B", "B", "B", "A"],
        ["A", "A", "A", "B", "B"],
        ["B", "A", "A", "B", "B"],
        ["A", "A", "B", "B", "A"],
        ["B", "A", "A", "B", "B"],
        ["A", "B", "B", "B", "B"],
        ["A", "B", "B", "B", "B"],
        ["A", "A", "A", "A", "A"],
        ["B", "A", "A", "B", "B"],
        ["B", "A", "A", "B", "B"],
        ["A", "B", "B", "A", "B"],
        ["B", "B", "B", "B", "A"],
        ["A", "A", "B", "A", "B"],
        ["B", "B", "B", "B", "A"],
        ["A", "A", "B", "B", "B"],
        ["B", "B", "B", "B", "B"],
        ["A", "A", "B", "A", "A"],
        ["B", "B", "B", "A", "A"],
        ["B", "B", "A", "A", "B"],
        ["B", "B", "B", "B", "A"],
        ["B", "A", "B", "A", "B"],
        ["A", "B", "B", "B", "A"],
        ["A", "B", "A", "B", "B"],
        ["B", "A", "B", "B", "B"],
        ["A", "B", "B", "B", "B"],
    ]
degrees_y_train = ["SECOND", "FIRST", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", "FIRST",
    "SECOND", "SECOND", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND",
    "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND"]

X_train_iphone = [
        [1, 3, "fair"], #no
        [1, 3, "excellent"], #no
        [2, 3, "fair"], #yes
        [2, 2, "fair"], #yes
        [2, 1, "fair"], #yes
        [2, 1, "excellent"], #no
        [2, 1, "excellent"], #yes
        [1, 2, "fair"], #no
        [1, 1, "fair"], #yes
        [2, 2, "fair"], #yes
        [1, 2, "excellent"], #yes
        [2, 2, "excellent"], #yes
        [2, 3, "fair"], #yes
        [2, 2, "excellent"], #no
        [2, 3, "fair"] #yes
    ]
y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", \
                    "yes", "yes", "yes", "yes", "no", "yes"]


def test_kneighbors_classifier_kneighbors():
    """Tests the kNN classifier's kneighbors function to ensure the correct k neighbors
    are chosen based on the euclidean distances. 3 Test cases.
    """
    #TEST CASE 1
    X_train_class_example1 = [[1.0, 1.0], [1.0, 0], [0.33, 0], [0, 0]] #already normalized
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    my_knn_clf = MyKNeighborsClassifier()
    my_knn_clf.fit(X_train_class_example1, y_train_class_example1)
    test1_distances, test1_neighbor_indices = my_knn_clf.kneighbors([[.33, 1.0]])
    #assert
    assert np.allclose(test1_distances, [[0.6699999999999999, 1.0, 1.0530432089900206]])
    assert np.allclose(test1_neighbor_indices, [[0, 2, 3]])

    #TEST CASE 2
    X_train_class_example2 = [[3, 2], [6, 6], [4, 1], [4, 4], [1, 2], [2, 0], [0, 3], [1, 6]]
    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    my_knn_clf.fit(X_train_class_example2, y_train_class_example2)
    test2_distances, test2_neighbor_indices = my_knn_clf.kneighbors([[2,3]])
    assert np.allclose(test2_neighbor_indices, [[0, 4, 6]])
    assert np.allclose(test2_distances, [[1.4142135623730951, 1.4142135623730951, 2.0]])

    #TEST CASE 3
    X_train_bramer_example = [[0.8, 6.3], [1.4, 8.1], [2.1, 7.4], [2.6, 14.3], [6.8, 12.6],\
    [8.8, 9.8], [9.2, 11.6], [10.8, 9.6], [11.8, 9.9], [12.4, 6.5], [12.8, 1.1], [14.0, 19.9],\
    [14.2, 18.5], [15.6, 17.4], [15.8, 12.2], [16.6, 6.7], [17.4, 4.5], [18.2, 6.9], [19.0, 3.4],\
    [19.6, 11.1]]
    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
    "-", "-", "+", "+", "+", "-", "+"]
    my_knn_clf3 = MyKNeighborsClassifier(n_neighbors=5)
    my_knn_clf3.fit(X_train_bramer_example, y_train_bramer_example)
    test3_distances, test3_neighbor_indices = my_knn_clf3.kneighbors([[9.1, 11.0]])
    assert np.allclose(test3_neighbor_indices, [[6, 5, 7, 4, 8]])
    assert np.allclose(test3_distances, [[0.608276253, 1.236931687, 2.202271554, 2.801785145, 2.915475947]])

def test_kneighbors_classifier_predict():
    """Tests the kNN classifier's predict function. Ensures the correct prediction is made
    based off of given data. 3 test cases.
    """
    #TEST CASE 1
    X_train_class_example1 = MyPyTable(data=[[1.0, 1.0], [1.0, 0], [0.33, 0], [0, 0]])
    y_train_class_example1 = MyPyTable(data=["bad", "bad", "good", "good"])
    X_test = [[.33, 1.0]]
    my_knn_clf = MyKNeighborsClassifier()
    my_knn_clf.fit(X_train_class_example1.data, y_train_class_example1.data)
    predictions = my_knn_clf.predict(X_test)
    assert predictions == ['good']

    #TEST CASE 2
    X_train_class_example2 = [[3, 2], [6, 6], [4, 1], [4, 4], [1, 2], [2, 0], [0, 3], [1, 6]]
    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test = [[2, 3]]
    my_knn_clf2 = MyKNeighborsClassifier()
    my_knn_clf2.fit(X_train_class_example2, y_train_class_example2)
    predictions = my_knn_clf2.predict(X_test)
    assert predictions == ['yes']

    #TEST CASE 3
    X_train_bramer_example = [[0.8, 6.3], [1.4, 8.1], [2.1, 7.4], [2.6, 14.3], [6.8, 12.6],\
    [8.8, 9.8], [9.2, 11.6], [10.8, 9.6], [11.8, 9.9], [12.4, 6.5], [12.8, 1.1], [14.0, 19.9],\
    [14.2, 18.5], [15.6, 17.4], [15.8, 12.2], [16.6, 6.7], [17.4, 4.5], [18.2, 6.9], [19.0, 3.4],\
    [19.6, 11.1]]
    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
    "-", "-", "+", "+", "+", "-", "+"]
    X_test = [[9.1, 11.0]]
    my_knn_clf3 = MyKNeighborsClassifier(n_neighbors=5)
    my_knn_clf3.fit(X_train_bramer_example, y_train_bramer_example)
    predictions = my_knn_clf3.predict(X_test)
    assert predictions == ["+"]

def test_naive_bayes_classifier_fit():
    #=================TEST CASE 1: 8 instance example - ipad LT1
    inclass_example_col_names = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    #classifier
    nb_clf = MyNaiveBayesClassifier()
    nb_clf.fit(X_train_inclass_example, y_train_inclass_example)
    #results,test
    priors_soln = [3/8, 5/8] #no, yes alphabetical
    posteriors_soln = [ [[2/3, 4/5], #1- 1
                        [1/3, 1/5]], #1- 2
                        [[2/3, 2/5], #2- 5
                        [1/3, 3/5]] ]#2- 6

    assert np.allclose(nb_clf.priors, priors_soln)
    assert np.allclose(nb_clf.posteriors, posteriors_soln)

    #=================TEST CASE 2: 15 instance example - RQ5
    # RQ5 (fake) iPhone purchases dataset
    nb_clf.fit(X_train_iphone, y_train_iphone)

    priors_soln = [5/15, 10/15] #no, yes -- order read in
    posteriors_soln = [ [[3/5, 2/10], #standing 1
                        [2/5, 8/10]], #standing 2
                        [[1/5, 3/10], #job status 1
                        [2/5, 4/10], #js 2
                        [2/5, 3/10]], #js 3
                        [[3/5, 3/10], #credit excellent --ALPHABETICAL
                        [2/5, 7/10]] #credit fair
                        ]

    assert np.allclose(nb_clf.priors, priors_soln)
    #assert np.allclose(nb_clf.posteriors, posteriors_soln)
    assert nb_clf.posteriors == posteriors_soln #none are infinte fractions

    #=================TEST CASE 3: train dataset - Bramer 3.2
    X_train_train_table = [
        ['weekday', 'spring', 'none', 'none'],
        ['weekday', 'winter', 'none', 'slight'],
        ['weekday', 'winter', 'none', 'slight'],
        ['weekday', 'winter', 'high', 'heavy'],
        ['saturday', 'summer', 'normal', 'none'],
        ['weekday', 'autumn', 'normal', 'none'],
        ['holiday', 'summer', 'high', 'slight'],
        ['sunday', 'summer', 'normal', 'none'],
        ['weekday', 'winter', 'high', 'heavy'],
        ['weekday', 'summer', 'none', 'slight'],
        ['saturday', 'spring', 'high', 'heavy'],
        ['weekday', 'summer', 'high', 'slight'],
        ['saturday', 'winter', 'normal', 'none'],
        ['weekday', 'summer', 'high', 'none'],
        ['weekday', 'winter', 'normal', 'heavy'],
        ['saturday', 'autumn', 'high', 'slight'],
        ['weekday', 'autumn', 'none', 'heavy'],
        ['holiday', 'spring', 'normal', 'slight'],
        ['weekday', 'spring', 'normal', 'none'],
        ['weekday', 'spring', 'normal', 'slight']
    ]
    y_train_train_table = ['on time', 'on time', 'on time', 'late', 'on time',
                            'very late', 'on time', 'on time', 'very late', 'on time',
                            'cancelled', 'on time', 'late', 'on time', 'very late',
                            'on time', 'on time', 'on time', 'on time', 'on time']

    nb_clf.fit(X_train_train_table, y_train_train_table)
    priors_soln = [1/20, 2/20, 14/20, 3/20] #alphabetical: canc, late, on time, VL
    posteriors_soln = [
        [   [0/1, 0/2, 2/14, 0/3], #holida
            [1/1, 1/2, 2/14, 0/3], #sat
            [0/1, 0/2, 1/14, 0/3], #sun
            [0/1, 1/2, 9/14, 3/3]], #weekday
        [   [0/1, 0/2, 2/14, 1/3], #autumn
            [1/1, 0/2, 4/14, 0/3], #sprint
            [0/1, 0/2, 6/14, 0/3], #summer
            [0/1, 2/2, 2/14, 2/3]], #winter
        [   [1/1, 1/2, 4/14, 1/3], #high
            [0/1, 0/2, 5/14, 0/3], #none
            [0/1, 1/2, 5/14, 2/3]], #normal
        [   [1/1, 1/2, 1/14, 2/3], #heavy
            [0/1, 1/2, 5/14, 1/3], #none
            [0/1, 0/2, 8/14, 0/3]] #slight
    ]

    assert np.allclose(nb_clf.priors, priors_soln)
    #assert np.allclose(nb_clf.posteriors, posteriors_soln) a warning prevented this
    assert nb_clf.posteriors == posteriors_soln


def test_naive_bayes_classifier_predict():
    #=================TEST CASE 1: in class
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    #classifier
    nb_clf = MyNaiveBayesClassifier()
    nb_clf.fit(X_train_inclass_example, y_train_inclass_example)
    #predict
    X_test1 = [[1,5]]
    y_predicted1 = nb_clf.predict(X_test1)
    assert y_predicted1 == ["yes"]


    #=================TEST CASE 2: iphone dataset
    X_train_iphone = [
        [1, 3, "fair"], #no
        [1, 3, "excellent"], #no
        [2, 3, "fair"], #yes
        [2, 2, "fair"], #yes
        [2, 1, "fair"], #yes
        [2, 1, "excellent"], #no
        [2, 1, "excellent"], #yes
        [1, 2, "fair"], #no
        [1, 1, "fair"], #yes
        [2, 2, "fair"], #yes
        [1, 2, "excellent"], #yes
        [2, 2, "excellent"], #yes
        [2, 3, "fair"], #yes
        [2, 2, "excellent"], #no
        [2, 3, "fair"] #yes
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    nb_clf.fit(X_train_iphone, y_train_iphone)
    X_test2 = [ [2, 2, "fair"],
                [1, 1, "excellent"]]
    y_predicted2 = nb_clf.predict(X_test2)
    assert y_predicted2 == ["yes", "no"]

    #=================TEST CASE 3: train dataset
    X_train_train_table = [
        ['weekday', 'spring', 'none', 'none'],
        ['weekday', 'winter', 'none', 'slight'],
        ['weekday', 'winter', 'none', 'slight'],
        ['weekday', 'winter', 'high', 'heavy'],
        ['saturday', 'summer', 'normal', 'none'],
        ['weekday', 'autumn', 'normal', 'none'],
        ['holiday', 'summer', 'high', 'slight'],
        ['sunday', 'summer', 'normal', 'none'],
        ['weekday', 'winter', 'high', 'heavy'],
        ['weekday', 'summer', 'none', 'slight'],
        ['saturday', 'spring', 'high', 'heavy'],
        ['weekday', 'summer', 'high', 'slight'],
        ['saturday', 'winter', 'normal', 'none'],
        ['weekday', 'summer', 'high', 'none'],
        ['weekday', 'winter', 'normal', 'heavy'],
        ['saturday', 'autumn', 'high', 'slight'],
        ['weekday', 'autumn', 'none', 'heavy'],
        ['holiday', 'spring', 'normal', 'slight'],
        ['weekday', 'spring', 'normal', 'none'],
        ['weekday', 'spring', 'normal', 'slight']
    ]
    y_train_train_table = ['on time', 'on time', 'on time', 'late', 'on time',
                            'very late', 'on time', 'on time', 'very late', 'on time',
                            'cancelled', 'on time', 'late', 'on time', 'very late',
                            'on time', 'on time', 'on time', 'on time', 'on time']

    nb_clf.fit(X_train_train_table, y_train_train_table)
    X_test3 = [
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "high", "heavy"],
        ["sunday", "summer", "normal", "slight"]
        ]
    y_predicted3 = nb_clf.predict(X_test3)
    assert y_predicted3 == ["very late", "on time", "on time"]


def test_decision_tree_classifier_fit():
    #===================TEST CASE 1===================
    #create/fit tree classifier
    tree_clf = MyDecisionTreeClassifier()
    tree_clf.fit(interview_X_train, interview_y_train)

    #assert tree solution
    interview_tree = \
        ["Attribute", "att0",
            ["Value", "Junior", 
                ["Attribute", "att3",
                    ["Value", "no", 
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes", 
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]
    assert tree_clf.tree == interview_tree

    #===================TEST CASE 2===================
    tree_clf.fit(degrees_X_train, degrees_y_train)
    degrees_tree = \
        ["Attribute", "att0", 
            ["Value", "A",
                ["Attribute", "att4", 
                    ["Value", "A",
                        ["Leaf", "FIRST", 5, 14]
                    ],
                    ["Value", "B",
                        ["Attribute", "att3",
                            ["Value", "A",
                                ["Attribute", "att1",
                                    ["Value", "A", 
                                        ["Leaf", "FIRST", 1, 2]
                                    ],
                                    ["Value", "B", 
                                        ["Leaf", "SECOND", 1, 2]
                                    ]
                                ]
                            ],
                            ["Value", "B",
                                ["Leaf", "SECOND", 7, 9]
                            ]
                        ]
                    ]
                ]
            ],
            ["Value", "B",
                ["Leaf", "SECOND", 12, 26]
            ]
        ]
    assert tree_clf.tree == degrees_tree

    #===================TEST CASE 3===================
    iphone_tree = \
        ["Attribute", "att0",
            ["Value", 1, 
                ["Attribute", "att1",
                    ["Value", 1,
                        ["Leaf", "yes", 1, 5]
                    ],
                    ["Value", 2,
                        ["Attribute", "att2",
                            ["Value", "excellent",
                                ["Leaf", "yes", 1,2]
                            ],
                            ["Value", "fair",
                                ["Leaf", "no", 1,2]
                            ]
                        ]
                    ],
                    ["Value", 3,
                        ["Leaf", "no", 2, 5]
                    ]
                ]
            ],
            ["Value", 2,
                ["Attribute", "att2",
                    ["Value", "excellent",
                        ["Leaf", "no", 2,4]
                    ],
                    ["Value", "fair",
                        ["Leaf", "yes", 6,10]
                    ]
                ]
            ]
        ]
    #fit classifier
    tree_clf.fit(X_train_iphone, y_train_iphone)
    #assert
    assert tree_clf.tree == iphone_tree

def test_decision_tree_classifier_predict():
    #===================TEST CASE 1===================
    #create/fit tree classifier
    tree_clf = MyDecisionTreeClassifier()
    tree_clf.fit(interview_X_train, interview_y_train)
    X1 = ["Junior", "Java", "yes", "no"]
    X2 = ["Junior", "Java", "yes", "yes"]
    prediction1 = tree_clf.predict([X1, X2])
    assert prediction1 == ["True", "False"]

    #===================TEST CASE 2===================
    X_test_2_cases = [["B", "B", "B", "B", "B"], ["A", "A", "A", "A", "A"], ["A", "A", "A", "A", "B"]]
    tree_clf.fit(degrees_X_train, degrees_y_train)
    prediction2 = tree_clf.predict(X_test_2_cases)
    assert prediction2 == ["SECOND", "FIRST", "FIRST"]

    #===================TEST CASE 3===================
    X_test_3 = [[2, 2, "fair"], [1, 1, "excellent"]]
    tree_clf.fit(X_train_iphone, y_train_iphone)
    prediction3 = tree_clf.predict(X_test_3)
    assert prediction3 == ["yes", "yes"]

interview_X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
interview_y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True",
                         "True", "True", "True", "True", "False"]
