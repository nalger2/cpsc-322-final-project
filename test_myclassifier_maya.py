import numpy as np
from sklearn import naive_bayes
from sklearn.linear_model import LinearRegression
from mysklearn import myevaluation
from mysklearn.myclassifier_maya import MyRandomForestClassifier, MyDecisionTreeClassifier #for testing

from mysklearn.mypytable import MyPyTable

# interview dataset
header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
X_train_interview = [
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
y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

# note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
# note: the attribute values are sorted alphabetically
tree_interview = \
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
        
# bramer degrees dataset
header_degrees = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
X_train_degrees = [
    ['A', 'B', 'A', 'B', 'B'],
    ['A', 'B', 'B', 'B', 'A'],
    ['A', 'A', 'A', 'B', 'B'],
    ['B', 'A', 'A', 'B', 'B'],
    ['A', 'A', 'B', 'B', 'A'],
    ['B', 'A', 'A', 'B', 'B'],
    ['A', 'B', 'B', 'B', 'B'],
    ['A', 'B', 'B', 'B', 'B'],
    ['A', 'A', 'A', 'A', 'A'],
    ['B', 'A', 'A', 'B', 'B'],
    ['B', 'A', 'A', 'B', 'B'],
    ['A', 'B', 'B', 'A', 'B'],
    ['B', 'B', 'B', 'B', 'A'],
    ['A', 'A', 'B', 'A', 'B'],
    ['B', 'B', 'B', 'B', 'A'],
    ['A', 'A', 'B', 'B', 'B'],
    ['B', 'B', 'B', 'B', 'B'],
    ['A', 'A', 'B', 'A', 'A'],
    ['B', 'B', 'B', 'A', 'A'],
    ['B', 'B', 'A', 'A', 'B'],
    ['B', 'B', 'B', 'B', 'A'],
    ['B', 'A', 'B', 'A', 'B'],
    ['A', 'B', 'B', 'B', 'A'],
    ['A', 'B', 'A', 'B', 'B'],
    ['B', 'A', 'B', 'B', 'B'],
    ['A', 'B', 'B', 'B', 'B']
]
y_train_degrees = ['SECOND', 'FIRST', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                   'SECOND', 'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND',
                   'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND', 'FIRST',
                   'SECOND', 'SECOND', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                   'SECOND', 'SECOND']

tree_degrees = \
        ['Attribute', 'att0', 
            ['Value', 'A', 
                ['Attribute', 'att4', 
                    ['Value', 'A', 
                        ['Leaf', 'FIRST', 5, 14]
                    ], 
                    ['Value', 'B', 
                        ['Attribute', 'att3', 
                            ['Value', 'A', 
                                ['Attribute', 'att1', 
                                    ['Value', 'A', 
                                        ['Leaf', 'FIRST', 1, 2]
                                    ], 
                                    ['Value', 'B', 
                                        ['Leaf', 'SECOND', 1, 2]
                                    ]
                                ]
                            ], 
                            ['Value', 'B', 
                                ['Leaf', 'SECOND', 7, 9]
                            ]
                        ]
                    ]
                ]
            ], 
            ['Value', 'B', 
                ['Leaf', 'SECOND', 12, 26]
            ]
        ]

# RQ5 (fake) iPhone purchases dataset
header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
X_train_iphone = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"]
]
y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

tree_iphone = \
        ['Attribute', 'att0', 
            ['Value', 1, 
                ['Attribute', 'att1', 
                    ['Value', 1, 
                        ['Leaf', 'yes', 1, 5]
                    ], 
                    ['Value', 2, 
                        ['Attribute', 'att2', 
                            ['Value', 'excellent', 
                                ['Leaf', 'yes', 1, 2]
                            ], 
                            ['Value', 'fair', 
                                ['Leaf', 'no', 1, 2]
                            ]
                        ]
                    ], 
                    ['Value', 3, 
                        ['Leaf', 'no', 2, 5]
                    ]
                ]
            ], 
            ['Value', 2, 
                ['Attribute', 'att2', 
                    ['Value', 'excellent', 
                        ['Leaf', 'no', 4, 4] # this is a case 3 and a clash, would like num to be 4,10, but not big deal
                        # change denominator to size of partition for that certain case
                    ], 
                    ['Value', 'fair', 
                        ['Leaf', 'yes', 6, 10]
                    ]
                ]
            ]
        ]
'''
def test_decision_tree_classifier_fit():

    decision_tree = MyDecisionTreeClassifier()
    decision_tree.fit(X_train_interview, y_train_interview)
    assert decision_tree.tree == tree_interview

    decision_tree.fit(X_train_degrees, y_train_degrees)
    assert decision_tree.tree == tree_degrees
    
    decision_tree.fit(X_train_iphone, y_train_iphone)
    assert decision_tree.tree == tree_iphone

def test_decision_tree_classifier_predict():

    X_test = [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]
    y_test = ["True", "False"]
    decision_tree = MyDecisionTreeClassifier()
    decision_tree.fit(X_train_interview, y_train_interview)
    y_predicted = decision_tree.predict(X_test)
    assert y_predicted == y_test

    X_test = [["B", "B", "B", "B", "B"], ["A", "A", "A", "A", "A"], ["A", "A", "A", "A", "B"]]
    y_test = ["SECOND", "FIRST", "FIRST"]
    decision_tree.fit(X_train_degrees, y_train_degrees)
    y_predicted = decision_tree.predict(X_test)
    assert y_predicted == y_test

    X_test = [[2, 2, "fair"], [1, 1, "excellent"]]
    y_test = ["yes", "yes"]
    decision_tree.fit(X_train_iphone, y_train_iphone)
    y_predicted = decision_tree.predict(X_test)
    assert y_predicted == y_test
'''

tree1 = \
    ['Attribute', 'att3', 
        ['Value', 'no', 
            ['Attribute', 'att0', 
                ['Value', 'Junior', 
                    ['Leaf', 'True', 4, 8]
                ], 
                ['Value', 'Mid', 
                    ['Leaf', 'True', 1, 8]
                ], 
                ['Value', 'Senior', 
                    ['Attribute', 'att2', 
                        ['Value', 'no', 
                            ['Leaf', 'False', 2, 3]
                        ], 
                        ['Value', 'yes', 
                            ['Leaf', 'True', 1, 3]
                        ]
                    ]
                ]
            ]
        ], 
        ['Value', 'yes', 
            ['Leaf', 'False', 1, 9]
        ]
    ]

tree2 = \
    ['Attribute', 'att0', 
        ['Value', 'Junior', 
            ['Attribute', 'att3', 
                ['Value', 'no', 
                    ['Leaf', 'True', 4, 6]
                ], 
                ['Value', 'yes', 
                    ['Leaf', 'False', 2, 6]
                ]
            ]
        ], 
        ['Value', 'Mid', 
            ['Leaf', 'True', 1, 9]
        ], 
        ['Value', 'Senior', 
            ['Leaf', 'False', 2, 9]
        ]
    ]

tree3 = \
    ['Attribute', 'att1', 
        ['Value', 'Java', 
            ['Leaf', 'False', 3, 9]
        ], 
        ['Value', 'Python', 
            ['Leaf', 'False', 2, 2]
        ], 
        ['Value', 'R', 
            ['Leaf', 'True', 4, 9]
        ]
    ]

# split dataset into test set and "remainder set"
X_train_holdout, X_test_holdout, y_train_holdout, y_test_holdout = \
        myevaluation.train_test_split(X_train_interview, y_train_interview, 0.33, None, False)

def test_random_forest_classifier_fit():
    np.random.seed(0)

    train = [X_train_holdout[i] + [y_train_holdout[i]] for i in range(len(X_train_holdout))]

    random_forest_clf = MyRandomForestClassifier(random_forest=True, N=5, M=3, F=2)
    random_forest_clf.fit(X_train_holdout, y_train_holdout)

    test_trees = []
    test_trees.append(tree1)
    test_trees.append(tree2)
    test_trees.append(tree3)

    for i in range(len(random_forest_clf.trees)):
        assert random_forest_clf.trees[i] == test_trees[i]

def test_random_forest_classifier_predict():
    np.random.seed(0)
    y_pred_sol = ["True", "False", "False", "True", "False"]
    random_forest_clf = MyRandomForestClassifier(random_forest=True, N=5, M=3, F=2)
    random_forest_clf.fit(X_train_holdout, y_train_holdout)
    y_pred = random_forest_clf.predict(X_test_holdout)
    print("y_test:", y_test_holdout)
    print("y_pred:", y_pred)   

    for i in range(len(y_pred)):
        assert y_pred == y_pred_sol

