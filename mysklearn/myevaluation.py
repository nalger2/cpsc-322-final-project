"""This module contains classifier evaluation techniques implemented manually, modeled off
of sklearn.
"""
import math
import numpy as np # for random #s
from mysklearn import myutils
from mysklearn.mypytable import MyPyTable

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code (NOTE using numpy)
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!
    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    np.random.seed(random_state) #seed random
    num_instances = len(X)

    #shuffle in place
    if shuffle is True:
        myutils.randomize_in_place(X, random_state, parallel_lists=[y])

    #num instances based on test size
    if test_size < 1: # int - proportional
        num_test_instances = math.ceil(num_instances * test_size)
    else: # float - absolute
        num_test_instances = test_size
    num_train_instances = num_instances - num_test_instances

    #create tables
    X_train = [X[i] for i in range(num_train_instances)] #i.e. indexes [0, 3)
    y_train = [y[i] for i in range(num_train_instances)]
    X_test = [X[i] for i in range(num_train_instances, num_instances)] #i.e. indexes [3, 5)
    y_test = [y[i] for i in range(num_train_instances, num_instances)]
    return X_train, X_test, y_train, y_test

def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold
    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    #seed random state
    np.random.seed(random_state)
    #shuffle if true
    X_indexes = list(range(len(X))) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if shuffle is True:
        myutils.randomize_in_place(X, random_state, parallel_lists=[X_indexes])

    #create list of n_splits folds (lists)
    folds = [[] for i in range(n_splits)] #[[], [], [], [], []]

    #"deal" append indexes from len(X) into folds
    folds = [X_indexes[i::n_splits] for i in range(n_splits)]

    X_test_folds = list(folds)
    X_train_folds = [[i for i in X_indexes if i not in fold] for fold in folds] #

    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.
    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.
    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    #seed random state
    np.random.seed(random_state)
    #shuffle if true
    X_indexes = list(range(len(X))) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if shuffle is True:
        myutils.randomize_in_place(X, random_state, parallel_lists=[y, X_indexes])

    #HAVE TO SEPARATE CLASSES TO STRATIFY
    #make mypytable of values
    X_table = MyPyTable(column_names=["feature " + str(i) for i in range(len(X[0]))], data=X)
    #add a rows for index and classlabels
    X_table.column_names.append("class label")
    X_table.column_names.insert(0, "index")
    for i in range(len(X)):
        X_table.data[i].insert(0, X_indexes[i])
        X_table.data[i].append(y[i])

    #GROUP BY class label
    class_tables = myutils.group_by(X_table.data, X_table.column_names, "class label")

    #create list of n_splits folds (lists)
    folds = [[] for i in range(n_splits)] #[[], [], [], [], []]

    #make a new list of indexes now that the class labels are sorted one then the other
    indexes = []
    for class_group in class_tables:
        for instance in class_group:
            indexes.append(instance[0]) #instance[0] is the instance's index

    #"deal" append indexes from len(X) into folds
    folds = [indexes[i::n_splits] for i in range(n_splits)]

    X_test_folds = list(folds)
    X_train_folds = [[i for i in indexes if i not in fold] for fold in folds] #

    return X_train_folds, X_test_folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.
    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """
    #seed random
    np.random.seed(random_state)
    X_sample = []
    y_sample = []
    if n_samples is None: #default value
        n_samples = len(X)
    #collect samples
    for n in range(n_samples):
        rand_index = np.random.randint(0, len(X))
        X_sample.append(X[rand_index])
        if y is not None:
            y_sample.append(y[rand_index])
    #leftovers
    X_out_of_bag = []
    y_out_of_bag = []
    for i in range(len(X)):
        if X[i] not in X_sample:
            X_out_of_bag.append(X[i])
            if y is not None:
                y_out_of_bag.append(y[i])
    if y is None:
        y_out_of_bag = None
        y_sample = None
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix
    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class
    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []
    for label in labels:
        row = [0 for label in labels] #[0, 0, 0]
        matrix.append(row) #[[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(len(labels)): #check true
        for j in range(len(labels)): #check predicted
            for n in range(len(y_true)): #iterate through number of samples
                if y_true[n] == labels[i] and y_pred[n] == labels[j]:
                    matrix[i][j] += 1
    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.
    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).
    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    correctly_predicted = 0
    all_predicted = 0
    for i in range(len(y_true)): #run through n_samples
        if y_true[i] == y_pred[i]:
            correctly_predicted += 1
        all_predicted += 1
    #print(correctly_predicted, all_predicted)
    if normalize is True:
        return correctly_predicted / all_predicted
    return correctly_predicted

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    #ensure order of labels
    labels = myutils.reorder_labels(labels, pos_label)
    matrix = confusion_matrix(y_true, y_pred, labels)
    tp, fp, tn, fn = myutils.binary_confusion_matrix_labels(matrix)
    if tp == 0 and fp == 0: #error for 0/0 = 0
        return 0
    precision = tp / (tp + fp)
    #print("precision", tp, "/", tp+fp, precision)
    return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    labels = myutils.reorder_labels(labels, pos_label)
    matrix = confusion_matrix(y_true, y_pred, labels)
    tp, fp, tn, fn = myutils.binary_confusion_matrix_labels(matrix)
    if(tp + fn) != 0:
        recall = tp / (tp + fn)
    else: recall=0
    #print("recall", tp, "/", tp+fn, recall)
    return recall

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(y_true, y_pred, labels=labels, pos_label=pos_label)
    recall = binary_recall_score(y_true, y_pred, labels=labels, pos_label=pos_label)
    if precision + recall == 0:
        return 0
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
