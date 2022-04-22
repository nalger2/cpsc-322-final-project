"""
This is a utility functions file with reusable functions I have
implemented in mysklearn classes
"""
import math
import copy
import numpy as np
from tabulate import tabulate

from mysklearn import myevaluation, mypytable

def hi_low_discretizor(value):
    """Discretizes a value into 2 categories: high or low
    Args:
        value(int): value to discretize
    Returns:
        new_val(str): discretized string value
    """
    if value >= 100:
        new_val = "high"
    else:
        new_val = "low"
    return new_val

def pos_neg_discretizor(value):
    """Discretizes a value into 2 categories: positive or negative
    Args:
        value(int): value to discretize
    Returns:
        new_val(str): discretized string value
    """
    if value >= 100:
        new_val = "high"
    else:
        new_val = "low"
    return new_val

def compute_euclidean_distance(v1, v2):
    """Computes euclidean distance between two points
    Args:
        v1(list int or float): point [x,y]
        v2(list int or float): point [x,y]
    Returns:
        dist(float): euclidean distance
    """
    under_sum = [(v1[i] - v2[i]) ** 2 for i in range(len(v1))]
    summed = sum(under_sum)
    dist = math.sqrt(summed)
    return dist

def compute_categorical_distance(v1, v2):
    """computes "distance" between two points, when the points are categorical attributes
    in the form of strings
    Args:
        v1(list str): instance's attributes
        v2(list str): other instance's attributes
    Returns:
        dist(int): 1 if different, 0 if same
    """
    if v1 == v2:
        return 0
    else:
        return 1

def min_max_normalize(list_vals):
    """Normalizes a list of values
    Args:
        list_vals(list float): list of values to be normalized
    Returns:
        normalize_list(list float): list of values normalized
    """
    list_max = max(list_vals)
    list_min = min(list_vals)
    normalized_list = []
    for val in list_vals:
        normalized_val = (val - list_min) / (list_max - list_min)
        normalized_list.append(normalized_val)
    return normalized_list

def get_frequencies_nominal(list_names):
    """This reusable function gathers the frequencies of a NOMINAL list
    Args:
        list_names (list str): list of names
    Returns:
        freq(dict): dictionary of names with their given frequencies
    """
    freq = {}
    for name in list_names:
        if name in freq:
            freq[name] += 1
        else:
            freq[name] = 1
    return freq #returns frequencies dictionary

def get_frequencies_numerical(list_vals):
    """This reusable function gathers the frequencies of a numerical list of values
    Args:
        list_vals (list int or float): list of numbers
    Returns:
        values(list int or float): list of the various values found in the column
        counts(list int): list of the counts for each of the values in values (parallel list)
    """
    list_vals.sort() #in place sort
    values = []
    counts = []
    for val in list_vals:
        if val in values:
            counts[-1] += 1
        else:
            values.append(val)
            counts.append(1)
    return values, counts

def find_highest_frequency_name(list_names):
    """Returns the highest occuring class label (numerical or nominal) by finding frequencies
    Arguments:
        list_names (list str): list of class labels
    Returns:
        highest_freq_name(str): name with the highest frequency
    """
    if isinstance(list_names[0], str): #if names are strings
        freqs = get_frequencies_nominal(list_names) #dictionary
        highest_freq_name = max(freqs, key=freqs.get)#accesses max value freq's key name
    else: #if items in list are numbers
        vals, freqs = get_frequencies_numerical(list_names)
        highest_freq_name = vals[freqs.index(max(freqs))]
    return highest_freq_name

def auto_discretizer(mpg):
    """Discretizes an mpg value into the DOE ranking categories
    Args:
        mpg(float): a given mpg value
    Returns:
        rating(int): the car's rating based on it's mpg
    """
    if mpg <= 13:
        rating = 1
    elif mpg == 14:
        rating = 2
    elif mpg <= 16:
        rating = 3
    elif mpg <= 19:
        rating = 4
    elif mpg <= 23:
        rating = 5
    elif mpg <= 26:
        rating = 6
    elif mpg <= 30:
        rating = 7
    elif mpg <= 36:
        rating = 8
    elif mpg <= 44:
        rating = 9
    else:
        rating = 10
    return rating

def pretty_print_classifier_result(step_num, classifier_name, test_set, predictions, actuals):
    """Pretty prints the classifying results of a classifier for PA4 assignment steps.
    Args:
        step_num(int): step number in the assignment
        classifier_name(str): name of classifier used
        test_set(MyPyTable): test set of data
        predictions(list list): list of predictions by this classifier
        actuals(list): list of actual values for each test instance
    """
    print("===========================================")
    print("STEP", step_num, ":", classifier_name)
    print("===========================================")
    for i in range(len(test_set.data)):
        print("Instance:", test_set.data[i])
        print("Class:", predictions[i][0], "Actual:", actuals[i])
    correct=0
    for i in range(len(predictions)):
        if predictions[i][0] == actuals[i]:
            correct += 1
    accuracy = correct / len(predictions)
    print("Accuracy:", accuracy)

def randomize_in_place(alist, random_state, parallel_lists=None): #useful for train test split, etc in PA5
    """Shuffles (randomizes) a list and parallel list in place.
    Args:
        alist (list): list to randomize
        random_state(int): to seed random state for reproducible results
        parallel_list (list): parallel list to alist for shuffling
    """
    np.random.seed(random_state)
    for i in range(len(alist)):
        #generate random index to swap value at i with
        rand_index = np.random.randint(0, len(alist)) #[0, len(alist))
        #swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_lists is not None:
            for parallel_list in parallel_lists:
                parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]

def get_column(table, header, col_name):
    """Reusable function accesses a column in a given table based on the column name
    Args:
        table (list list): table of data
        header(list str): column header names
        col_name(str): column name to access and return
    Returns:
        column(list): the desired column in list form
    """
    col_index = header.index(col_name)
    column = []
    for row in table:
        value = row[col_index]
        if value not in ('NA', 'N/A', ''):
            column.append(value)
    return column

def group_by(table, header, groupby_col_name):
    """TODO doc string
    """
    groupby_col_index = header.index(groupby_col_name)
    groupby_col = get_column(table, header, groupby_col_name)
    group_names = sorted(list(set(groupby_col)))
    groupby_subtables = [[] for _ in group_names] #e.g. [[],[],[]]

    for row in table:
        groupby_val = row[groupby_col_index] #this row's value at this index
        groupby_val_subtable_index = group_names.index(groupby_val)
        groupby_subtables[groupby_val_subtable_index].append(row.copy()) #make a copy

    return group_names, groupby_subtables

def pretty_print_step(step_name):
    print("===========================================")
    print(step_name)
    print("===========================================")

def pretty_print_evaluation_results(method, kNN_acc, dummy_acc):
    print(method)
    print("k Nearest Neighbors Classifier: accuracy =", round(kNN_acc, 3), "error rate =", round(1.0 - kNN_acc, 3))
    print("Dummy Classifier: accuracy =", round(dummy_acc, 3), "error rate =", round(1.0 - dummy_acc, 3))

def group_by_class(X_train, y_train, classes):
    """PA6 - TODO"""
    groupby_subtables = [[] for _ in classes]
    for row_index in range(len(X_train)):
        groupby_val = y_train[row_index]
        groupby_val_subtable_index = classes.index(groupby_val)
        groupby_subtables[groupby_val_subtable_index].append(X_train[row_index].copy()) #make a copy
    return groupby_subtables

def all_attribute_options(X_train):
    num_cols = len(X_train[0])
    all_options = [[] for _ in range(num_cols)]
    for col in range(num_cols):
        for row in X_train:
            if row[col] not in all_options[col]:
                all_options[col].append(row[col])
    for att in all_options:
        att.sort()
    return all_options

def one_fold_splits(X, y, X_train_folds, X_test_folds, fold_index):
    X_train = []
    y_train = []
    for index in X_train_folds[fold_index]:
        X_train.append(X[index])
        y_train.append(y[index])
    X_test = []
    y_test = []
    for index in X_test_folds[fold_index]:
        X_test.append(X[index])
        y_test.append(y[index])
    return X_train, y_train, X_test, y_test

def reorder_labels(labels, pos_label):
    """"Reorders labels if positive label is not alread at position 0

    Args:
        labels(list): list of class labels
        pos_label(): desired positive label
    Returns:
        labeles(list): reordered if necessary list of lcass labels
    """
    if pos_label is not None:
        if labels[0] != pos_label:
            labels[0], labels[1] = labels[1], labels[0] #switch labels order
    return labels

def binary_confusion_matrix_labels(matrix):
    """Returns positional labels for each location in a binary matrix, assuming pos label
    is at positions 0 and 0

    Args:
        matrix(list list int): confusion matrix for a classifier

    Returns:
        tp (int): true positives
        fp (int): false positives
        tn (int): true negatives
        fn (int): false negatives
    """
    tp = matrix[0][0]
    fp = matrix[0][1]
    tn = matrix[1][1]
    fn = matrix[1][0]
    return tp, fp, tn, fn

def package_classifier_results(y_true, y_pred, labels):
    """neatly packages all evaluation results of a binary classifier

    Args:
        y_true(list): list of true class labels
        y_pred(list): list of classifier's predicted class labels
    Returns:
        accuracy(float): tp + tn / p + n
        error(float): 1 - accuracy
        precision(float): tp / tp + fp
        recall(float): tp / tp + fn
        f1(float): 2 * precision * recall / precision + recall
        matrix(list list int): confusion matrix
    """
    accuracy = myevaluation.accuracy_score(y_true, y_pred)
    error = 1 - accuracy
    precision = myevaluation.binary_precision_score(y_true, y_pred, labels=labels)
    recall = myevaluation.binary_recall_score(y_true, y_pred, labels=labels)
    f1 = myevaluation.binary_f1_score(y_true, y_pred, labels=labels)
    matrix = myevaluation.confusion_matrix(y_true, y_pred, labels)
    return accuracy, error, precision, recall, f1, matrix

def pa6_pretty_print_results(classifier, accuracy, error, precision, recall, f1):
    """pretty prints a pa6 classifier result"""
    print("\n=====================")
    print(classifier)
    print("=====================")
    print("Accuracy:\t", round(accuracy, 2))
    print("Error Rate:\t", round(error, 2))
    print("Precision:\t", round(precision, 2))
    print("Recall:\t\t", round(recall, 2))
    print("F1 measure:\t", round(f1, 2))

def show_results(true, pred, labels, title):
    accuracy, error, precision, recall, f1, matrix = package_classifier_results(true, pred, labels)
    pa6_pretty_print_results(title, accuracy, error, precision, recall, f1)
    print("Confusion matrix:\n", tabulate(matrix, headers=labels, showindex=labels))

"""PA7 DECISION TREE FUNCTIONS"""

def tdidt(current_instances, available_attributes, header, attribute_domains): #makes a tree out of x_train and y_train
    # basic approach (uses recursion!!):
    #print("\nRECURSION!\n")
    #print("available attributes:", available_attributes)

    # select an attribute to split on (random, entropy, etc.)
    attribute = select_attribute(current_instances, available_attributes, attribute_domains) 
    #print("splitting on", attribute)
    available_attributes.remove(attribute) #remove attribute from avail to split on
    tree = ["Attribute", attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    attribute_index = header.index(attribute) #find this attribute's index in header (also att domains dict key)
    partitions = partition_instances(current_instances, attribute_index, attribute_domains[attribute_index])

    # for each partition, repeat unless one of the following occurs (base case)
    for att_value, att_partition in partitions.items():
        #print("\nLooping partition", att_value)
        #print("current att value:", att_value, "--number of instances:", len(att_partition))
        value_subtree = ["Value", att_value] 
        
        # CASE 1: all class labels of the partition are the same => make a leaf node
        if len(att_partition) and all_same_class(att_partition):
            #print("CASE 1 all same class")
            #print("appending leaf node")
            leaf_class = att_partition[0][-1]
            leaf = ["Leaf", leaf_class, len(att_partition), len(current_instances)]
            value_subtree.append(leaf)
            tree.append(value_subtree)

        
        # CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            #print("CASE 2 no more attributes")
            #print("finding majority class...")
            mv_class = majority_vote(att_partition)
            #print("majority class:", mv_class)
            leaf = ["Leaf", mv_class, len(att_partition), len(current_instances)]
            value_subtree.append(leaf)
            tree.append(value_subtree)

        # CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(att_partition) == 0:
            #print("CASE 3 empty partition")
            keys = list(partitions.keys())
            this_ind = keys.index(att_value)
            prev_ind = keys[this_ind - 1]
            prev = partitions[prev_ind]
            mv = majority_vote(prev)
            tree = ["Leaf", mv, len(prev), len(current_instances)]
        
        else: #no conditions were true - RECURSE
            subtree = tdidt(att_partition, available_attributes.copy(), header, attribute_domains)
            value_subtree.append(subtree)
            tree.append(value_subtree)
    return tree

def select_attribute(instances, attributes, domains):
    classes = set([i[-1] for i in instances])
    Enews = [] #will be parallel to attributes
    for attribute in attributes:
        att_index = int(attribute[-1]) #access the 0 in att0, etc
        partitions = partition_instances(instances, att_index, domains[att_index])
        this_attribute_entropies = []
        weights = [] 
        for domain_name, domain_partition in partitions.items():
            if len(domain_partition) > 0:
                domain_entropy = calculate_entropy(domain_partition, classes)
                this_attribute_entropies.append(domain_entropy)
                weights.append(len(domain_partition)/len(instances)) #build each domain's weights for E-new
        #calculate E-new
        #weights = [weight / len(instances) for weight in weights]
        att_Enew = 0
        for i in range(len(weights)):
            att_Enew += weights[i] * this_attribute_entropies[i]
        Enews.append(att_Enew)
    return attributes[Enews.index(min(Enews))] #return minimum E-new's attribute

def calculate_entropy(partition, classes):
    class_counts = []
    for c in classes:
        count = 0
        for instance in partition:
            if instance[-1] == c:
                count += 1
        class_counts.append(count)
    fracs = [count / sum(class_counts) for count in class_counts]
    entropy = 0
    for frac in fracs:
        if frac == 0: #handle log of 0 error
            entropy = 0
        else:
            entropy -= (frac * math.log(frac, 2))
    return entropy


def partition_instances(instances, split_attribute_index, att_domains_list):
    #group by attribute domain
    partitions = {} #key (attribute value) : value (subtable)
    for att_value in att_domains_list:
        partitions[att_value] = []
        for instance in instances:
            if instance[split_attribute_index] == att_value:
                partitions[att_value].append(instance)
    return partitions

def all_same_class(list_instances):
    #list of classes
    classes = [i[-1] for i in list_instances]
    if (len(set(classes))==1): #all the same class
        return True
    return False

def majority_vote(list_instances):
    classes = [i[-1] for i in list_instances]
    #print("classes:", classes)
    class_labels = set(classes)
    counts = {c:0 for c in class_labels}
    print("counts", counts)
    for instance in list_instances:
        counts[instance[-1]] += 1
    #if class counts are all equal, alphabetic first class
    if (len(set(counts.values()))==1):
        return min(class_labels)
    max_class = max(counts, key=counts.get)
    return max_class

"""Utils below specifically for final project"""

def general_numerize(stroke_data, col_to_numerize, dic_strings_to_nums):
    col_index = stroke_data.get_col_index(col_to_numerize)
    for row in stroke_data.data:
        val = row[col_index]
        row[col_index] = dic_strings_to_nums[val]

def discretize_by_ten(age):
    #returns 0 for  0-9, 1 for 10-19 etc
    return age // 10

def discretize_glucose(gluc):
    if gluc <= 90:
        rating = 1
    elif gluc == 130:
        rating = 2
    elif gluc <= 170:
        rating = 3
    elif gluc <= 210:
        rating = 4
    elif gluc <= 250:
        rating = 5
    else:
        rating = 6
    return rating

    
def discretize_attributes_for_stroke_classification(stroke_data):
    stroke_data_discretized = copy.deepcopy(stroke_data)
    #remove irrelevant column: ID
    stroke_data_discretized.remove_columns([0])

    #discretize
    for row in stroke_data_discretized.data:
        age = row[stroke_data_discretized.get_col_index("age")]
        row[stroke_data_discretized.get_col_index("age")] = discretize_by_ten(age)
        glucose = row[stroke_data_discretized.get_col_index("avg_glucose_level")]
        row[stroke_data_discretized.get_col_index("avg_glucose_level")] = discretize_glucose(glucose)
        bmi = row[stroke_data_discretized.get_col_index("bmi")]
        row[stroke_data_discretized.get_col_index("bmi")] = discretize_by_ten(bmi)

    return stroke_data_discretized

def numerize_all_strings(discretized_data):
    stroke_data_numerized = copy.deepcopy(discretized_data)
    #numerize
    gender_dic = {"Female": 0, "Male": 1, "Other": 2}
    married_dic = {"No": 0, "Yes": 1}
    work_type_dic = {'Govt_job':0, 'Never_worked':1, 'Private':2, 'Self-employed':3, 'children':4}
    residence_dic = {'Rural':0, 'Urban':1}
    smoking_dic = {'Unknown':0, 'formerly smoked':1, 'never smoked':2, 'smokes':3}
    general_numerize(stroke_data_numerized, "gender", gender_dic)
    general_numerize(stroke_data_numerized, "ever_married", married_dic)
    general_numerize(stroke_data_numerized, "work_type", work_type_dic)
    general_numerize(stroke_data_numerized, "Residence_type", residence_dic)
    general_numerize(stroke_data_numerized, "smoking_status", smoking_dic)
    return stroke_data_numerized
