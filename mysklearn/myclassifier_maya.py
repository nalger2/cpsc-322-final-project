import operator
import math

#from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor

def group_by_tree(instances, attribute_name, index):
    seen = []
    for row in instances:
        if row[index] not in seen:
            seen.append(row[index])
    #print(seen)
    groupby_instances = []
    for value in seen:
        for row in instances:
            if row[index] == value:
                groupby_instances.append(row)

    return groupby_instances, seen

def calculate_entropy(self, instances, attributes, index, header):
    header_index = header.index(attributes)
    grouped_instances, available_attributes = group_by_tree(instances, attributes, header_index)

    att_entropy = []
    enew = []
    len_of_att = []
    for att in available_attributes: # value in seen
        instances_for_att = []
        for row in grouped_instances:
            if row[header_index] == att:
                instances_for_att.append(row)
        len_of_att.append(len(instances_for_att))
        available_class_labels = []
        for row in instances_for_att:
            if row[-1] not in available_class_labels:
                available_class_labels.append(row[-1])
        class_label_counts = [] # values of num of class labels for each class_label, should be parallel to class_label
        for class_label in available_class_labels:
            count = 0
            for row in instances_for_att: # for row (in curr instance) - 2D list of groupby instances
                if row[-1] == class_label: # if the curr partition class label == class_label
                    count += 1
            class_label_counts.append(count)
        # calculating entropy
        total = 0
        for label_amt in class_label_counts:
            fraction = label_amt / len(instances_for_att)
            value = (-fraction) * (math.log2(fraction))
            #print("VALUE", value)
            total = total + value
        att_entropy.append(total)
    total = 0
    for value in att_entropy:
        att_index = att_entropy.index(value)
        curr_len = len_of_att[att_index]
        # get fraction for calculating Enew
        fraction = curr_len / len(self.X_train)
        total = total + (fraction) * (value)
    enew.append(total)

    return enew

def select_attribute(self, instances, attributes, header):

    enew_all_att = []
    for class_label in attributes:
        entropy = 0
        Enew = 0
        class_index = attributes.index(class_label)
        enew_for_att = calculate_entropy(self, instances, class_label, class_index, header)
        enew_all_att.append(enew_for_att)

    split_on_value = min(enew_all_att)
    split_on_index = enew_all_att.index(split_on_value)
    split_on_att = attributes[split_on_index]

    return split_on_att

def partition_instances(self, instances, split_attribute, header, attribute_domains):

    partitions = {} # key (attribute value): value (subtable)
    att_index = header.index(split_attribute) # e.g. level -> 0
    att_domain = attribute_domains[att_index] # e.g. ["Junior", "Mid", "Senior"]
    for att_value in att_domain:
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    return partitions

def all_same_class(self, att_partition):

    first_label = att_partition[0][-1]
    for partition in att_partition:
        if partition[-1] != first_label:
            return False

    return True

def tdidt(self, current_instances, available_attributes, header, attribute_domain):

    # select an attribute to split on
    attribute = select_attribute(self, current_instances, available_attributes, header)
    available_attributes.remove(attribute) # can't split on this again in
    # this subtree
    tree = ["Attribute", attribute] # start to build the tree!!

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(self, current_instances, attribute, header, attribute_domain)
    for att_value, att_partition in partitions.items():
        case_3 = False # boolean value for case 3
        value_subtree = ["Value", att_value] # list starting with "value"

        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(att_partition) > 0 and all_same_class(self, att_partition):
            leaf_node_value = att_partition[0][-1]
            total_instances = []
            for key in partitions:
                value_items = partitions[key]
                for row in value_items:
                    total_instances.append(row)
            leaf_node = ["Leaf", leaf_node_value, len(att_partition), len(total_instances)]
            value_subtree.append(leaf_node)
            tree.append(value_subtree)

        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            max_length = 0
            # find all class labels
            class_labels = []
            for label in self.y_train:
                if label not in class_labels:
                    class_labels.append(label)

            max_length = 0
            for label in class_labels:
                ct = 0
                for row in att_partition:
                    if label == row[-1]:
                        ct += 1
                if ct > max_length:
                    max_length = ct
                    max_label = label

            leaf_node_value = max_label
            leaf_node = ["Leaf", leaf_node_value, max_length, len(att_partition)]
            value_subtree.append(leaf_node)
            tree.append(value_subtree)

        #    CASE 3: no more instances to partition (empty partition) => backtrack and
        #       replace attribute node with majority vote leaf node
        elif len(att_partition) == 0:
            case_3 = True
            break

        else: # none of the previous conditions were true... recurse!
            subtree = tdidt(self, att_partition, available_attributes.copy(), header, attribute_domain)
            value_subtree.append(subtree)
            tree.append(value_subtree)

    if case_3 is True:
        # do case 3 calculations
        max_count = 0
        seen = []
        total_instances = []
        for key in partitions:
            value_items = partitions[key]
            for row in value_items:
                total_instances.append(row)

        for row in total_instances:
            if row[-1] not in seen:
                seen.append(row[-1])
        for label in seen:
            count = 0
            for row in total_instances:
                if row[-1] == label:
                    count += 1
            if count > max_count:
                max_count = count # majority vote count
                max_label = label # majority vote label
        leaf_node_value = max_label # majority vote label
        tree = ["Leaf", leaf_node_value, len(total_instances), len(total_instances)] # same ratio because its for entire attribute

    return tree

def tdidt_predict(self, header, tree, instance):
    data_type = tree[0]
    if data_type == "Leaf":
        return tree[1]

    att_index = header.index(tree[1])
    for i in range(2, len(tree)):
        value_list = tree[i]
        if value_list[1] == instance[att_index]:
            return tdidt_predict(self, header, value_list[2], instance)

def predict_decision(self, header, tree, class_label):

    if tree[0] == "Leaf":
        leaf_list = tree[0]
        print("THEN", class_label, "=", end=' ')
        leaf_label = tree[1]
        print(leaf_label, end='\n')
        return

    if tree[0] == "Attribute":
        att_label = tree[1]
        print(att_label, "=", end=' ')

        value_list = tree[2]
        value_label = value_list[1]
        print(value_label, end=' ')

        if value_list[2] == "Leaf":
            print("found a leaf")
            leaf_list = value_list[2]
            leaf_label = leaf_list[1]
            print(leaf_label)
            return

        else:
            if value_list[2][0] != "Leaf":
                print("AND", end=' ')
            predict_decision(self, header, value_list[2], class_label)



class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # need fit_starter_code in testing because need header
        self.X_train = X_train
        self.y_train = y_train

        att_header = []
        attribute_domain = {}
        for row in self.X_train:
            num_att = len(row)
        for i in range(num_att):
            label = "att" + str(i)
            att_header.append(label)
            seen = []
            for row in self.X_train:
                if row[i] not in seen:
                    seen.append(row[i])
            seen.sort()
            attribute_domain[i] = seen

        train = [self.X_train[i] + [self.y_train[i]] for i in range(len(self.X_train))]
        available_attribute = att_header.copy()

        self.tree = tdidt(self, train, available_attribute, att_header, attribute_domain)


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        att_header = []
        for row in X_test:
            num_att = len(row)
        for i in range(num_att):
            label = "att" + str(i)
            att_header.append(label)

        y_predicted = []

        for instance in X_test:
            tree = self.tree
            label = tdidt_predict(self, att_header, tree, instance)
            y_predicted.append(label)


        return y_predicted