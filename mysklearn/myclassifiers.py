""" This file contains 4 classifiers: my own kNN classifier, dummy classifier,
and my Naive Bayes classifier, and a Decision Tree Classifier. The dummy classifier simply 
finds the most frequent class label and classifies all instances with that label.
"""
import operator
import itertools
from pickle import NONE
import numpy as np #TODO DELETE
from mysklearn import myutils

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        k = self.n_neighbors
        distances = []
        neighbor_indices = []
        for test_instance in X_test: #this whole loop finds k neighbors for one test instance
            row_index_distances = []
            for i, train_instance in enumerate(self.X_train):
                #numerical attribute
                if isinstance(test_instance[0], str):
                    dist = myutils.compute_categorical_distance(train_instance, test_instance)
                else:
                    dist = myutils.compute_euclidean_distance(train_instance, test_instance)
                row_index_distances.append([i, dist])
            row_index_distances.sort(key=operator.itemgetter(-1)) #sort by item's list at index -1 (distance)
            top_k = row_index_distances[:k] #k closest
            instance_distances = [] #this instance's top k distances
            instance_neighbor_indices = [] #this instance's top k indexes
            for row in top_k:
                instance_neighbor_indices.append(row[0])
                instance_distances.append(row[1])
            distances.append(instance_distances) # append this list to the list of all x_test distances
            neighbor_indices.append(instance_neighbor_indices)
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        self.fit(self.X_train, self.y_train) #fit to data
        distances, neighbor_indices = self.kneighbors(X_test)
        predictions = []
        for i in range(len(X_test)):
            instance_prediction_labels = []
            for index in neighbor_indices[i]: #neighbor indices for this x_test instance
                instance_prediction_labels.append(self.y_train[index])
            y_prediction = myutils.find_highest_frequency_name(instance_prediction_labels)
            predictions.append(y_prediction)
        return predictions

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.
    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()
    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.
        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        self.most_common_label = myutils.find_highest_frequency_name(y_train)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = [self.most_common_label] * len(X_test)
        return y_predicted


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(list float): The prior probabilities computed for each
            label in the training set.
        posteriors(list list float): The posterior probabilities computed for each
            attribute value/label pair in the training set.
        all_attribute_options(list): a list of all the avaliable options for attribute
            values, and parallel to posteriors
        class_labels(list): list of the dataset class labels, in numerical/alphabetical order

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None
        self.all_attribute_options = None
        self.class_labels = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        #find all class labels
        self.class_labels = sorted(list(set(y_train)))
        num_instances = len(y_train)

        #group by class
        class_groups = myutils.group_by_class(X_train, y_train, self.class_labels) #parallel to class_labels

        #compute priors
        priors = [0] * len(self.class_labels) #parallel to class_labels
        for i in range(len(class_groups)):
            priors[i] = len(class_groups[i])/num_instances
        self.priors = priors

        #compute posteriors
        self.all_attribute_options = myutils.all_attribute_options(X_train)
        posteriors = [[] for op_group in self.all_attribute_options]
        #go thru options for each attrubute and count num, add to posteriors
        for col_i in range(len(self.all_attribute_options)):
            for value in self.all_attribute_options[col_i]:
               #for each option value, create a row in posteriors:
                new_value_fracs = []
                #for each class, look at all the instances in the class group
                for class_i in range(len(class_groups)):
                    counter = 0
                    for instance in class_groups[class_i]:
                        if instance[col_i] == value:
                            counter += 1
                    new_value_fracs.append(counter/len(class_groups[class_i]))
                posteriors[col_i].append(new_value_fracs)
        self.posteriors = posteriors

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        y_predicted = []
        for test_instance in X_test:
            class_probabilities = []
            for class_i in range(len(self.priors)):
                class_probability = self.priors[class_i] #then multiply by each att probability
                for test_col in range(len(test_instance)):
                    for att_op_index in range(len(self.all_attribute_options[test_col])):
                        if test_instance[test_col] == self.all_attribute_options[test_col][att_op_index]:
                            class_probability *= self.posteriors[test_col][att_op_index][class_i]
                class_probabilities.append(class_probability)
            prediction = self.class_labels[class_probabilities.index(max(class_probabilities))]
            y_predicted.append(prediction)
        return y_predicted

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
        self.header = None
        self.attribute_domains = None

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
        self.X_train = X_train
        self.y_train = y_train

        #programatically create a header (["att1", "att2"])
        self.header = ["att" + str(ind) for ind in range(len(self.X_train[0]))]
        #print("HEADER:", self.header)
        #AND create an attribute domains dictionary (example: attribute_domains)
        self.attribute_domains = {ind:[] for ind in range(len(self.header))}
        for instance in self.X_train:
            for val_ind in range(len(instance)):
                if instance[val_ind] not in self.attribute_domains[val_ind]:
                    self.attribute_domains[val_ind].append(instance[val_ind])
                    self.attribute_domains[val_ind].sort() #alphabetical order*
        
        #print("ATTRIBUTE DOMAINS:", self.attribute_domains)

        # STITCH X_train and y_train together
        train = [self.X_train[i] + [self.y_train[i]] for i in range (len(self.X_train))]

        #make a copy of header because tdidt() will modify the list
        available_attributes = self.header.copy()
        tree = myutils.tdidt(train, available_attributes, self.header, self.attribute_domains)
        #print("FINAL TREE:", tree)
        self.tree = tree
        # note: unit test will assert tree == interview_tree_solution (MIND THE ORDER)
    

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for instance in X_test:
            prediction = self.tdidt_predict(self.header, self.tree, instance)
            y_predicted.append(prediction)

        return y_predicted
    
    def tdidt_predict(self, header, tree, instance):
        #recursive function to traverse tree
        #figure out where we are in the tree - leaf node? att node?
        info_type = tree[0]
        if info_type == "Leaf": #BASE CASE
            return tree[1]
        #match instance's attribute value with the right att list in tree
        #for loop that traverses thru value list
        # recurse on match w instance's value
        att_index = header.index(tree[1])
        for i in range(2, len(tree)):
            value_list = tree[i]
            if(value_list[1] == instance[att_index]):
                return self.tdidt_predict(header, value_list[2], instance)

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        print("\n\n========DECISION RULES=======\n")
        if attribute_names is None:
            attribute_names = self.header
        instance = ["Mid", "Python", "no", "no"]
        
        all_domains = self.attribute_domains.values()
        all_poss_instances = list(itertools.product(*all_domains))
        for instance in all_poss_instances:
            print(instance)
            clas = self.tdidt_predict(self.header, self.tree, instance)
            print("IF", end=" ")
            if len(attribute_names) > 1:
                for i in range(len(attribute_names) - 1):
                    print(attribute_names[i], "==", instance[i], "AND", end=" ")
            print(attribute_names[len(attribute_names)], "==", instance[len(attribute_names)], "THEN", end=" ")
            print(class_name, "=", clas)

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this