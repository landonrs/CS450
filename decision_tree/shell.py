from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

import pandas

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from filereader import FileReader
from id3_classifier import ID3Classifier

K_VALUE = 3

class Shell(object):

    def __init__(self):

        self.iris = datasets.load_iris()
        # this is determined by whether we read from a file or not
        self.data_set = None
        self.data_targets = None

        self.file_reader = FileReader()
        # these hold our training and testing values after we split the data
        self.training_data = None
        self.training_targets = None
        self.test_data = None
        self.test_targets = None

        self.classifier = None
        self.model = None

        self.predicted_targets = None

    def prepare_data_set(self, data, targets):
        self.training_data, self.test_data, self.training_targets, self.test_targets = train_test_split(
            data, targets, test_size=0.3)
        # verify that data has been split correctly
        # print("Test Data")
        # for index, target in enumerate(self.test_targets):
        #     print("data:" + str(self.iris.data[index]) + " Target: " + str(target) + " number: " + str(index))

        # Now we will transform the data using the standard scalar class
        scaler = preprocessing.StandardScaler().fit(self.training_data)
        self.training_data = scaler.transform(self.training_data)
        self.test_data = scaler.transform(self.test_data)
        self.training_data = np.round(self.training_data, 2)
        self.test_data = np.round(self.test_data, 2)

    def create_model(self, classifier):
        self.classifier = classifier
        self.model = self.classifier.fit(self.training_data, self.training_targets)
        # print(self.model)

    def predict(self):
        self.predicted_targets = np.array(self.model.predict(self.test_data))
        # print(self.predicted_targets)

    def determine_accuracy(self):
        correct_predictions = 0
        for index, target in enumerate(self.test_targets):
            if self.predicted_targets[index] == self.test_targets[index]:
                correct_predictions += 1
        accuracy = correct_predictions / len(self.test_targets)
        print("%{0:.2f}".format(accuracy * 100))

    def determine_regression_accuracy(self):
        # self.predicted_targets = np.round(self.predicted_targets, 3)
        # print(self.predicted_targets)
        # self.test_targets = np.round(self.test_targets, 3)
        # print(self.test_targets)
        print(mean_absolute_error(self.test_targets, self.predicted_targets))


    def cross_val_test(self, data, targets, test_type):
        if test_type == 'neg_mean_absolute_error':
            skKNN = KNeighborsRegressor(n_neighbors=K_VALUE)
        else:
            skKNN = KNeighborsClassifier(n_neighbors=K_VALUE)

        print("sklearn model: ")
        scores = cross_val_score(skKNN, data, targets, scoring=test_type, cv=10)
        print(scores)
        final_score = self.average_score(scores)
        print(final_score)


    def average_score(self, scores):
        total_score = 0
        for score in scores:
            total_score += score
        return abs(total_score / 10)


def main():
    shell = Shell()
    panda_dataframe = shell.file_reader.test_data()
    # print(list(panda_dataframe)[0:-1])

    tree = ID3Classifier()
    tree_model = tree.fit(panda_dataframe, panda_dataframe['Loan'].values.tolist())
    print(tree_model.tree_node_list)

main()

