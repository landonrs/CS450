from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import numpy as np

import pandas
from filereader import FileReader
from id3_classifier import ID3Classifier
from sklearn import tree as tr


class Shell(object):

    def __init__(self):

        # this is determined by whether we read from a file or not
        self.data_set = None
        self.data_targets = None

        self.file_reader = FileReader()
        # these hold our training and testing values after we split the data
        self.training_data = None
        self.training_targets = None
        self.test_data = None
        self.test_targets = None

        self.most_common = None

        self.classifier = None
        self.model = None

        self.predicted_targets = None

    def prepare_data_set(self, data, targets):
        self.training_data, self.test_data, self.training_targets, self.test_targets = train_test_split(
            data, targets, test_size=0.3)

        self.most_common = max(data[0], key=data.count)

    def make_pd_set(self, column_names):
        self.training_data = pandas.DataFrame(self.training_data, columns=column_names)
        self.test_data = pandas.DataFrame(self.test_data, columns=column_names)


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




def main():
    target_name = 'party'
    shell = Shell()
    data_set = shell.file_reader.get_vote_data()
    column_names = list(data_set.columns)
    shell.prepare_data_set(data_set.values.tolist(), data_set['party'].values.tolist())
    # first test sklearn's implementation
    sk_classifier = tr.DecisionTreeClassifier()
    sk_classifier = sk_classifier.fit(shell.training_data, shell.training_targets)
    shell.predicted_targets = sk_classifier.predict(shell.test_data)
    print("sklearn tree accuracy:")
    shell.determine_accuracy()

    # now testing my implementation
    shell.make_pd_set(column_names)
    tree = ID3Classifier(target_name, shell.most_common)
    tree_model = tree.fit(dataframe=shell.training_data, targets=shell.training_data[target_name].values.tolist(),
                          column_names=list(shell.training_data)[1:])
    print(tree_model.tree_node_list)
    shell.predicted_targets = tree_model.predict(shell.test_data)
    print("my tree's accuracy:")
    shell.determine_accuracy()


main()

