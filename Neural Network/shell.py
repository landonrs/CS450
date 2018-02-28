from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import numpy as np

from sklearn.metrics import mean_absolute_error
from filereader import FileReader
from neuralnetwork import NeuralNetwork

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
        # self.training_data = np.round(self.training_data, 2)
        # self.test_data = np.round(self.test_data, 2)

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

    def average_score(self, scores):
        total_score = 0
        for score in scores:
            total_score += score
        return abs(total_score / 10)


def week_6_main():
    # test_row = np.array([[1, 2]])
    # test_targets = [1]
    shell = Shell()
    # part 1, iris dataset
    print("iris data output")
    shell.data_set, shell.data_targets = shell.file_reader.read_data_from_file("iris.csv")
    shell.prepare_data_set(shell.data_set, shell.data_targets)
    network = NeuralNetwork(num_layers=2, nodes_per_layer=[2, 3], learning_rate=.2)
    network.generate_weights(num_inputs=shell.training_data.shape[1])
    for i in range(1000):
        network.fit(shell.training_data, shell.training_targets)
    shell.predicted_targets = network.fit(shell.test_data, shell.test_targets)
    shell.determine_accuracy()
    print("sklearn implementation for iris")
    sklearn_MLP = MLPClassifier(hidden_layer_sizes=(2, 2), solver='sgd', learning_rate_init=.2, max_iter=1000)
    sklearn_MLP.fit(shell.training_data, shell.training_targets)
    shell.predicted_targets = sklearn_MLP.predict(shell.test_data)
    shell.determine_accuracy()

    # part 2 diabetes dataset
    print("Diabetes Data output:")
    shell.data_set, shell.data_targets = shell.file_reader.read_diabetes_data()
    shell.prepare_data_set(shell.data_set, shell.data_targets)
    network = NeuralNetwork(num_layers=2, nodes_per_layer=[2, 2], learning_rate=.2)
    network.generate_weights(num_inputs=shell.training_data.shape[1])
    for i in range(200):
        network.fit(shell.training_data, shell.training_targets)
    shell.predicted_targets = network.fit(shell.test_data, shell.test_targets)
    shell.determine_accuracy()
    print("sklearn implementation")
    sklearn_MLP = MLPClassifier(hidden_layer_sizes=(2, 2), solver='sgd', learning_rate_init=.2, max_iter=10000)
    sklearn_MLP.fit(shell.training_data, shell.training_targets)
    shell.predicted_targets = sklearn_MLP.predict(shell.test_data)
    shell.determine_accuracy()





week_6_main()
