from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
# from HardCodedClassifier import *
# from filereader import FileReader


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

    def create_model(self, classifier):
        self.classifier = classifier
        self.model = self.classifier.fit(self.training_data, self.training_targets)
        # print(self.model)

    def predict(self):
        self.predicted_targets = self.model.predict(self.test_data)
        # print(self.predicted_targets)

    def determine_accuracy(self):
        correct_predictions = 0
        for index, target in enumerate(self.test_targets):
            if self.predicted_targets[index] == self.test_targets[index]:
                correct_predictions += 1
        accuracy = correct_predictions / len(self.test_targets)
        print("%{0:.2f}".format(accuracy * 100))


def main():
    shell = Shell()
    decision = input("Load data from file? y/n")

    if decision == 'y':
        shell.data_set, shell.data_targets = shell.file_reader.read_data_from_file("iris.csv")
    else:
        shell.data_set = shell.iris.data
        shell.data_targets = shell.iris.target

    # part 1 - 4 of the assignment
    shell.prepare_data_set(shell.data_set, shell.data_targets)
    shell.create_model(GaussianNB())
    shell.predict()
    shell.determine_accuracy()

    # part 5
    shell.create_model(HardCodedClassifier())
    shell.predict()
    shell.determine_accuracy()

main()
