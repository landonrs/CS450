import math
from sklearn.base import BaseEstimator

class KNNClassifier(BaseEstimator):

    def __init__(self, neighbors):
        self.k = neighbors
        self.knn_model = None

    def fit(self, training_data, training_targets):
        self.knn_model = KNNModel(training_data, training_targets, self.k)
        return self.knn_model


    def predict(self, test_data):
        return self.knn_model.predict(test_data)



class KNNModel:

    def __init__(self, training_data, training_targets, k):
        self.training_data = training_data
        self.training_targets = training_targets
        self.k = k
        if self.k is None:
            self.k = 3

    def predict(self, test_data):
        """measures the distance between each data instance"""
        targets = []

        # for each data point find the target based in the nearest neighbors
        for array in test_data:
            targets.append(self.find_nearest_neighbors(array))

        # print(targets)
        return targets

    def find_nearest_neighbors(self, test_instance):
        """creates an array of nodes that represent the closest data points to the test instance"""
        nearestNeighbors = []
        for index in range(len(self.training_data)):
            distance = self.find_distances(test_instance, self.training_data[index])
            # during the first k iterations, we automatically insert the neighbor into the array
            if index < self.k:
                nearestNeighbors.append(KNode(distance, self.training_targets[index]))
                # print("inserting neighbor " + str(index) + " into neighbor array: distance = " + str(distance))
            else:
                # compare this distance to all other distances in our neighbor array
                for neighbor in range(self.k):
                    # if the distance is closer, insert a new node into the list
                    if distance < nearestNeighbors[neighbor].distance:
                        nearestNeighbors.insert(neighbor, KNode(distance, self.training_targets[index]))
                        # print("found closer neighbor " + str(nearestNeighbors[neighbor].distance))
                        break

        # find the highest occurring target value based on the closest k neighbors
        return self.determine_target_value(nearestNeighbors[0:self.k])

    def find_distances(self, test_data, train_data):
        """determines the distance between train and test data using euclidean distance
        it goes through each item in the test_array and finds the difference from the training data items."""
        distance = 0
        for data_item in range(len(test_data)):
            distance += pow((test_data[data_item] - train_data[data_item]), 2)
        # since we don't need to find the square root to determine the closest neighbor
        # we simply return the number as is
        return distance

    def determine_target_value(self, NN):
        """simply find the highest target frequency of the nearest neighbors"""
        targets = []
        # pull the targets out of the neighbor nodes to form a set
        for node in NN:
            targets.append(node.target)
        target = max(set(targets), key=targets.count)
        return target



class KNode:
    """holds the distance and target data for each data point"""
    def __init__(self, distance, target_value):
        # for comparing distances between data
        self.distance = distance
        # for determining the target value of the closest neighbors
        self.target = target_value
