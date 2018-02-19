import random
import math


class NeuralNetwork:

    def __init__(self, node_num):
        self.inputs = []
        self.node_layer = []
        self.output = []
        self.node_num = node_num
        count = 0
        while count < node_num:
            neuron = Neuron()
            self.node_layer.append(neuron)
            count += 1

    def fit(self, training_data, training_targets):
        """Takes in a numpy array of training data and a list of targets
        and creates connections between the inputs and the node layer.
        It then iterates through each row in the training data and determines the
        h and a values for each node to find the output"""
        # first determine the number of attributes
        num_attributes = training_data.shape[1]
        for node in self.node_layer:
            node.create_random_weights(num_attributes)

        for row in training_data:
            node_instance_ouput = []
            for node in self.node_layer:
                node.determine_h_value(row)
                node_instance_ouput.append(node.determine_a_value())
            self.output.append(node_instance_ouput)
            print("output for this row was {}".format(node_instance_ouput))


class Neuron:

    def __init__(self):
        self.weights = []
        self.h = None
        self.a = None

    def determine_h_value(self, inputs):
        input_sum = 0
        for index, item in enumerate(inputs):
            input_sum += inputs[index] * self.weights[index]
        # finally add the value for the bias node
        input_sum += (-1 * self.weights[-1])
        self.h = input_sum
        print("h value = {}".format(self.h))

    def determine_a_value(self):
        self.a = 1 / (1 + math.exp(-1 * self.h))
        print("a value = {}".format(self.a))
        if self.a < 0.5:
            return 0
        else:
            return 1

    def create_random_weights(self, input_number):
        count = 0
        # here we set it less than or equal so we add an extra weight for the bias node
        while count <= input_number:
            weight = random.uniform(-0.4, 0.4)
            # print("weight {}: {}".format(count, weight))
            self.weights.append(weight)
            count += 1


