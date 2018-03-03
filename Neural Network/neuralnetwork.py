import random
import math


class NeuralNetwork:

    def __init__(self, num_layers, nodes_per_layer, learning_rate=0.1):
        """num layers = the number of layers in the network including output
           nodes-per-layer = an array of ints each signifying the number of nodes per layer. The length of this array
           must match the num_layers value"""
        self.inputs = []
        self.node_layers = []
        # self.layer_output = []
        self.num_layers = num_layers
        self.number_of_nodes_per_layer = nodes_per_layer
        self.learning_rate = learning_rate
        if len(self.number_of_nodes_per_layer) != self.num_layers:
            raise ValueError('The length of the nodes_per_layer array does not match the number of layers in the '
                             'network.')
        for i in range(self.num_layers):
            count = 0
            current_node_layer = []
            while count < self.number_of_nodes_per_layer[i]:
                neuron = Neuron()
                current_node_layer.append(neuron)
                count += 1
            self.node_layers.append(current_node_layer)

    def generate_weights(self, num_inputs):
        # first determine the number of attributes
        layer_count = 0
        for node_layer in self.node_layers:
            for node in node_layer:
                # for the first layer, the number of weights generated is determined by the number of inputs
                if layer_count == 0:
                    node.create_random_weights(num_inputs)
                # for every other layer the number of weights is determined by the previous layer
                else:
                    node.create_random_weights(self.number_of_nodes_per_layer[layer_count - 1])
            layer_count += 1

    def print_weights(self):
        for node_layer in self.node_layers:
            for node in node_layer:
                print(node.weights)


    def fit(self, training_data, training_targets):
        """Takes in a numpy array of training data and a list of targets.
        It iterates through each row in the training data and determines the
        h and a values for each node to find the output. It then predicts the class
        given the highest a value of the output nodes. Weights are then adjusted."""
        predicted_targets = []
        layer_output = []
        for index, row in enumerate(training_data):
            layer_count = 0
            layer_output.clear()

            for node_layer in self.node_layers:
                node_instance_ouput = []
                for node in node_layer:
                    if layer_count == 0:
                        node.determine_h_value(row)
                    else:
                        node.determine_h_value(layer_output[layer_count - 1])
                    node_instance_ouput.append(node.determine_a_value())
                layer_output.append(node_instance_ouput)
                # print("output for this row was {}".format(node_instance_ouput))
                layer_count += 1
            predicted_targets.append(self.predict_class(layer_output[-1]))
            self.adjust_weights(training_targets[index], row)
        return predicted_targets

    def predict_class(self, output):
        if self.number_of_nodes_per_layer[-1] != 1:
            highest_a = output[0]
            for a in output:
                if a > highest_a:
                    highest_a = a
            return output.index(highest_a)
        elif output[0] > .5:
            return 1
        else:
            return 0

    def adjust_weights(self, target, inputs):
        # determine the d values for the output layer
        for node_num, node in enumerate(self.node_layers[-1]):
            if self.number_of_nodes_per_layer[-1] != 1:
                if node_num == target:
                    node.determine_output_d_value(1)
                    # print(node.d)
                else:
                    node.determine_output_d_value(0)
            else:
                node.determine_output_d_value(target)


        # now find the d values for every hidden layer
        for layer in range(self.num_layers - 2, -1, -1):
            for node_num, node in enumerate(self.node_layers[layer]):
                node.determine_hidden_d_value(self.node_layers[layer + 1], node_num)

        # finally adjust the weights of each node
        for layer, node_layer in enumerate(self.node_layers):
            # for the first layer, pass in the training data for weight adjustment
            if layer == 0:
                for node in node_layer:
                    node.adjust_input_weights(inputs, self.learning_rate)
            # for every other layer we pass in the nodes of the previous layer
            else:
                for node in node_layer:
                    node.adjust_node_weights(self.node_layers[layer - 1], self.learning_rate)







class Neuron:

    def __init__(self):
        self.weights = []
        self.h = None
        self.a = None
        self.d = None

    def determine_h_value(self, inputs):
        input_sum = 0
        for index, item in enumerate(inputs):
            input_sum += inputs[index] * self.weights[index]
        # finally add the value for the bias node
        input_sum += (-1 * self.weights[-1])
        self.h = input_sum
        # print("h value = {}".format(self.h))

    def determine_a_value(self):
        self.a = 1 / (1 + math.exp(-1 * self.h))
        # print("a value = {}".format(self.a))
        return self.a

    def determine_output_d_value(self, target):
        self.d = self.a * (1 - self.a) * (self.a - target)

    def determine_hidden_d_value(self, k_nodes, node_num):
        error_sum = 0
        for index, item in enumerate(k_nodes):
            error_sum += k_nodes[index].d * k_nodes[index].weights[node_num]
        # now use the error sum in the d calculation
        self.d = self.a * (1 - self.a) * error_sum

    def adjust_input_weights(self, inputs, learning_rate):
        for index, input in enumerate(inputs):
            self.weights[index] = self.weights[index] - (learning_rate * self.d * input)
        # adjust bias node weight
        self.weights[-1] = self.weights[-1] - (learning_rate * self.d * (-1))
        # print(self.weights[-1])

    def adjust_node_weights(self, i_nodes, learning_rate):
        for index, node in enumerate(i_nodes):
            self.weights[index] = self.weights[index] - (learning_rate * self.d * node.a)
        # adjust bias node weight
        self.weights[-1] = self.weights[-1] - (learning_rate * self.d * (-1))

    def create_random_weights(self, input_number):
        count = 0
        # here we set it less than or equal so we add an extra weight for the bias node
        while count <= input_number:
            weight = random.uniform(-0.9, 0.9)
            # print("weight {}: {}".format(count, weight))
            self.weights.append(weight)
            count += 1


