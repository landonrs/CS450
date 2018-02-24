import random
import math


class NeuralNetwork:

    def __init__(self, num_layers, nodes_per_layer):
        """num layers = the number of layers in the network including output
           nodes-per-layer = an array of ints each signifying the number of nodes per layer. The length of this array
           must match the num_layers value"""
        self.inputs = []
        self.node_layers = []
        self.layer_output = []
        self.num_layers = num_layers
        self.number_of_nodes_per_layer = nodes_per_layer
        self.predicted_targets = []
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


    def fit(self, training_data, training_targets):
        """Takes in a numpy array of training data and a list of targets
        and creates connections between the inputs and the node layer.
        It then iterates through each row in the training data and determines the
        h and a values for each node to find the output"""
        # first determine the number of attributes
        num_attributes = training_data.shape[1]
        layer_count = 0
        for node_layer in self.node_layers:
            for node in node_layer:
                # for the first layer, the number of weights generated is determined by the number of inputs
                if layer_count == 0:
                    node.create_random_weights(num_attributes)
                # for every other layer the number of weights is determined by the previous layer
                else:
                    node.create_random_weights(self.number_of_nodes_per_layer[layer_count - 1])
            layer_count += 1

        for row in training_data:
            layer_count = 0
            self.layer_output.clear()
            for node_layer in self.node_layers:
                node_instance_ouput = []
                for node in node_layer:
                    if layer_count == 0:
                        node.determine_h_value(row)
                    else:
                        node.determine_h_value(self.layer_output[layer_count - 1])
                    node_instance_ouput.append(node.determine_a_value())
                self.layer_output.append(node_instance_ouput)
                print("output for this row was {}".format(node_instance_ouput))
                layer_count += 1
            self.predicted_targets.append(self.predict_class(self.layer_output[-1]))
        return self.predicted_targets

    def predict_class(self, output):
        highest_a = output[0]
        for a in output:
            if a > highest_a:
                highest_a = a
        return output.index(highest_a)



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
        return self.a

    def create_random_weights(self, input_number):
        count = 0
        # here we set it less than or equal so we add an extra weight for the bias node
        while count <= input_number:
            weight = random.uniform(-0.4, 0.4)
            print("weight {}: {}".format(count, weight))
            self.weights.append(weight)
            count += 1


