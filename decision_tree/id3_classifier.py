import numpy as np

class ID3Classifier:

    def __init__(self):
        self.tree_node_list = {}
        self.next_node = None

    def calculate_branch_entropy(self, targets):
        """determine the starting entropy for the current data set so the info gain can be later determined"""
        branch_entropy = 0
        total_targets = len(targets)
        class_frequencies = []
        unique_classes = np.unique(targets)
        for c in unique_classes:
            class_frequencies.append(targets.count(c) / total_targets)
        for probability in class_frequencies:
            branch_entropy -= self.calculate_entropy(probability)
        return branch_entropy



    def calculate_entropy(self, probability):
        if probability != 0:
            return probability * np.log2(probability)
        else:
            return 0

    def determine_feature_entropy(self, targets, feature_values, feature_indexes):
        feature_entropy = 0
        for value in feature_values:
            branch_targets = []
            for index in feature_indexes[value]:
                # print("feature {} class: {}".format(value, targets[index]))
                branch_targets.append(targets[index])
            feature_entropy += self.calculate_branch_entropy(branch_targets) * (len(branch_targets) / len(targets))
        # print(feature_entropy)
        return feature_entropy


    def create_tree(self, data, targets, column_names):
        num_rows = len(data)
        num_features = len(column_names)
        most_common_class = max(set(targets), key=targets.count)

        if num_rows == 0 or num_features == 0:
            return most_common_class
        elif targets.count(targets[0]) == num_rows:
            return targets[0]

        else:
            feature_entropies = []
            all_feature_values = []
            for feature in column_names:
                feature_values = []
                feature_indexes = {}
                # this holds the indexes for which data values follow down the branch
                filtered_data = []
                index = 0
                for value in data[feature]:
                    if value not in feature_values:
                        feature_values.append(value)
                        feature_indexes[value] = [index]
                        index += 1
                    else:
                        feature_indexes[value].append(index)
                        index += 1
                # print(feature_indexes)
                all_feature_values.append(feature_values)
                feature_entropies.append(self.determine_feature_entropy(targets, feature_values, feature_indexes))
            best_feature = feature_entropies.index(min(feature_entropies))
            best_feature_name = column_names[best_feature]
            tree_node_list = {best_feature_name: {}}
            new_names = [x for i, x in enumerate(column_names) if i != best_feature]
            # print(new_names)
            # print(tree_node_list)

            # now check each branch of this feature
            for value in all_feature_values[best_feature]:
                new_data = data.loc[data[best_feature_name] == value]
                # print(new_data)
                subtree = self.create_tree(new_data, new_data['Loan'].values.tolist(), new_names)

                tree_node_list[best_feature_name][value] = subtree
            return tree_node_list


    def fit(self, dataframe, targets):
        tree_model = TreeModel()
        tree_model.tree_node_list = self.create_tree(dataframe, targets, list(dataframe)[0:-1])
        return tree_model

class TreeModel:

    def __init__(self):
        self.tree_node_list = None




