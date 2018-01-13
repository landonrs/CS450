class HardCodedClassifier:

    def __init__(self):
        pass

    def fit(self, training_data, training_targets):
        return HardCodedModel()


class HardCodedModel:

    def __init__(self):
        pass

    def predict(self, test_data):
        targets = []

        for array in test_data:
            targets.append(0)

        return targets
