from sklearn.tree import DecisionTreeClassifier

def train_coarse_tree_classifier(training_data):
    # Extract predictors and response
    input_table = training_data
    predictor_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9',
                       'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19',
                       'X20', 'X21', 'X22', 'X23', 'X24', 'X25', 'X26', 'X27', 'X28', 'X29',
                       'X30', 'X31', 'X32', 'X33', 'X34']
    predictors = input_table[:, predictor_names]
    response = input_table.X35
    #! is_categorical_predictor = [False] * len(predictor_names)

    # Train a classifier
    classification_tree = DecisionTreeClassifier(
        criterion='gini',
        max_depth=4,
        min_samples_split=2,
        min_samples_leaf=1
    )

    classification_tree.fit(predictors, response)

    # Return the trained classifier
    return classification_tree



def validate_coarse_tree_classifier(trained_classifier, validation_data):
    # Extract predictors and response
    input_table = validation_data
    predictor_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9',
                       'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19',
                       'X20', 'X21', 'X22', 'X23', 'X24', 'X25', 'X26', 'X27', 'X28', 'X29',
                       'X30', 'X31', 'X32', 'X33', 'X34']
    predictors = input_table[:, predictor_names]
    response = input_table.X35

    # Predict using the trained classifier
    predictions = trained_classifier.predict(predictors)

    # Calculate validation accuracy
    correct_predictions = (predictions == response).sum()
    total_samples = len(response)
    accuracy = correct_predictions / total_samples

    # Return the validation accuracy
    return accuracy
