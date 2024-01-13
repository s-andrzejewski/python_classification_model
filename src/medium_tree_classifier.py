from sklearn.tree import DecisionTreeClassifier

def train_medium_tree_classifier(training_data):
    predictor_names = training_data.columns[:-1]
    predictors = training_data[predictor_names]
    response = training_data['X35']

    # Train a classifier
    classification_tree = DecisionTreeClassifier(
        criterion='gini',
        max_depth=20,
        random_state=42
    )
    classification_tree.fit(predictors, response)

    trained_classifier = {
        'predictor_names': predictor_names,
        'classifier': classification_tree,
    }

    return trained_classifier

def validate_medium_tree_classifier(trained_classifier, validation_data):
    # Extract predictors and response for validation
    predictor_names = trained_classifier['predictor_names']
    predictors = validation_data[predictor_names]
    response = validation_data['X35']

    # Compute validation predictions
    validation_predictions = trained_classifier['classifier'].predict(predictors)

    # Compute validation accuracy
    validation_accuracy = (validation_predictions == response).mean()

    return validation_accuracy
