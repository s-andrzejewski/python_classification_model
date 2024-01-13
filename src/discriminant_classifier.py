from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def train_discriminant_classifier(training_data):
    # Extract predictors and response
    predictor_names = Xdataset.columns[:-1]
    predictors = training_data[predictor_names]
    response = training_data['X35']

    # Train a classifier
    classification_discriminant = LinearDiscriminantAnalysis()
    classification_discriminant.fit(predictors, response)

    # Create the result struct
    trained_classifier = {
        'predictor_names': predictor_names,
        'classifier': classification_discriminant,
    }

    return trained_classifier


def validate_discriminant_classifier(trained_classifier, validation_data):
    # Extract predictors and response for validation
    predictor_names = trained_classifier['predictor_names']
    predictors = validation_data[predictor_names]
    response = validation_data['X35']

    # Compute validation predictions
    validation_predictions = trained_classifier['classifier'].predict(predictors)

    # Compute validation accuracy
    validation_accuracy = (validation_predictions == response).mean()

    return validation_accuracy
