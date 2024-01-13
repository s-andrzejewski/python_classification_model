import numpy as np

"""
Calculate various classification metrics based on a confusion matrix.

Parameters:
- c_matrix (numpy.ndarray): Confusion matrix representing the model's performance.

Returns:
- dict: A dictionary containing metrics such as Accuracy, Error, Sensitivity, Specificity, Precision, F1 Score,
        Matthews Correlation Coefficient, Kappa, and details such as True Positives (TP), True Negatives (TN),
        False Positives (FP), and False Negatives (FN).

Example Output:
{
    'Accuracy': 0.85,
    'Error': 0.15,
    'Sensitivity': 0.90,
    'Specificity': 0.80,
    'Precision': 0.88,
    'FalsePositiveRate': 0.20,
    'F1_score': 0.89,
    'MatthewsCorrelationCoefficient': 0.75,
    'Kappa': 0.70,
    'TP': array([50, 75]),
    'TN': array([75, 50]),
    'FP': array([10, 25]),
    'FN': array([15,  5])
}
"""

def calculate_params_mod(c_matrix):
    # Get the dimensions of the confusion matrix
    row, col = c_matrix.shape
    
    # Check if the matrix is square
    if row != col:
        raise ValueError('Confusion matrix dimension is wrong')

    # Determine the number of classes
    n_class = row

    # Initialize a dictionary to store the result metrics
    Result = {}

    if n_class == 2:
        # For binary classification
        TP = c_matrix[0, 0] #* True Positives
        FN = c_matrix[0, 1] #* False Negatives
        FP = c_matrix[1, 0] #* False Positives
        TN = c_matrix[1, 1] #* True Negatives

        # Calculate accuracy and error
        accuracy = (TP + TN) / (TP + FN + FP + TN)
        Error = 1 - accuracy

        # Store accuracy and error in the result dictionary
        Result['Accuracy'] = accuracy
        Result['Error'] = Error

    else:
        # For multi-class classification
        TP = np.zeros(n_class)
        FN = np.zeros(n_class)
        FP = np.zeros(n_class)
        TN = np.zeros(n_class)

        # Calculate True Positives, False Negatives, False Positives, and True Negatives for each class
        for i in range(n_class):
            TP[i] = c_matrix[i, i]
            FN[i] = np.sum(c_matrix[i, :]) - c_matrix[i, i]
            FP[i] = np.sum(c_matrix[:, i]) - c_matrix[i, i]
            TN[i] = np.sum(c_matrix) - TP[i] - FP[i] - FN[i]

        # Calculate accuracy and error for multi-class
        accuracy = TP / (TP + FN + FP + TN)
        Error = FP / (TP + FN + FP + TN)

        # Store accuracy and error in the result dictionary
        Result['Accuracy'] = np.sum(accuracy)
        Result['Error'] = np.sum(Error)

    # Calculate True Positives, False Negatives, False Positives, and True Negatives for overall classes
    P = TP + FN
    N = FP + TN

    # Calculate Sensitivity, Specificity, Precision, False Positive Rate, and F1 Score
    Sensitivity = TP / P
    Specificity = TN / N
    Precision = TP / (TP + FP)
    FPR = 1 - Specificity

    beta = 1
    F1_score = ((1 + beta ** 2) * (Sensitivity * Precision)) / ((beta ** 2) * (Precision + Sensitivity))

    # Calculate Matthews Correlation Coefficient
    MCC = np.array([(TP * TN - FP * FN) / np.sqrt((TP + FP) * P * N * (TN + FN)),
                    (FP * FN - TP * TN) / np.sqrt((TP + FP) * P * N * (TN + FN))])

    MCC = np.max(MCC)

    # Calculate overall accuracy and error
    pox = np.sum(accuracy)
    Px = np.sum(P)
    TPx = np.sum(TP)
    FPx = np.sum(FP)
    TNx = np.sum(TN)
    FNx = np.sum(FN)
    Nx = np.sum(N)

    pex = ((Px * (TPx + FPx)) + (Nx * (FNx + TNx))) / ((TPx + TNx + FPx + FNx) ** 2)

    kappa_overall = np.max([(pox - pex) / (1 - pex), (pex - pox) / (1 - pox)])

    # Store the calculated metrics in the result dictionary
    Result['Sensitivity'] = np.mean(Sensitivity)
    Result['Specificity'] = np.mean(Specificity)
    Result['Precision'] = np.mean(Precision)
    Result['FalsePositiveRate'] = np.mean(FPR)
    Result['F1_score'] = np.mean(F1_score)
    Result['MatthewsCorrelationCoefficient'] = np.mean(MCC)
    Result['Kappa'] = kappa_overall
    Result['TP'] = TP
    Result['TN'] = TN
    Result['FP'] = FP
    Result['FN'] = FN

    # Return the result dictionary
    return Result




# Example Binary Confusion Matrix
confusion_matrix_binary = np.array([[30, 5], [10, 55]])

result_metrics = calculate_params_mod(confusion_matrix_binary)
print(result_metrics)
