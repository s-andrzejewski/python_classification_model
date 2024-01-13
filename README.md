# python_classification_model
Excercise for the "Artifical Inteligence Tools" course during my studies.
Code translated from MatLab, needs fixes and modifications.

**Excercise aims:**
1. Classification of datasets using various methods: e.g. KNN, SVM, decision trees (decision trees), ensembles (ensembles), linear discrimination and others.
2. Calculation of goodness of fit using various values: Sensitivity, Specificity, Precision, False Negative Rate, False Positive Rate, Accuracy, F1 score, Matthews correlation coefficient and learning time.
3. Graphical representation of the data and the results obtained (feature scatter plot, confusion matrix, ROC curve). 
4. Prevention of overlearning. The use of cross-validation and the study of the effect of the kFold parameter on the effectiveness of classification.
5. Study of the effect of individual parameters (meta-parameters) of the methods used on the value of classification results.
6. Analysis of the obtained results.

**Detailed description of exercise tasks:**

1. **Data Preparation:**
   - Acquire a dataset (e.g., "fisheriris.csv," "CreditRatingHistorical," "ionosphere," "census1994," or any other dataset from a database or other sources).
   - Prepare the dataset for classification by splitting it into training and testing sets.
   - Ensure the dataset has an adequate number of samples and a minimum of 3 different features.

2. **Data Visualization:**
   - Present the data graphically using a time plot (e.g., `plot(time, feature)`) and a scatter plot (e.g., `scatterplot(feature1, feature2)`).

3. **Classification:**
   - Perform the exercise on the selected dataset.
   - Implement classification using various methods (e.g., Tree, Ensemble, KNN, SVM, NN, GPR).

4. **Quality Metrics and Timing:**
   - Calculate and present the classification quality in tabular and graphical form.
     - Metrics: Sensitivity, Specificity, Precision, False Negative Rate, False Positive Rate, Accuracy, F1 score, Matthews correlation coefficient.
     - Include the duration of the classification process (`tic`, `toc`).
   - Generate ROC curves and confusion matrices. Provide interpretations.

5. **Cross-Validation Parameter Analysis:**
   - Modify relevant lines of code to analyze the impact of different `kFold` cross-validation values on the computed classification quality and duration for various classification methods.
   - Present the results in tabular or graphical form and provide interpretations.

6. **Individual Parameters Analysis:**
   - Adjust the code to analyze the impact of different values of individual parameters in the applied methods on the computed classification quality and training duration.
   - Present the results in tabular or graphical form (e.g., `plot(parameterQuality)`, `roc`) and provide interpretations.
