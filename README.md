# deep_learning_challenge
Module 21 Challenge

Report on Deep Learning Model for Alphabet Soup

Overview of the Analysis

The purpose of this analysis is to develop a deep learning model to predict the success of charitable donations based on a set of features provided by the organization "Alphabet Soup." This model will help the organization identify which funding applications are likely to be successful, thereby optimizing resource allocation. The goal is to design, train, and evaluate a deep neural network to achieve an accuracy of 75% or higher on the test dataset.

## Results

* Data Preprocessing
    Target Variable(s):
    * The target variable for this analysis is IS_SUCCESSFUL, which indicates whether a charity donation application was successful (1) or not (0).

    Feature Variable(s):
    * The features include the categorical and numerical variables that describe the donation applications:
        * APPLICATION_TYPE
        * AFFILIATION
        * CLASSIFICATION
        * USE_CASE
        * ORGANIZATION
        * STATUS
        * INCOME_AMT
        * SPECIAL_CONSIDERATIONS
        * ASK_AMT

    Removed Variables:
    * The columns EIN and NAME were removed from the dataset as they are identification numbers that do not contribute meaningfully to the prediction of success or failure of an application.


* Compiling, Training, and Evaluating the Model

    Neurons, Layers, and Activation Functions:
    * Neurons:
        The model started with 3 hidden layers. Across multiple optimization attempts, we adjusted the number of neurons in each layer. For the final attempt, neurons used:
            * 100 neurons in the first layer
            * 80 neurons in the second layer
            * 50 neurons in the third layer
            * 25 neurons in the fourth layer
    * Activation Functions:
        * The tanh activation function was used in the first layer to allow for both positive and negative outputs, followed by relu activation functions in subsequent layers to introduce non-linearity and handle vanishing gradients.
        * The output layer used a sigmoid activation function, as this is a binary classification task.

    * Target Model Performance:
        * The target performance was 75% accuracy on the test data.
        * The final model achieved an accuracy of 72.78%, which, while close, fell short of the desired target.

    * Steps Taken to Increase Model Performance:
        * Several strategies were attempted to improve performance:
        * Adjusting neurons: The number of neurons was increased in hidden layers to give the model more capacity to learn complex relationships in the data.
        * Adding dropout regularization: Dropout layers with a dropout rate of 20–30% were added to prevent overfitting.
        * Using L2 regularization: This was applied to the first layer to reduce large weight values and help with overfitting.
        * Changing activation functions: We experimented with tanh and relu activation functions to ensure proper non-linear transformations.
        * Adjusting the learning rate: The learning rate was fine-tuned using the Adam optimizer, starting with 0.001 and reducing to 0.0003 for better convergence.
        * Increased batch size: A larger batch size of 64 was used to improve training efficiency.
        * Batch normalization: Added batch normalization to stabilize training and improve performance.
        * Early stopping: Early stopping was employed to prevent overfitting by stopping the training when the validation loss stopped improving.


## Summary

The deep learning model developed for Alphabet Soup achieved an accuracy of 72.78%, which, while close, did not meet the 75% target. Several optimization techniques were applied, including increasing the complexity of the network with more neurons and layers, adding regularization, and fine-tuning hyperparameters such as the learning rate and batch size.

Recommendations for Future Work:
Given that the deep learning model did not achieve the target accuracy, a different approach could be considered. One possible solution is to try ensemble learning methods, such as random forests or gradient boosting (e.g., XGBoost), which are often more effective for structured/tabular data like this.

These models are particularly good at handling imbalanced datasets, noise, and non-linear relationships, which could be present in this dataset.
Ensemble methods combine the predictions of multiple models, often resulting in improved generalization and better overall performance.

Next Steps:
Implement and evaluate random forests and gradient boosting models.
Perform hyperparameter tuning using techniques like grid search or random search to optimize model performance.
Explore feature engineering further to enhance the dataset with additional derived features.
By combining deep learning with ensemble methods and advanced feature engineering, it is possible that the target accuracy of 75% or higher can be achieved.