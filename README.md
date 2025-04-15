# Salary Prediction using Artificial Neural Network (ANN) - Learning Journey

This project was undertaken to learn and implement a machine learning model for predicting the estimated salary of bank customers using an Artificial Neural Network (ANN). The goal was to understand the end-to-end process of building a regression model with ANNs, from data preparation to deployment.

## Project Overview

In this project, we aimed to predict the 'EstimatedSalary' of customers based on various features available in a dataset. The approach involved using an Artificial Neural Network (ANN) built with TensorFlow and Keras. The project covered several key stages:

1.  **Data Preprocessing:** Preparing the raw data for the model by handling categorical features and scaling numerical features.
2.  **Model Building:** Designing and constructing the architecture of the ANN.
3.  **Model Training:** Training the ANN on the prepared data to learn the relationship between the features and the target variable.
4.  **Deployment:** The trained model has been deployed as an interactive web application using Streamlit.

## Visit the Deployed Application

You can visit the live deployed application using the following link:

[https://salaryregression-jf49mczvnm7db2faevmbxg.streamlit.app/](https://salaryregression-jf49mczvnm7db2faevmbxg.streamlit.app/)

Feel free to interact with the application by entering different customer features to see the salary predictions.

## What I Learned

Through this project, I gained valuable insights and practical experience in the following areas:

### 1. Data Preprocessing

* Handling Categorical Data: I learned how to convert categorical features like 'Gender' (using Label Encoding) and 'Geography' (using One-Hot Encoding) into numerical representations that can be used by a neural network.
* Feature Scaling: I understood the importance of scaling numerical features (using `StandardScaler`) for neural network training to ensure faster convergence and prevent features with larger values from dominating the learning process.
* Data Splitting: I learned how to split the dataset into training and testing sets to properly evaluate the model's performance on unseen data.

### 2. Model Building with ANN

* Neural Network Architecture: I gained experience in designing a sequential neural network with multiple dense layers, choosing appropriate activation functions (ReLU for hidden layers and linear for the output layer in regression).
* Input Shape: I learned how to define the input layer of the ANN based on the number of features in the dataset.

### 3. Model Training

* Choosing Optimizer and Loss Function: I understood how to select an appropriate optimizer (Adam) and loss function (Mean Absolute Error - MAE) for a regression problem.
* Metrics: I learned how to track relevant metrics (MAE) during the training process to monitor the model's performance.
* Callbacks: I explored the use of callbacks like TensorBoard for visualizing training progress and EarlyStopping to prevent overfitting.

### 4. Deployment with Streamlit

* Creating a User Interface: I learned how to use the Streamlit library to create a simple and interactive web application to get user inputs for the features.
* Loading Saved Models and Preprocessors: I gained experience in saving and loading trained models (`.h5` files) and preprocessing objects (scalers and encoders using `pickle`) for use in a deployment setting.
* End-to-End Prediction: I understood the flow of taking user inputs, preprocessing them in the same way as the training data, using the loaded model to make predictions, and displaying the results.

## Key Takeaways

* Building a regression model with an ANN involves a systematic process of data preparation, model design, training, and evaluation.
* Proper preprocessing of categorical and numerical features is crucial for the performance of neural networks.
* Choosing the right architecture, optimizer, and loss function is important for effective model training.
* Deployment tools like Streamlit make it relatively easy to create interactive applications to showcase machine learning models.
* Hyperparameter tuning is an important next step to further optimize the model's performance.

This project provided a hands-on learning experience in applying artificial neural networks to a regression problem and understanding the key steps involved in the machine learning pipeline.
