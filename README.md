# AliAnt268

Objective:
This document outlines the process of deploying a text-based regression model using Flask for predicting labels of entered text. The model is built using TfidfVectorizer for text vectorization and a simple regression algorithm.

Steps:

1.Data Preprocessing:

Convert every text column in the dataset to a string.
Remove columns with null values.
Split the data into 80% training and 20% test sets.

2.Text Vectorization:

Utilize TfidfVectorizer to convert text data into numerical format suitable for machine learning.

3.Model Building:

Train a simple regression model using the training dataset.

4.Flask Application Setup:

Implement a Flask application for creating a RESTful API.
Define routes and functions to handle text inputs.

6.Requirements.txt:

Maintain a requirements.txt file containing all the necessary libraries and dependencies for the project.

6.make_request.py:

Create a script (make_request.py) to facilitate making predictions using the trained model.
Accept text input and call the prediction function from the deployed model.

7.Flask API Endpoint:

Implement an API endpoint in the Flask application to receive text inputs and return predictions.



The model exactly predicted the label 'ft' for the text Lebensmittel kommssionierung
The solution is achieved through dockerization of RESTAPI



![S1](https://github.com/alishaantony/AliAnt268/assets/36256101/34d33895-2d9b-48e2-8e49-c569ee4b9be1)

