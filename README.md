# AliAnt268

Objective:
This document outlines the process of deploying a text-based regression model using Flask for predicting labels of entered text. The model is built using TfidfVectorizer for text vectorization and a simple regression algorithm.

Steps:

Data Preprocessing:

Convert every text column in the dataset to a string.
Remove columns with null values.
Split the data into 80% training and 20% test sets.
Text Vectorization:

Utilize TfidfVectorizer to convert text data into numerical format suitable for machine learning.
Model Building:

Train a simple regression model using the training dataset.
Flask Application Setup:

Implement a Flask application for creating a RESTful API.
Define routes and functions to handle text inputs.
Requirements.txt:

Maintain a requirements.txt file containing all the necessary libraries and dependencies for the project.

make_request.py:

Create a script (make_request.py) to facilitate making predictions using the trained model.
Accept text input and call the prediction function from the deployed model.
