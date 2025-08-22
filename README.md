# Disaster Response Pipeline Project

## Project Description

This project if part of an Udacity Nanodegree, quoting from the project overview in the course:

"In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events. You will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

Your project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data."

Therefore the project is divided in the following sections:

1. Processing the data by building an ETL pipeline the extract the data from the source (an Excel file containing pre-labeled messages).
2. Building a machine learning pipeline to train a model which classifies a message into different categories (a message can belong to several categories).
3. Building a web app which can classify a new message in real time and also shows some information about the training data.

## Dependencies
- Python 3.6, with higher version the dependencies could not be resolved for the packages
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Processing Libraries: NLTK
- SQL Database Libraries: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly, Joblib, Colorama

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Additional Material
In the **data** and **model** folder are th Jupyther notebooks which show more clearly the process of getting to the Python code.

1. **ETL Preparation Notebook**: Explains how th ETL pipeline is build up
2. **ML Pipeline Preparation Notebook**: Contain additional optimization step which are not in include in the Python code like a Grid Search