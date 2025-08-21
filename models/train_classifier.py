# Import general libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
from pathlib import Path

# Importing libraries for natural language processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# importing libraries for machine learning
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    Loading database into dataframe
    Args:
        database_filepath (str): Filepath to database containing the messages and classifications

    Results:
        X: dataframe containing the features
        Y: dataframe containing the labels
        category_names: List of category names
    """
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    table_name = Path(database_filepath).stem + '_table'
    df = pd.read_sql_table(table_name, engine)
    # X: Our input to be classified
    X = df['message']
    # Y: The goal of our classification
    Y = df.iloc[:,4:] # Y are all the other columns except the first 4

    category_names = Y.columns # To be used for visualizations
    return X, Y, category_names


def tokenize(text):
    """
    Preparing (tokenizing) the messages to be used for the ML algorithm

    Parameters
    ----------
        text (str): Message to be tokenized

    Returns
    -------
        clean_tokes: List of tokes extracted from the text
    """
    # Replacing all urls with placeholder
    url_place_holder_string = 'urlplaceholder'
    
    # Regular expression for detecting URLs
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    # Replace each URL in text string with placeholder
    for url in detected_urls:
        #text = re.sub(url, url_place_holder_string, text) causes error because some detected urls contain brackets
        text = text.replace(url, url_place_holder_string)

    # Replacing all punctuation with space making sure no brackets are left
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    
    # Extractring the tokens from the text
    tokens = word_tokenize(text)
    
    # Removing stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    
    # Using lemmatizer to remove inflection, endings, ...
    # Initiating lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Iterate through each token, using list comprehension
    clean_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
    
    return clean_tokens

def build_model():
    """
    Builds model

    Parameters
    ----------
        none
    Returns
    -------
        model: results from fitting pipeline

    """

    # Pipeline including RandomForestClassifier for classification
    pipeline = Pipeline([
        ('counts_vectorizer', CountVectorizer(tokenizer=tokenize)), 
        ('tfidf_transformer', TfidfTransformer()), 
        ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Training the model
    model = pipeline
    return model

def build_model_gridsearch():
    """
    Builds model using grid search

    Parameters
    ----------
        none
    Returns
    -------
        pipeline sklearn.model_selection.GridSearchCV incl. sklearn estimater

    """

    # Pipeline including RandomForestClassifier for classification
    pipeline = Pipeline([
        ('counts_vectorizer', CountVectorizer(tokenizer=tokenize)), 
        ('tfidf_transformer', TfidfTransformer()), 
        ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Parameters for Grid Search
    # Main goal is classifying the messages, searching part of the classifier space, showing the principle by choosing two different
    # parameters to optimize, using only two values each because otherwise it took forever
    parameters ={
        'classifier__estimator__n_estimators' : [50, 100],
        'classifier__estimator__min_samples_split' : [2, 3]
    }

    # Creating grid search object, adding verbose to see that it is doing something
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose = 3)

    model = cv
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model and print the classification and accuracy score

    Parameters
        model: The model to be evaluated
        X_test: The messages to be used for evaluation
        Y_test: The classification of the models to be evaluated

    Output
        Prints the classification report and accuracy score
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_pred, Y_test.values, target_names=category_names))

    # Print accuracy score
    print(f"Accuracy score: {np.mean(Y_test.values == Y_pred)}")



def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()