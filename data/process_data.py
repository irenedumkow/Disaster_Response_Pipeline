"""
Preprossing of message date, takes a csv file of messages, adds categories to it and saves it in a database
"""
# Importing of necessary libraries
import sys
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Loads the message and categories datasets and merges them to one dataset
    Arg:
        messages_filepath (str): The filepath to the messages dataset
        categories_filepath (str): The filepath to the catagories dataset, the messages have already been labeled to
                                    categories

    Returns
        df: Dataframe combining messages and categories dataset
    """
    
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)

    # Merging datasets
    df = messages.merge(categories, how='left', on=['id'])
    return df


def clean_data(df):
    """
    Cleaning the categories column for use in machine learning algorithms

    Arg:
        df: Dataframe containing messages and categories
    
    Returns
        df: Dataframe with the categories cleaned up
    """
    # Creating a separate categories dataframe, splitting the categories columns at the ;
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = row.str[:-2]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Keeping only the 0 and 1 in the categorie columns
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Child alone column contains only 0, which can influence the ML algorithms
    # Dropping the child alone column
    categories.drop(['child_alone'], axis=1, inplace=True)

    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], sort=False, axis=1)

    # Related columns contains values of 2, assuming input errors, keeping only rows with related != 2
    df = df[df['related'] != 2]

    # Removing duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """
    Saving data to database
    Args:
        df: Dataframe
        database_filename (str): Filepath for database

    Returns:
        none

    """
    engine = create_engine('sqlite:///'+ database_filename)
    table_name = database_filename.replace(".db", '_table')
    print(f"Tabelname: {table_name}")
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()