import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import warnings

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.externals import joblib

warnings.filterwarnings('ignore')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# load data from database
def load_data(database_filepath):
    '''
    Load data from database.
    Args:
        database_filepath(str): Filepath of database.
    Returns:
        X(ndarray): Messages.
        Y(ndarray): Categories.
        category_names(list): Names of categories
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    t_name = database_filepath.split('/')[-1].split('.')[0]
    df = pd.read_sql_table(table_name=t_name,con=engine)
    category_names = list(df.columns[4:])
    X = df['message'].values
    Y = df.drop(['id','message','original','genre'],axis=1).values

    return (X,Y,category_names)

def tokenize(text):
    '''
    Tokenize the text.
    Args:
        text(str): File for tokenize.
    Returns:
        tokens(list): Tokens.
    '''    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():

    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier(random_state=7)))
        ])

    parameters = {'tfidf__use_idf': (True, False),
    'clf__estimator__min_samples_leaf':[1,3,5],
    'clf__estimator__n_estimators':[100,200]}

    model = GridSearchCV(pipeline,param_grid=parameters,cv=5)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    # bad_categories = []

    for x in range(len(category_names)):
        accuracy =  accuracy_score(Y_test[:, x], Y_pred[:, x])
        precision = precision_score(Y_test[:, x], Y_pred[:, x],average='weighted')
        recall = recall_score(Y_test[:,x], Y_pred[:, x],average='weighted')
        f1 = f1_score(Y_test[:, x], Y_pred[:, x],average='weighted')
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        # if accuracy < 0.90:
        #     bad_categories.append(x)
        print(category_names[x])
        print("\tAccuracy: {:.3f}\t\t Precision: {:.3f}\t\t Recall: {:.3f}\t\t F1_score: {:.3f}".\
            format(accuracy,precision,recall,f1))
    print('Overall mean')
    print("\tAccuracy: {:.3f}\t\t Precision: {:.3f}\t\t Recall: {:.3f}\t\t F1_score: {:.3f}".\
        format(np.mean(accuracy_list),np.mean(precision_list),np.mean(recall_list),np.mean(f1_list)))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


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