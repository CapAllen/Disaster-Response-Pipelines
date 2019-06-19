import re
import json
import random
import pandas as pd

from random import randrange

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask,render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


from pyecharts import options as opts
from pyecharts.charts import Bar

from jinja2 import Markup, Environment, FileSystemLoader
from pyecharts.globals import CurrentConfig

CurrentConfig.GLOBAL_ENV = Environment(loader=FileSystemLoader("./templates"))

app = Flask(__name__)

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

def bar_base(series,title):
    x = list(series.index)
    y = list(series)
    c = (
        Bar(init_opts=opts.InitOpts(width='1500px'))
        .add_xaxis(x)
        .add_yaxis("", y,color='#337ab7')
        .set_global_opts(title_opts=opts.TitleOpts(title=title))
    )
    return c
    
# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")

#extract data needed for visuals
genre_counts = df.groupby('genre').count()['message']
category_counts = df.iloc[:,4:].sum().sort_values(ascending=False)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')

@app.route("/message")
def message_bar():
    c = bar_base(genre_counts,'Distribution of Message Genres')
    return c.dump_options()

@app.route("/topten")
def top_bar():
    c = bar_base(category_counts[:10],'Top 10 Categories')
    return c.dump_options()

@app.route("/bottomten")
def bottom_bar():
    c = bar_base(category_counts.sort_values()[:10],'Bottom 10 Categories')
    return c.dump_options()

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()