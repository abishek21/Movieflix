from flask import Flask, render_template, url_for, request, redirect
import pandas as pd
import numpy as np
import requests
from urllib.request import urlopen as uReq

from flask_cors import cross_origin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
#from plot import recommendations
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

#df=pd.read_csv("data.csv",error_bad_lines=False,dtype='unicode')
#df['overview']=df['overview'].astype(str)

movies = pd.read_csv('movies_metadata.csv',error_bad_lines=False,dtype='unicode',nrows=30000)
df=movies[['overview','title']]

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    ##text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
    return text


df['overview'] = df['overview'].astype(str)
df['overview'] = df['overview'].apply(clean_text)
df['overview']=df['overview'].str.lower()
df['title']=df['title'].str.lower()
print("data cleaned")
df.set_index('title', inplace=True)
#df_sample = df.iloc[0:30000,]



indices = pd.Series(df.index)
count = CountVectorizer()
count_matrix = count.fit_transform(df['overview'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

print("similarity matrix built")
app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def homepage():
    return render_template("index.html")


@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        title = request.form['content']
        title=str(title)
        title=title.lower()
        #recommended_movies = recommendations(title, cosine_sim, indices, df)
        recommended_movies = []

        # gettin the index of the movie that matches the title
        idx = indices[indices == title].index[0]

        # creating a Series with the similarity scores in descending order
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

        # getting the indexes of the 20 most similar movies
        top_5_indexes = list(score_series.iloc[1:6].index)

        # populating the list with the titles of the best 20 matching movies
        for i in top_5_indexes:
            recommended_movies.append(list(df.index)[i])
        return render_template('results.html', recommended_movies=recommended_movies)
    else:
        return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=False)