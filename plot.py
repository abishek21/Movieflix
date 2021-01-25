import nltk
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import re
import pickle

#nltk.download('stopwords')



movies = pd.read_csv('movies_metadata.csv',error_bad_lines=False,dtype='unicode')
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
df.set_index('title', inplace=True)
df_sample = df.iloc[0:30000,]
df_sample.to_csv("data.csv")

count = CountVectorizer()
count_matrix = count.fit_transform(df_sample['overview'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)
# print(type(cosine_sim))
# pickle.dump(cosine_sim, open("cosine_sim.pkl", "wb"), protocol=4)
# np.save('cosine_sim.npy', cosine_sim)
#cosine_sim = np.load('cosine_sim.npy')
indices = pd.Series(df_sample.index)
#np.save('indices.npy',indices)


def recommendations(title, cosine_sim,indices,df):
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

    return recommended_movies,idx


print(recommendations('Toy Story 2',cosine_sim,indices,df))
