# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# %%
#reading csv files
movies = pd.read_csv("movies.csv")
tags = pd.read_csv("tags.csv")


# %%
movies.head()


# %%
movies_tags = pd.merge(movies, tags)
movies_tags.head()
movies_tags.shape


# %%
#combine tags based on movie
movies_tags_combined = movies_tags.groupby(["movieId", "title", "genres"])["tag"].apply(" ".join).reset_index()
movies_tags_combined.head()


# %%
def combine_genres_tags(row):
    genres = row["genres"].replace("|", " ")
    return genres + " " + row["tag"]
movies_tags_combined["features"] = movies_tags_combined.apply(combine_genres_tags, axis=1)
movies_tags_combined.head()


# %%
#remove year from title to allow searching
movies_tags_combined["title"] = movies_tags_combined["title"].str[:-7]
movies_tags_combined.head(10)


# %%
vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(movies_tags_combined["features"])


# %%
def index_to_title(index):
    return movies_tags_combined[movies_tags_combined.index == index]["title"].values[0]
def title_to_index(title):
    return movies_tags_combined[movies_tags_combined.title == title].index.values[0]


# %%
similarity = cosine_similarity(matrix)
movie = input("Enter a movie: ")
movieindex = title_to_index(movie)
similar_movies = list(enumerate(similarity[movieindex]))
sorted_list = sorted(similar_movies, key=lambda m: m[1], reverse=True)[1:]


# %%
for i in range(5):
    print(index_to_title(sorted_list[i][0]))

