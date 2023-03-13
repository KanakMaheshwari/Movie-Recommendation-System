#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing dependencies
import pandas as pd  #analyse data nd manipulation of data
import numpy as np #arrays
import difflib #if in case movie name is misspelled by user, it will choose the closest matching movie from dataset.
from sklearn.feature_extraction.text import TfidfVectorizer   #to convert textual data into numnerical values
from sklearn.metrics.pairwise import cosine_similarity #to know the similarity score for movie matching


# In[2]:


#loading the data from csv file to a pandas dataframe
movies_data=pd.read_csv('movies.csv')
movies_data.head()
movies_data.columns = movies_data.columns.str.strip()


# In[3]:


# printing the first 5 rows of the dataframe
movies_data.head() #this head prints the first five frames of the dataframe


# In[4]:


#number of rows and columns in the data frame
movies_data.shape


# In[5]:


#selecting the relevant features for reommendation
selected_features=['genres','keywords','tagline','cast', 'director']
print(selected_features)


# In[6]:


#textual data contains null value so here we are converting these values into null string
for feature in selected_features:
  movies_data[feature]=movies_data[feature].fillna('')


# In[7]:


#combining all the 5 selected features
combined_features=movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']


# In[8]:


print(combined_features)


# In[9]:


#converting the text data to feature vectors
vectorizer=TfidfVectorizer() #vectorizer is a variable


# In[10]:


feature_vectors=vectorizer.fit_transform(combined_features) #to store numerical values we are using feature_vectors and vectorizer.fit_tranform is used to covert text to numerical


# In[11]:


print(feature_vectors)


# In[12]:


#cosine similarity
#getting the similarity scores using cosine similarity
similarity=cosine_similarity(feature_vectors)


# In[13]:


print(similarity)


# In[14]:


print(similarity.shape)#movie index,similarity score:4803,4803


# In[15]:


#getting the movie name from the user
movie_name=input("Enter your favourite movie name:")


# In[16]:


#creating a list with all the movie names given in the dataset
list_of_all_titles=movies_data['title'].tolist()#toist function is used to create list
print(list_of_all_titles)


# In[17]:


#finding the close match for the movie name given by the user
find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)#comparing on getting close match in movie given by the user from the data present in dataset
print(find_close_match)


# In[18]:


close_match=find_close_match[0]


# In[19]:


print(close_match)


# In[20]:


#finding the index of the movie with title
index_of_the_movie=movies_data[movies_data.title==close_match]['index'].values[0]
print(index_of_the_movie)


# In[21]:


#getting a list of similar movies
similarity_score=list(enumerate(similarity[index_of_the_movie])) #enumerate runs in loop (list 0-n range)
print(similarity_score)


# In[22]:


len(similarity_score)#number of movies compared 


# In[23]:


#sorting the movies based on their similarity score
sorted_similar_movies=sorted(similarity_score, key = lambda x:x[1],reverse=True)
print(sorted_similar_movies)


# In[24]:


#print the name of similar movies based on the index
print('Movies suggested for you:\n')
i=1
for movie in sorted_similar_movies:
    index=movie[0]
    title_from_index=movies_data[movies_data.index==index]['title'].values[0]
    if (i<30):
        print(i,'.',title_from_index)
        i+=1
    


# In[25]:


#Movie Recommendation system
#getting the movie name from the user
movie_name=input("Enter your favourite movie name:")
#creating a list with all the movie names given in the dataset
list_of_all_titles=movies_data['title'].tolist()#toist function is used to create list
#finding the close match for the movie name given by the user
find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)#comparing on getting close match in movie given by the user from the data present in dataset
close_match=find_close_match[0]
#finding the index of the movie with title
index_of_the_movie=movies_data[movies_data.title==close_match]['index'].values[0]
#getting a list of similar movies
similarity_score=list(enumerate(similarity[index_of_the_movie])) #enumerate runs in loop (list 0-n range)
#sorting the movies based on their similarity score
sorted_similar_movies=sorted(similarity_score, key = lambda x:x[1],reverse=True)
#print the name of similar movies based on the index
print('Movies suggested for you:\n')
i=1
for movie in sorted_similar_movies:
    index=movie[0]
    title_from_index=movies_data[movies_data.index==index]['title'].values[0]
    if (i<30):
        print(i,'.',title_from_index)
        i+=1
    


# In[ ]:




