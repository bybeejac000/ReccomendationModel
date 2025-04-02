#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
articles = pd.read_csv('shared_articles.csv')
users = pd.read_csv('users_interactions.csv')


# In[2]:


users['eventType'].unique()
rating_map = {"VIEW": 1, "LIKE": 2, "FOLLOW": 3, "BOOKMARK": 4, "COMMENT CREATED": 5}
users["rating"] = users["eventType"].map(rating_map)


# In[3]:


merged = pd.merge(right=articles, left=users, right_on='contentId', left_on='contentId', how='left')


# In[4]:


triple = merged[['personId', 'contentId', 'rating']]


# In[5]:


from scipy.sparse import coo_matrix

triple['mappingPerson'] = triple['personId'].astype('category').cat.codes
triple['mappingcontent'] = triple['contentId'].astype('category').cat.codes

sparse_matrix = coo_matrix((triple['rating'], (triple['mappingPerson'], triple['mappingcontent'])))


# In[6]:


personMapping = {row.mappingPerson:row.personId for row in triple.itertuples()}
contentMapping = {row.mappingcontent:row.contentId for row in triple.itertuples()}
triple.drop(columns=['personId', 'contentId'], inplace=True)


# In[7]:


from sklearn.neighbors import NearestNeighbors

knn_model = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='auto')
knn_model.fit(sparse_matrix)


# In[14]:


distance, indices = knn_model.kneighbors(sparse_matrix.getrow(55), n_neighbors=5)


# In[15]:


reccs = [contentMapping[content] for content in indices[0]]


# In[16]:


for rec in reccs:
    print(articles[articles['contentId'] == rec]['title'])

