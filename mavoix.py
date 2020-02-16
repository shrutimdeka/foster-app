#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer  #for other skills
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


# In[2]:


####### load data #########################
all_data = pd.read_csv("C:/Users/Shruti/Downloads/foster-app-master/foster-app-master/sample-data/mavoix_ml_sample_dataset.csv", encoding = 'cp437' )


# In[3]:


#replace empty skills box
def replace_empty_skills(df):    #pass 'Other skills' col
    ret_df= df.fillna('no other skill')
    return ret_df


# In[4]:


### Performances N/A to 0
def performance_fillna(df):   #pass dataframe, return dataframe
    df['Performance_PG'] = df['Performance_PG'].fillna(0)
    df['Performance_UG']= df['Performance_UG'].fillna(0)
    df['Performance_12'] = df['Performance_12'].fillna(0)
    df['Performance_10'] = df['Performance_10'].fillna(0)
    return df


# In[5]:


### Performances pg - standardize values ###

def pg_ug_standardize(df): #send only one column-pg/ug performance
    scores_pg =[]
    for x in df:
        if isinstance(x, str):
            scores_pg.append(x.split("/"))
        else:
            scores_pg.append([0, 0])

    percent_pg =[]
    for i in range(0, len(scores_pg)):
        if scores_pg[i][1] == '10':
            calc = float(scores_pg[i][0]) * 9.5 
            percent_pg.append(calc)
        elif scores_pg[i][1] == '4':
            calc = (float(scores_pg[i][0])/4 ) * 100 
            percent_pg.append(calc)
        elif scores_pg[i][1] == '7':
            calc = (float(scores_pg[i][0])/7 ) * 100 
            percent_pg.append(calc)
        elif scores_pg[i][1] == '100':
            percent_pg.append(scores_pg[i][0])
        else:
            percent_pg.append(int(0))
          
    df= [x for x in percent_pg]
    return df #list


# In[6]:


### Performances 12 - standardize values ###
def standardize_10_12(df):    #send only one column- 10/12 performance
    scores_12 =[]
    for x in df:
        if isinstance(x, str):
            scores_12.append(x.split("/"))
        else:
            scores_12.append([0, 0])

    percent_12 =[]
    for i in range(0, len(scores_12)):
        if float(scores_12[i][0]) > 10 :  #is in percentage
            percent_12.append(scores_12[i][0])
        elif float(scores_12[i][0]) <= 4 :  #4 point cgpa
            calc = (float(scores_12[i][0])/4 ) * 100 
            percent_12.append(calc)
        elif float(scores_12[i][0]) == 0:
            percent_12.append(int(0))
        else:                               #10 point cgpa
            calc = float(scores_12[i][0]) * 9.5 
            percent_12.append(calc)
    
    df= [x for x in percent_12]
    return df #list


# In[7]:


######## Create dummy variables ############
def dummy_stream_degree(df):    #pass stream/degree column
    dumm = pd.get_dummies(df)
    return dumm

#all_data = all_data.merge(dumm, left_index=True, right_index=True)


# In[8]:


def clean_other_skills(col, skills):    #pass 'other skills' col
    new_col = []
    for i in range(0, len(col)):
        token = col[i].split(', ')
        
        w = ', '.join([x for x in token if x.lower() in skills])
        if len(w) == 0:
            w = 'noskill'
        col[i] = w
    return col

#n = clean_other_skills(all_data['Other skills'], skills)
#n


# In[9]:


#tfidf for skill
def create_countvector(df):       #pass train 'other skills' col
    vect = CountVectorizer(ngram_range = (1,1))
    vect.fit(df)
    #sk = vect.get_feature_names() #new features
    return vect

def vector_transform(vect, df):           #pass train/test 'Other skills' col
    skill_matrix = vect.transform(df)
    return skill_matrix


# In[10]:


# combine tfidf skills with main data
def combine_skill_data(new_data, skill_matrix, vect):    
    tfidf_data = pd.DataFrame(skill_matrix.toarray(), columns = vect.get_feature_names())
    for col in tfidf_data.columns:
        new_data[col] = tfidf_data[col].values
    return new_data


# In[11]:


####################################################

all_data['Other skills'] = replace_empty_skills(all_data['Other skills'])


# In[12]:


############### call functions-  #######################

all_data['Other skills'] = replace_empty_skills(all_data['Other skills'])
all_data = performance_fillna(all_data)

#standardize marks
df = pg_ug_standardize(all_data['Performance_PG'])
all_data['Performance_PG']= df


df = pg_ug_standardize(all_data['Performance_UG'])
all_data['Performance_UG']= df

df = standardize_10_12(all_data['Performance_12'])
all_data['Performance_12']= df

df = standardize_10_12(all_data['Performance_10'])
all_data['Performance_10']= df


#create dummy variables
#stream column
dum_str = dummy_stream_degree(all_data['Stream'])
all_data = all_data.merge(dum_str, left_index=True, right_index=True)

#degree column
dum_deg = dummy_stream_degree(all_data['Degree'])
all_data = all_data.merge(dum_deg, left_index=True, right_index=True)

#split data
test = all_data[0: int(all_data.shape[0]/4)]
train = all_data[int(all_data.shape[0]/4): ]
    
#define skills to clean up 'other skills' attribute
desired_features = ['algorithms', 'cloud computing', 'analytics', 'data science', 'data structure', 'flask', 'shiny', 'j2ee', 'mvc', 'android', 'angularjs', 'aws', 'azure', 'api', 'asp net', 'artificial intelligence', 'power bi', 'big data', 'blockchain', 'django', 'hadoop', 'image processing', 'java', 'natural language', 'linux', 'machine learning', 'statistical modeling', 'neural networks', 'nlp', 'search engine', 'recommendation system', 'eclipse', 'jquery', 'jsp', 'kotlin', 'matlab', 'tableau', 'regression', 'classification', 'postgresql', 'sqlite', 'rest api', 'statistics', 'raspberry pi', 'scripting', 'servlets', 'xml', 'spss']
skills = [f for f in set(desired_features)]

#cleanup other skills
n= clean_other_skills(all_data['Other skills'], skills)
all_data['Other skills'] = n

#no split
vect = create_countvector(all_data['Other skills'])
countmatrix = vector_transform(vect, all_data['Other skills'] )  #no splitting

dropped_df = all_data.drop(['Current City','Unnamed: 10', 'Other skills', 'Stream', 'Degree' ], axis=1)
df_full = combine_skill_data(dropped_df, countmatrix, vect)

###############################################################################################


# In[ ]:





# In[13]:


############## normalize before PCA ##############################
x = df_full.iloc[:, 1:]
x = StandardScaler().fit_transform(x)


# In[14]:


#pca to visualize the labeled data and reduce features
pca = PCA()
pca_val = pd.DataFrame(pca.fit_transform(x)) #transformed features, not include Application ID

pca_feat = pd.DataFrame(pca.components_)  #eigenvectors
X = pd.DataFrame(pca_val.iloc[:, 0:2]) #training features from top 2 pc's - FOR easy visualization


# In[15]:


########### CLustering using PC features ##############
cls = KMeans(n_clusters=2, max_iter=300, )
cls.fit(X)    #not sending application ID
df_full['labels'] = cls.labels_  #join col to record


# In[16]:


#visualize the clusters
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(X[0],X[1],c=df_full['labels'],s=30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(scatter)

fig.show()


# In[17]:


########## find which categories of candidate has most number of 'other skills' ####

candidate_skills = pd.DataFrame(countmatrix.toarray(), columns=vect.get_feature_names())
candidate_skills['labels'] = df_full['labels']
candidate_skills[candidate_skills['labels']== 1 ] #most skill set- not common ones


# In[18]:


#################### Candidate profiles that were ranked highest ###############
df_full[df_full['labels'] == 1]


# In[19]:


##########################################################################################################


# In[20]:


#split data for predictive model- select candidates with most relevant skills-for both web dev & ML
x_train,x_test, y_train, y_test = train_test_split(df_full.iloc[:, 0:-1], df_full['labels'], random_state = 1)


# In[21]:


#Model based on the clusters obtained(supervised learning here)

rd = RandomForestClassifier()         #as lots of non prominent features
rd.fit(x_train.iloc[:, 1:], y_train)   #not pass 'Application_Id' feature here

#prediction on test data
pred = rd.predict(x_test.iloc[:, 1:])

#accuracy
print("Confusion matrix: ")
print(confusion_matrix(pred, y_test))
print("F1 ", f1_score(pred, y_test))  #95%
print("Accuracy ", accuracy_score(pred, y_test))  #96%


# In[22]:


########################################################################################################


# In[25]:


application = pd.read_csv("C:/Users/Shruti/Downloads/foster-app-master/foster-app-master/sample-data/mavoix_ml_sample_dataset.csv", encoding = 'cp437' )

# Choose candidates from cluster label=0 as they have higher PCA values- hence more info in the dataset
#that is- more skills
#Also, each record in yellow clusters shows prominent ML skills than web development

df_full['labels'][df_full['labels'] == 1] = 'not selected'
df_full['labels'][df_full['labels']== 0] = 'selected'

selected_app = df_full[df_full['labels']== 'selected']['Application_ID']

########### THESE ARE THE PREFERRED CANDIDATES' PROFILES according to clustering ####################
print("##########    PREFERRED CANDIDATES PROFILE  ###################")
application[application['Application_ID'].isin(selected_app)]


# In[24]:


### NOTE: for realistic predictions, features will require some kind of weightage according to what the client in looking for
        # in a candidate, or simply provide an annotated set of data for previously selected/not selected candidates
        # e.g; being a BTech is very important, or a particular pass out year, etc

