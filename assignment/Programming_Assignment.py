
# coding: utf-8

# # Programming Assignment - Jupyter Notebook

# *You are expected to use this notebook to answer the questions in the Programming Assignment Quiz.*
# 
# *Cells that correspond to the different quiz questions are clearly marked. You can write your code there and run it, keeping a note of the output.*
# 
# *Make sure that you run any code already provided in this notebook - this is, in many cases, required for answering the Programming Assignment quiz questions.*
# 
# Good luck!

# In[1]:


import pandas as pd
import sklearn
import numpy as np


# ## Part 1: Programming in Python

# ### 1.1 Lists

# The code in the following cell specifies a list of proteins.

# In[2]:


proteins = ['selectin', 'collagen', 'elastin', 'insulin', 'coronin', 'myosin']


# ### <font color='mediumblue'>Quiz Question 1</font>

# Write your code in the next cell to remove the 'coronin' element.

# In[4]:


proteins.remove('coronin')


# ### 1.2 DataFrames

# Run the code in the next cell to read a (fictitious) dataset about doctors working in a particular hospital.

# In[5]:


doctors = pd.read_csv('./readonly/doctors.csv')


# ### <font color='mediumblue'>Quiz Question 2</font>

# Write your code in the next cell to get the names of the columns in this dataset.

# In[15]:


#print("names of columns:")
doctors.columns


# *Optional*: If you are interested in exploring the doctors dataset further (e.g. view its first few rows), then you can use the following cell. Remember that you can add more cells if you wish to.

# In[17]:


print('shape of data',doctors.shape)
doctors


# ### <font color='mediumblue'>Quiz Question 3</font>

# Write your code in the next cell to get all records of doctors that have 15 or more years of experience.

# In[18]:


doctors[doctors.experience>=15]


# ### <font color='mediumblue'>Quiz Question 4</font>

# Write your code in the next cell to get the maximum rating for all doctors in the dataset.

# In[19]:


doctors['rating'].max()


# ### <font color='mediumblue'>Quiz Question 5</font>

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib


# Write your code in the next cell to get a boxplot for ratings of all doctors that have 15 or more years of experience.

# In[58]:


doctors[doctors.experience>=15]['rating']
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
doctors[doctors.experience>=15]['rating'].plot.box()


# ## Part 2: Machine Learning

# ### 2.1 Familiarising ourselves with the data

# In this programming task, we are going to be using the **Breast Cancer Wisconsin Dataset**. This includes data about different patients and the corresponding diagnosis. In particular, features are computed from a digitised image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. The diagnosis involves characterising the tumour as 'malignant' or 'benign' (labelled 0 and 1, respectively). 
# 
# This dataset is built in scikit-learn, just like the iris dataset that we saw in this weeks' programming task.
# 
# We'll load the dataset and call it *cancer_dataset*.

# In[23]:


from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()


# Note that, similarly to the iris_dataset object that we saw in this week's programming task, the *cancer_dataset* object that is returned by *load_breast_cancer* is a *Bunch* object. By running the next cell, you will see that its structure is very similar to that of the iris_dataset object.

# In[25]:


print("Keys of cancer_dataset: ", cancer_dataset.keys())


# *Optional*: If you are interested in exploring the cancer_dataset object (e.g. its feature names, target names, etc.), then write your code in the following cell and run it. Remember that you can add more cells if you wish to.

# In[47]:


print("feature_names of cancer_dataset: ", cancer_dataset['feature_names'])
print("target_names of cancer_dataset: ", cancer_dataset['target_names'])
print("shape of cancer_dataset: ", cancer_dataset['data'].shape)


# ### <font color='mediumblue'>Quiz Question 6</font>

# Write your code in the next cell to get the shape of the *data* part of the cancer_dataset.

# In[32]:



print("shape of cancer_dataset: ", cancer_dataset['data'].shape)


# ### 2.2 Splitting our dataset into training data and test data

# In[33]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    cancer_dataset['data'], cancer_dataset['target'], random_state=0)


# ### 2.3 K Nearest Neighbours

# We will now learn how to build a classification model for the breast cancer dataset with the use of the k nearest neighbours algorithm.

# #### Building and evaluating the model for 1 nearest neighbour

# Run the code below to create a *KNeighborsClassifier* model called *knn_model*. Note that *n_neighbors=1* is setting the number of nearest neighbours to 1.

# In[34]:


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, y_train)


# In[35]:


print("Test set score: {:.3f}".format(knn_model.score(X_test, y_test)))


# ### <font color='mediumblue'>Quiz Question 7</font>

# Write your code in the next cell(s) to build and evaluate a K Nearest Neighbours model for 5 neighbours.

# In[36]:


knn_model_5 = KNeighborsClassifier(n_neighbors=5)
knn_model_5.fit(X_train, y_train)


# In[37]:


print("Test set score: {:.3f}".format(knn_model_5.score(X_test, y_test)))


# #### Using the model to make predictions

# The following code specifies a previously unseen patient case.

# In[38]:


X_new = np.array([[
  1.239e+01, 1.538e+01, 1.328e+02, 1.382e+03, 1.007e-01, 2.661e-01, 3.791e-01,
  1.001e-01, 2.009e-01, 6.371e-02, 6.895e-01, 8.943e-01, 4.259e+00, 9.594e+01,
  5.789e-03, 3.864e-02, 3.233e-02, 1.187e-02, 3.003e-02, 5.923e-03, 2.242e+01,
  1.689e+01, 1.926e+02, 2.721e+03, 1.782e-01, 5.461e-01, 6.579e-01, 1.958e-01,
  4.811e-01, 1.008e-01]])


# ### <font color='mediumblue'>Quiz Question 8</font>

# Write your code in the next cell to use your K Nearest Neighbours model for 5 neighbours to make a prediction for this new patient case.

# In[59]:


prediction=knn_model_5.predict(X_new)
print(prediction)
cancer_dataset['target_names'][prediction]


# ### 2.4 Decision Tree

# Use the training and test data specified in Section 2.2 to create a Decision Tree with maximal depth 5.
# 
# *Important note*: You should set the *random_state* parameter of the DecisionTreeClassifier to 20.

# In[63]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=5,random_state=20)
tree.fit(X_train, y_train)


# ### <font color='mediumblue'>Quiz Question 9</font>

# Evaluate the accuracy of the decision tree that you just built.

# In[64]:


print("Test set score: {:.3f}".format(tree.score(X_test, y_test)))


# ### <font color='mediumblue'>Quiz Question 10</font>

# Write your code in the next cell(s) to use your decision tree model to make a prediction for the new patient case specified earlier in this notebook.

# In[52]:


prediction=tree.predict(X_new)


# In[53]:


cancer_dataset['target_names'][prediction]

