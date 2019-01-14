
# coding: utf-8

# # Machine Learning Task

# In this programming task, we'll try our hand at supervised learning. We'll work with a simple dataset and we'll build, evaluate and use two different machine learning models.<br><br>
# The dataset that we're going to be using in this programming task is from the botany domain, so it has to do with plants. Note that in this week's Programming Assignment you'll get to work with medical data!

# ## Part 1: Importing _scikit-learn_

# _scikit-learn_ is the most widely used Python library for machine learning. 
# We first need to tell Python that we're going to be using scikit-learn, with the use of the import command.

# In[1]:


import sklearn


# <br>
# ## Part 2: Familiarising ourselves with the data

# ### Loading the data

# scikit-learn comes with a few small standard datasets that do not require to download and to read data from external websites. In this programming task, we are going to be using the **Iris Plants Dataset**. This dataset contains information about different iris flowers, i.e. sepal length, sepal width, petal length, petal width and species (with three possible values for species: setosa, versicolor and virginica). <br><br>
# The iris dataset is typically used for supervised learning tasks, and in particular for classification. The idea is that we have measurements (i.e. sepal length, sepal width, petal length and petal width) for which we know the correct species. So if we go out in nature and find some iris flowers and measure their sepal length, sepal width, petal length and petal width, then we can use the iris dataset to predict which species each flower belongs to. Nice, ha? And since there are three possible values for the iris species, it's a classification task.<br>

# Let's load the dataset with the use of the *load_iris* function. We'll call it *iris_dataset* (but you could call it anything you want).

# In[2]:


from sklearn.datasets import load_iris
iris_dataset = load_iris()


# ### Getting a sense of the data

# The *iris_dataset* object that is returned by *load_iris* is a *Bunch* object, which contains some information about the dataset, as well as the actual data.
# Bunch objects are very similar to dictionaries (we were introduced to dictionaries in Week 1) and they contain keys and values.

# Run the code below to print the keys.

# In[3]:


print("Keys of iris_dataset: ", iris_dataset.keys())


# There are five types of information in the dataset:
# * DESCR
# * feature_names
# * target_names
# * data
# * target

# Let's have a closer look at each one of them.

# _DESCR_ is a short description of the dataset. Run the code below to get an extract of the first 200 characters. If you want to get a bigger extract, all you need to do is change *200* to a larger number.

# In[5]:


print(iris_dataset['DESCR'][:200] + "\n.......")


# *feature_names* corresponds to the names of all the features in the dataset, in other words all the variables that we take into account when building our machine learning model.
# Run the code below to print the names of all features.

# In[6]:


print("Feature names: ", iris_dataset['feature_names'])


# *target_names* corresponds to the class labels.
# By running the code below, we can see that there are three class labels: 'setosa', 'versicolor' and 'virginica'.

# In[7]:


print("Target names: ", iris_dataset['target_names'])


# The actual data is contained in the *data* and *target* fields. 
# *data* contains the values for the different features, e.g. sepal length.

# Run the code in the next cell to get the shape of *data*.

# In[8]:


print(iris_dataset['data'].shape)


# We can see that we have data for 150 iris flowers. For each flower case we have 4 features.

# Run the code in the next cell to get the first three rows in *data*.

# In[17]:


print("First three rows of data:\n", iris_dataset['data'][:3])


# According to this output, we get the following values for the first flower:
# * sepal length (cm): 5.1 
# * sepal width (cm): 3.5
# * petal length (cm): 1.4
# * petal width (cm): 0.2

# **Small challenge**: What if you wanted to get the first 6 rows of data? Write your code below and run it!

# In[18]:


# your code goes here
print("First six rows of data:\n",iris_dataset['data'][0:6])


# Run the code in the next two cells to get the shape of *target* and the first two elements.

# In[19]:


print("Shape of target: ", iris_dataset['target'].shape)


# In[20]:


print("First two elements in target: ", iris_dataset['target'][:2])


# We can see that *target* contains the species for each of the 150 iris flowers in the database. The species of the first two flowers is setosa, as 0 corresponds to setosa, 1 to versicolor and 2 to virginica. (How de we know this? It is a convention that elements in *target_names* appear in an increasing order, starting from 0.)

# <br>
# ## Part 3: Splitting our dataset into training data and test data

# Before using our model for previously unseen iris flowers, we need to know how well it performs. To do this, we split our labelled data in two parts: i) a training dataset that we use for building the model, and ii) a test dataset that we use for testing the accuracy of our model. We do this with the use of the *train_test_split* function, which shuffles the dataset randomly, and by default extracts 75% of the cases as training data and 25% of the cases as test data. 

# Run the code below to split the iris dataset into training and test data. 

# In[21]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)


# This is standard nomenclature. *X* corresponds to data (as in *data* in *iris_dataset*) and *y* to labels (as in *target* in *iris_dataset*). So, for the training dataset we get *X_train* and *y_train*, while for the test dataset we get *X_test* and *y_test*.
# 
# Note that if you wanted to split a different dataset called "my_dataset" into a training and a test dataset, then all you would need to do is substitute "iris_dataset" with "my_dataset" in the code above. 
# 
# By setting *random_state=0* we are making sure that, even though our dataset is randomly shuffled by the *train_test_split* function, we can reproduce our results by using the same fixed seed for the random number generator (in this case 0). So if in the future you want to reproduce the same training and test data, all you need to do is use *random_state=0*.

# Run the code below to get the shape of *X_train* and *y_train*.

# In[22]:


print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)


# In[23]:


print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)


# **Discussion prompt**: What do the outputs from the two previous cells mean? Post your thoughts in the discussion forums!

# Note that it is good practice to visualise our data to get a sense of how different features are related or to spot any abnormalities. We will not do this in this programming task so as to keep things simple and save time, but it is worth keeping in mind for the future.

# <br> 
# ## Part 4: Creating our first model: K Nearest Neighbours

# We will now learn how to build a classification model for the iris dataset with the use of the k nearest neighbours algorithm.

# ### Building the model

# To build a k nearest neighbours model, we will use the *KNeighborsClassifier* class from the *sklearn.neighbors* module.
# 
# Run the code below to create a *KNeighborsClassifier* object called *knn* (but we could give it any name we want). Note that *n_neighbors=1* is setting the number of nearest neighbours to 1.

# In[25]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


# Run the code below to build the model on the training set, i.e. *X_train* and *y_train*.
# 
# You can ignore the output for now.

# In[26]:


knn.fit(X_train, y_train)


# ### Evaluating the model

# We will now use the test dataset to evaluate the accuracy of our model. We can do this with the use of the *score* method, as shown below.

# In[27]:


print("Test set score: ", knn.score(X_test, y_test))


# The code in the next cell contains a variation of the previous code, in case you want to get the value of *knn.score(X_test, y_test)* rounded to three decimal places. If you wanted it rounded to two decimal places, then all you would need to do is change *{:.3f}* to *{:.2f}*.

# In[28]:


print("Test set score rounded to three decimal places: {:.3f}".format(knn.score(X_test, y_test)))


# *How is the accuracy of our model calculated?* Essentially, our model is used to make predictions for *X_test* and the values predicted are compared to the actual labels in *y_test*.

# ### Using the model to make predictions

# We will now use our model to make a prediction about a previously unseen iris flower case. We will first import the numpy libary, then we will specify the previously unseen iris flower case (we'll call it *X_unseen*) and finally we will use the *predict* method on *X_unseen* to get the prediction (we'll call the result *prediction*, but we could use any name we want).

# In[30]:


import numpy as np


# In[32]:


X_unseen = np.array([[5.3, 2.7, 1, 0.3]])


# In[33]:


prediction = knn.predict(X_unseen)

print("Prediction label: ", prediction)
print("Predicted target name: ", iris_dataset['target_names'][prediction])


# According to this output, the prediction for case *X_unseen* is setosa.

# ### Tweaking the model

# We can play around with the model to try different numbers of k nearest neighbours, e.g. 3 or 4.
# 
# **Challenge**: Provide some code below to build and evaluate such a model. All you need to do is reuse and modify parts of the code above.

# In[37]:


# your code goes he2e
knn2 = KNeighborsClassifier(n_neighbors=3)
knn2.fit(X_train, y_train)
print("Test set score: {:.3f}".format(knn2.score(X_test, y_test)))


# <br>
# ## Part 5: Creating a different model: Decision Trees

# We will now learn how to build a classification model for the iris dataset with the use of the decision tree classifier.

# *Important note*: We would normally use the same training and test data as before, so we would reuse *X_train*, *X_test*, *y_train* and *y_test* (so as to compare the results of the K Nearest Neighbours and Decision Tree models). However, in this programming task we will re-split the iris dataset into training and test data, in order to illustrate how we can get a new version by using a different fixed seed (in this case, 7). If you plan to use this notebook as a template for future machine learning projects, then you can delete the next cell. 

# In[39]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=7)


# To build a decision tree model, we will use the *DecisionTreeClassifier* class from the *sklearn.tree* module.
# 
# Run the code below to create a *DecisionTreeClassifier* object called *tree* and to fit the model on the training set, (i.e. *X_train* and *y_train*). You can ignore the information outputed.
# 
# Note that the decision tree classifier algorithm contains some randomness aspects (explaining these is beyond the scope of this course), so by setting *random_state=12* we can reproduce our results by using the same fixed seed for the random number generator (in this case 12).

# In[41]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=12)
tree.fit(X_train, y_train)


# Run the code below to evaluate the accuracy of the decision tree model that we just built. We'll distinguish between accuracy on the training set and accuracy on the test set.

# In[42]:


print("Accuracy on training set: ", tree.score(X_train, y_train))
print("Accuracy on test set: ", tree.score(X_test, y_test))


# The decision tree built has accuracy 100% on the training dataset. This means that our decision tree is over-fitting the training data.
# 
# In order to avoid overfitting (and hopefully improve the accuracy of the model on test data), we can stop before the entire tree is created. We can do this by setting the maximal depth of the tree.
# 
# Run the code below to create a new version of the tree with maximal depth 3. Note that the only difference to the code in the previous cell is *max_depth=3*.

# In[43]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=3, random_state=12)
tree.fit(X_train, y_train)

print("Accuracy on training set: ", tree.score(X_train, y_train))
print("Accuracy on test set: ", tree.score(X_test, y_test))


# The new decision tree has lower accuracy on the training dataset, but higher accuracy on the test dataset.

# We will now use our decision tree to make a prediction for the previously unseen iris case *X_unseen*, which was defined earlier in this notebook.

# In[44]:


prediction = tree.predict(X_unseen)

print("Prediction label: ", prediction)
print("Predicted target name: ", iris_dataset['target_names'][prediction])


# According to this output, the prediction for case *X_unseen* is setosa. This prediction is in line with the prediction that we got using the K Nearest Neighbours classifier.

# <br>
# ## Part 6 (Optional): Practise further

# We highly recommend that you practise further with what you've learnt in this programming task. Here are some ideas to get you started:
# - Build a K Nearest Neighbours model for a different number of neighbours and evaluate it. 
# - Build a Decision Tree model with a different maximal depth and evaluate it.
# - Build a Decision Tree model on the original training data (i.e. for the original split of data with random_state=0) and evaluate it.

# In[45]:


tree2 = DecisionTreeClassifier(max_depth=2, random_state=12)
tree2.fit(X_train, y_train)

print("Accuracy on training set: ", tree.score(X_train, y_train))
print("Accuracy on test set: ", tree.score(X_test, y_test))


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)
tree2 = DecisionTreeClassifier(max_depth=2, random_state=12)
tree2.fit(X_train, y_train)

print("Accuracy on training set: ", tree.score(X_train, y_train))
print("Accuracy on test set: ", tree.score(X_test, y_test))


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)
tree2 = DecisionTreeClassifier(max_depth=2, random_state=0)
tree2.fit(X_train, y_train)

print("Accuracy on training set: ", tree.score(X_train, y_train))
print("Accuracy on test set: ", tree.score(X_test, y_test))

