
# coding: utf-8

# # Introduction to Programming in Python

# Hello world!
# 
# Run the code in the cell below by selecting it and hitting Shift+Enter.

# In[1]:


2+3


# # Python Basics

# ## Basic data structures

# Let's first create a **numerical variable** (of type integer) called 'n'.

# In[3]:


n=5


# Let's print 'n'.

# In[4]:


print(n)


# Now let's create a **string variable** called 's', and return it (rather than print it).
# 
# Note that you can include more than one lines of code in a cell.
# 
# Also note that code lines starting with the hash symbol _#_ are comments.

# In[5]:


# This is a comment (code lines starting with '#' are comments)
s='yellow'
s


# Ask for the type of 'n'.

# In[7]:


type(s)


# *Small side note:* type( ) and print( ) are examples of functions in Python. This means that someone has specified what they are meant to do when we use them. We'll be using lots of different handy Python functions in this course. This makes our life easier, as this way we don't need to specify all functionality from scratch. Note that we could also specify our own functions in Python, if we wanted to - we won't learn how to do this in this course, so as to keep things simple.

# ### Use the variables created

# Add 3 to 'n'.

# In[8]:


n+3


# Make 's' uppercase.

# In[9]:


s.upper()


# Concatenate 's' with " submarine".

# In[10]:


s+' submarine'


# Overwrite the value of 'n' and print it.

# In[11]:


n=10
print(n)


# Print out information in a user-friendly form (e.g. by combining strings and the values of variables).

# In[12]:


print("The value of n is", n)


# As you can see, we need to include our string in quotes, but the variable doesn't need any quotes. Note also that 
# we use a comma ',' between different items - in our case, between the string *"The value of n is"* and *n*.

# In[13]:


print("Wait! What did you say the value of n was? \nIt is", n,"I said!")


# Note that the special character *\n* signifies a new line.

# ## Python data structures

# ### Tuples

# Create a new tuple called 'my_tuple' and return it.

# In[14]:


my_tuple = (120, 80, 100, 120, 35, 140, 120)
my_tuple


# Access the second element of 'my_tuple'.

# In[15]:


my_tuple[1]


# Try to assign a different value to that element. *This should fail!*

# In[16]:


my_tuple[1]=200


# ### Lists

# Create a new list of string elements.

# In[17]:


animal_list = ['cat', 'dog', 'lion', 'spider', 'eagle']


# Access the element with index 2.

# In[18]:


animal_list[2]


# Add a new element to the list and return the list.

# In[19]:


animal_list.append('bee')
animal_list


# Remove the 'dog' element from the list and return the list.

# In[20]:


animal_list.remove('dog')
animal_list


# ### Dictionaries

# Create a new dictionary called 'ages' and return it.

# In[21]:


ages = {"Ian" : 40, "Alice" : 25, "Kate" : 65}
ages


# Get Kate's age, i.e. the value for key 'Kate'.

# In[29]:


ages["Kate"]


# Modify Ian's age.

# In[25]:


ages["Ian"]=42


# # Data Structures in the Pandas Library

# We first need to tell Python that we're going to be using Pandas. We do this with the *import* command.

# In[30]:


import pandas as pd


# ## Series

# Create a series called 'my_grades' and return it.

# In[31]:


my_grades = pd.Series([89, 72, 55, 93], index=['Math', 'English', 'French', 'Chemistry'])
my_grades


# Print the index of 'my_grades' and then print the values of 'my_grades'.

# In[33]:


print(my_grades.index)
print(my_grades.values)


# Get the value for the element with index 'French'.

# In[35]:


my_grades['French']


# ## DataFrames

# Create a new DataFrame and return it.

# In[36]:


data = {'studentID': ['s08549383', 's062184743', 's17758784', 's17439450'],
        'name': ['John', 'Rhona', 'Clara', 'Dave'],
        'age': [22, 20, 36, 22],
        'grade': [55, 86, 62, 38]}
physics_class = pd.DataFrame(data)
physics_class


# We can specify the sequence of columns when creating our DataFrame.

# In[37]:


physics_class = pd.DataFrame(data, columns=['studentID', 'name', 'age', 'grade'])
physics_class


# Get the 'name' column.

# In[38]:


#physics_class['name']
physics_class.name


# Now let's add a column called 'degree' and return the 'physics_class' DataFrame.

# In[39]:


physics_class['degree'] = ['Biology', 'Physics', 'Physics', 'Medicine']
physics_class

