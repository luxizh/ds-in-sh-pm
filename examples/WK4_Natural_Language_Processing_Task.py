
# coding: utf-8

# # Natural Language Processing Task

# In this programming task, we'll learn how to do some basic natural processing tasks, e.g. tokenisation, stemming and named entity recognition. We'll get to work with two fictitious clinical pieces of text: a biopsy report and a medical note.

# ## Part 1: Importing packages needed

# ### Importing NLTK

# *NLTK* (Natural Language Toolkit) is a very popular suite of libraries and other resources for natural language processing. Let's tell Python that we're going to be using NLTK, with the use of the import command.

# In[1]:


import nltk


# We will also import a few more NLTK utilities for tokenisation, stemming, part of speech tagging and named entity recognition.

# In[2]:


from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


# ### Importing docx2txt

# *docx2txt* is a Python-based utility which converts text within docx files into plain text. We'll import this in order to be able to work with the two clinical docx files.

# In[3]:


import docx2txt


# <br>
# ## Part 2: Processing a Biopsy Report

# ### Loading the data

# The Biopsy Report is a fictitious example for the purpose of this programming task. It contains biopsy details about a mastecotmy specimen from a patient.
# 
# The Biopsy Report is a Microsoft Word document (in a docx file format). In order for it to be processed by the NLTK package, the document requires to be converted into plain text.  In this case, we are using the *process* method from the docx2txt Python package to do this. We are calling the resulting plain text "text".

# In[4]:


text = docx2txt.process('./readonly/Biopsy_Report.docx')


# By running the code in the following cell, you'll see that text is a string.

# In[5]:


type(text)


# Run the code in the following cell to get the first 160 characters in *text*.

# In[6]:


text[:160]


# You can recognise several words in this text, e.g. "Biopsy Report" and "General Surgery". Note that there are some special characters too:
# - \n corresponds to new line
# - \t corresponds to tab space

# ### Tokenisation

# Tokenisation breaks large strings into smaller chunks. Let's use NLTK's *word_tokenize* method to split the *text* string into words and punctuation. We'll call the resulting list of words and punctuation *tokens*.

# In[7]:


tokens = nltk.word_tokenize(text)


# Run the code in the following cell to get the first 10 elements of *tokens*. As you can see, there are both word- and punctuation-based tokens.

# In[8]:


tokens[:10]


# #### Removing stop words

# Now we want to remove any stop words (i.e. any commonly used words, such as "and") from the list of tokens. We'll call the resulting list of tokens *clean_tokens*. <br><br>
# Note that you don't need to understand the code in the following cell. You can simply reuse it in the future as a template, where you replace *tokens* with the name of your list of tokens.
# (But if you're really curious, this is what the following code does: It first creates a copy of *tokens* called *clean_tokens*. And then for each token in the list of tokens, if it's in the list of stop words in English, then it is removed.)

# In[9]:


clean_tokens = tokens[:]
for token in tokens:
    if token in stopwords.words('english'):
        clean_tokens.remove(token)


# Let's print out the number of tokens (with and without stop words).

# In[12]:


print("Number of tokens including stop words:  ",len(tokens))
print("Number of tokens excluding stop words:  ",len(clean_tokens))


# As you can see, there are not that many stop words in the Biopsy Report. However, removing stop words may be really useful when working with documents containing many pages of text.
# 

# #### Frequency distributions

# Next, we want to investigate the frequency of certain words in the Biopsy Report. We start by generating the frequency distribution of all tokens (including stop words) with the use of NLTK's FreqDist. Let's call this *freq*.

# In[16]:


freq = nltk.FreqDist(tokens)
type(freq)


# Run the code in the following cell to get the 10 most frequently encountered tokens and their frequencies.

# In[17]:


freq.most_common(10)


# Taking "lesion" and "lesions" as an example, we can see how many times they appear in the document. Note that we're distinguishing beween different versions of the word, such as upper case and lower case. These are not the only possible variations, they're just the ones we're interested in right now.

# In[18]:


print("Frequency of lesion:  ", freq["lesion"])
print("Frequency of lesions: ", freq["lesions"])
print("Frequency of LESION:  ", freq["LESION"])
print("Frequency of LESIONS: ", freq["LESIONS"])


# #### Lower case vs. upper case text

# It can be useful to disregard upper case text and lower case text.  Therefore, we can convert upper case text to lower case text. We can do this with the use of Python's *lower()* method. <br>
# 
# Let's apply the *lower()* method to each element in *tokens*, so that all text is counted as lower case text. We'll call this new list of lower case tokens *lowercase_tokens* (but you could call it anything you want).

# In[19]:


lowercase_tokens = [t.lower() for t in tokens]


# Run the code in the following cell to get the first 10 elements of *lowercase_tokens*. As you can see, this is all lower case.

# In[20]:


lowercase_tokens[:10]


# Next, we'll generate the frequency distribution of *lowercase_tokens*, and we'll call it *lowercase_freq*.

# In[21]:


lowercase_freq = nltk.FreqDist(lowercase_tokens)


# Now, let's print out the frequency of lesion(s)/LESION(S) as an example.   
# Observe that the frequencies for LESION and LESIONS are now 0.  

# In[22]:


print("Frequency of lesion:  ", lowercase_freq["lesion"])
print("Frequency of lesions: ", lowercase_freq["lesions"])
print("Frequency of LESION:  ", lowercase_freq["LESION"])
print("Frequency of LESIONS: ", lowercase_freq["LESIONS"])


# ### Stemming

# Stemming is the process of reducing a word to its stem.
# 
# NLTK includes several off-the-shelf stemmers. In this task, we'll be using the Porter stemmer, which implements the Porter stemming algorithm. 

# In[24]:


stemmer = nltk.PorterStemmer()


# Run the code in the following cell to get the stems of all tokens in the Biopsy Report.<br>
# Note that we're working with the converted lower case tokens. So all that we do here is apply *stem* on each token in *lowercase_tokens*. The resulting list of stemmed tokens (i.e. tokens where any affixes have been removed) is called *stem_tokens*. 

# In[25]:


stem_tokens = lowercase_tokens
stem_tokens[:] = [stemmer.stem(lt) for lt in lowercase_tokens]


# Next, we'll generate the frequency distribution of *stem_tokens*, and we'll call it *stem_freq*.

# In[26]:


stem_freq = nltk.FreqDist(stem_tokens)


# Now, lets print out the freqency of lesion(s)/LESION(S) as an example.  
# Note that the frequency of "lesions" is now 0, while the frequency of "lesion" is 6.  

# In[27]:


print("Frequency of lesion:  ", stem_freq["lesion"])
print("Frequency of lesions: ", stem_freq["lesions"])
print("Frequency of LESION:  ", stem_freq["LESION"])
print("Frequency of LESIONS: ", stem_freq["LESIONS"])


# <br>
# ## Part 3: Processing a Medical Note

# ### Loading the data

# The Medical Note is another made-up example for the purpose of this programming task. It contains information about a patient with low back pain.
# 
# Just like the Biopsy Report, the Medical Note is a Microsoft Word document (in a docx file format), so we'll use the *process* method from the docx2txt Python package to convert it into plain text. The resulting plain text is called "content". 

# In[28]:


content = docx2txt.process('./readonly/Medical_Note.docx')


# Run the code in the following cell to get the first 160 characters in *content*.

# In[29]:


content[:160]


# ### Tokenisation

# We'll first tokenise the Medical Note content into sentences. We'll do this with the use of NLTK's *sent_tokenize* method and we'll call the resulting list of sentences *sents*.

# In[30]:


sents = nltk.sent_tokenize(content)


# Run the code in the following cell to get the first 4 elements of *sents*, i.e. the first 4 sentences.

# In[31]:


sents[:4]


# We next want to further tokenise the second sentence into words and punctuation. Just like with the Biopsy Report, we'll use NLTK's *word_tokenize* method. <br>
# Remember that lists in Python are 0-indexed, so the second sentence has index 1 in *sents*. <br> 
# We'll call the resulting list of words and punctuation *medical_tokens*.

# In[32]:


medical_tokens = nltk.word_tokenize(sents[1])


# Let's get the entire *medical_tokens* list.

# In[33]:


medical_tokens


# ### Part-of-Speech Tagging

# A Part-of-speech tagger (or POS-tagger) processes a sequence of words and attaches a part of speech tag to each word.  
# 
# Run the code in the following cell to use NLTK's *pos_tag()* to attach a part of speech tag to each token in the second sentence. The resulting list is called *tagged*.

# In[34]:


tagged = nltk.pos_tag(medical_tokens)


# Now, let's ask for *tagged*.

# In[35]:


tagged


# Here you can see many different POS tags, such as "JJ" and "NN". 
# 
# The meaning of commonly used POS tags is provided below:
# - NN: noun 
# - JJ: adjective
# - DT: determiner
# - VB: verb in base form
# - VBD: verb in past tense
# - VBG: verb in gerund/present participle
# - RB: adverb
# - WRB wh-abverb (e.g. where)
# - IN: preposition/subordinating conjunction
# - CC: coordinating conjunction (e.g. and)

# ### Named Entity Recognition

# Now that we have the parts of speech, we can try named entity recognition, which is concerned with the task of finding entities in text and classifying them as persons, locations, dates, and so on.<br>
# 
# NLTK provides a classifier that has already been trained to recognise named entities, and which can be accessed with the *nltk.ne_chunk()* function. We'll apply it to *tagged* and call the resulting structure *entities*.

# In[36]:


entities = nltk.ne_chunk(tagged)


# Let's print out *entities*. 

# In[38]:


print(entities)


# As we can see, "Patient" has been correctly classified as a Person.

# <br>
# ## Part 4: Keep practising

# ### Quiz 4 - Question 5

# Make changes to the code below to find out how many times the word ‘malignant’ appears in the Biopsy Report.  This should include upper and lower case variations of this word – think about this carefully!  <br><br>
# Note: *freq* has been previously generated above for the programming task.  

# In[41]:


print("Frequency of malignant:  ", freq["malignant"])
print("Frequency of malignants: ", freq["malignants"])
print("Frequency of MALIGNANT:  ", freq["MALIGNANT"])
print("Frequency of MALIGNANTS: ", freq["MALIGNANTS"])

print("Frequency of malignant:  ", lowercase_freq["malignant"])
print("Frequency of malignant:  ", lowercase_freq["malignants"])
print("Frequency of malignant:  ", stem_freq["malign"])

# ### Further analysis of interest

# Why not analyse the two documents further? You could have a play with other sentences in the Medical Note document. Or you could try POS tagging for the Biopsy Report!

# In[44]:


tagged_report=nltk.pos_tag(tokens)
print(tagged_report)


# In[46]:


entities=nltk.ne_chunk(tagged_report)
print(entities)


# In[48]:


clean_m_tokens = medical_tokens[:]
for token in medical_tokens:
    if token in stopwords.words('english'):
        clean_m_tokens.remove(token)


# In[49]:


print(clean_m_tokens)

