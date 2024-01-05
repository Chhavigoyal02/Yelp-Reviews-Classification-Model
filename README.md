# Yelp-Reviews-Classification-Model

# PROBLEM STATEMENT
- In this project, Natural Language Processing (NLP) strategies will be used to analyze Yelp reviews data
- Number of 'stars' indicate the business rating given by a customer, ranging from 1 to 5
- 'Cool', 'Useful' and 'Funny' indicate the number of cool votes given by other Yelp Users.

# PROJECT OVERVIEW:
-- First of all,we need to check whether it is regression task or classifiction task...
        
    It is a classification task as we have to classify whether a specific review is a positive review or a negative one .
-- Here,we are going to use Naive Bayes and Natural Language Processing(NLP) to get this task done.

Natural Language Processing(NLP) enables computers to understand natural language as humans do. Whether the language is spoken or written, natural language processing uses artificial intelligence to take real-world input, process it, and make sense of it in a way a computer can understand.

  # STEPS TAKEN:

So,here are the step I followed in this problem statement:

--> Step-1: Importing the libraries and Loading the dataset.
- Begin by importing essential libraries such as NumPy for basic calculations, Pandas for handling dataframes, Matplotlib for plotting graphs, and Seaborn for visualizing data. 
- Load the dataset named "emails" from a CSV file using the Pandas library.

--> Step-2: Data Visualization
- First, add new column of length of a review in datset which is going to be useful in our analysis in future.
- Plot a histogram depicting the relationship between review length and frequency.
- Make new 2 datsets for 1-star reviews and 5-star reviews separately.
- Calculate the percentage distribution of !-star review and 5-star review.

--> Step-3 Creating testing and training dataset/data cleaning.
- Initiate data cleaning by removing punctuation from the text, as it does not contribute significantly to the analysis.
- Then Eliminate stopwords, common words in the English language that don't provide substantial information. Stop words are commonly used words in a language that are used in Natural Language Processing (NLP). They are used to remove words that are so common that they don't provide much useful information. Eg- A,An,The,On,Of,We,I.
- Now by using count vectorizer, we convert a collection of text documents into a numerical representation. It is part of the scikit-learn library, a popular machine learning library in Python.

--> Step-4 Training the model 
- First spilt the dataser into four parts: X_train,X_test,y_train,y_test.
- Importing the Multinomial Naive bayes classifier for training purposes.
- Then fit the classifier using our training set.

--> Step-5 Evaluating the model
- First import the tools such as classification report and confusion matrix.
- Now plot the confusion matrix for training and testing set.A confusion matrix is a table that summarizes the performance of a classification model. It's a performance evaluation tool in machine learning that displays the number of true positives, true negatives, false positives, and false negatives. 
- Then make classification report of model on testing set.The classification report shows a representation of the main classification metrics on a per-class basis.The classification report visualizer displays the precision, recall, F1, and support scores for the model.

--> Step-6 Adding additional feature TF-IDF
- Tf–idf stands for "Term Frequency–Inverse Document Frequency" is a numerical statistic used to reflect how important a word is to a document in a collection or corpus of documents. TFIDF is used as a weighting factor during text search processes and text mining.
- For this, we use TfidfTransformer. First fit the training set into this classifier and then repeat all the process on new dataset.


