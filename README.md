# Predicting Readability of Texts Using Machine Learning


```diff
+  The dataset was taken from https://www.kaggle.com/c/commonlitreadabilityprize/data.
```

## Problem Statement 
When we look at the present age, we see a __lot of articles__ and __reading materials__ available by different __authors__ and __bloggers__. As a result, we see text all around us and there is a lot to do with the text. Since there are more and more books and articles being produced everyday, it becomes important that one uses the text information and understand them so that they would be able to make the best use of it. 

When there are __millions of documents__ and __publications__ introducted, it is often not possible for an average reader to classify them based on their difficulty. This is because one would have to go over the materials fully and understand it before assigning the difficulty of the text. Therefore, we should think of ways at which we could automate this system which would classify the documents into categories.


Since there are a lot of __publications__ and __articles__ being published everyday, it sometimes becomes tedious and difficult for the __librarians__ to go over the materials and classify them based on their level of comprehension. As a result, a high difficulty text might be given to a child who is just about 10 years of age. On the contrary, a low difficulty text might be given to a highly educated individual who might easily understand the text that lacks much knowledge.

Therefore, it would be of great help to librarians and readers if there are algorithms that could classify the text based on the difficulty without these people having to go through the documents. As a result, this reduces the manpower needed to read the books and also saves a lot of time and effort on the part of humans. 

## Machine Learning and Deep Learning 

* With __machine learning__ and __deep learning__, it is possible to predict the readability of the text and understand some of the important features that determine the difficulty respectively. 
* Therefore, we have to consider a few important parameters when determining the difficulty of different machine learning models respectively.
* We have to take into consideration the difficulty of the text along with other important features such as the number of syllables and the difficulty of the words in order to determine the overall level of the text. 

## Natural Language Processing (NLP)

* We have to use the __natural language processing (NLP)__ when we are dealing with the text respectively.
* Since we have a text, we have to use various processing techniques so that they are considered into forms that could be easy for machine learning purposes.
* Once those values are converted into vectors, we are going to use them by giving them to different machine learning and deep learning models with different set of layers respectively.
* We would be working with different __machine learning__ and __deep learning algorithms__ and understand some of the important metrics that are needed for the problem at hand. 
* We see that since the target that we are going to be predicting is continuous, we are going to be using the regression machine learning techinques so that we get continuous output.

## Vectorizers

There are various vectorizers that were used to convert a given text into a form of a numeric vector representation so that it could be given to machine learning models for predictions for difficulty. Below are some of the vectorizers used to convert a given text to vectors.

* [__Count Vectorizer__](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
* [__Tfidf Vectorizer__](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
* [__Average Word2Vec (Glove Vectors)__](http://ethen8181.github.io/machine-learning/keras/text_classification/word2vec_text_classification.html)
* [__Tfidf Word2Vec__](https://datascience.stackexchange.com/questions/28598/word2vec-embeddings-with-tf-idf)

## Machine Learning Models

The output variable that we are considered is a continous variable, therefore, we should be using regression techniques for predictions. Below are some of the machine learning and deep learning models used to predict the difficulty of texts.

* [__Deep Neural Networks__](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
* [__Linear Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
* [__K - Neighbors Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
* [__PLS Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html)
* [__Decision Tree Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
* [__Gradient Boosting Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)

## Outcomes
* __TFIDF Word2Vec__ Vectorizer was the best encoding technique which results in significant reduction in the __mean absolute error__ respectively. 
* __Gradient Boosted Decision Trees (GBDT)__ were performing the best in terms of the __mean absolute__ and __mean squared error__ of predicting the difficulty of texts.

## Future Scope 
* The best model (__Gradient Boosted Decision Trees__) could be integrated in real-time in Office tools such as __Microsoft Word__ and __Microsoft Presentation__ so that an user can get an indication of the difficulty of his/her sentences.
* Additional text information could be added from other sources such as __Wikipedia__ to further reduce the __mean absolute error__ of the models.  
