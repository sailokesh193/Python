# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 21:53:39 2018

@author: Sai Lokesh

Copyrights 
"""

import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
data = pd.read_csv('C:\\Users\\Sai\\Desktop\\ML\\Training Data (2).csv')
data.head()
data.shape
from nltk.stem.snowball import SnowballStemmer

#Data Preprocessing
class Text_Chand(object):
    def clean_text(self,data,text1 = []):
        url_reg  = r'[a-z]*[:.]+\S+'
        for i in data['text']:
            text1.append(re.sub(url_reg, '',i))
        return text1
    
    def remove_puntuation(self,text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def to_lower(self,data,list = []):
        for i in data:
            list.append(i.lower())
        return list
    def converting_into_str(self,data,list_stemmed = []):
        data_stemmed = pd.DataFrame(data)
        for line in data_stemmed[0]:
            li = ' '.join(line)
            list_stemmed.append(li)
        return list_stemmed
    def stemming(self,data):
        stemmer = SnowballStemmer('english')
        data_stemmed = [[stemmer.stem(word) for word in text]for text in data]
        return data_stemmed
    
def main():
    try:
        data = pd.read_csv('C:\\Users\\Sai\\Desktop\\ML\\Training Data (2).csv')
        data1 = data.iloc[:,3:5]
    except:
        print('Please provide the correct file path (or) file is missing')
    cl = Text_Chand()
    clean_links = cl.clean_text(data1)
    clean_puntuation = [cl.remove_puntuation(text) for text in clean_links]  
    clean_lower = cl.to_lower(clean_puntuation)

    english_stopwords = set([word for word in nltk.corpus.stopwords.words('english')])

    clean_tokens = [nltk.word_tokenize(text) for text in clean_lower]

    #Stemming
    #data_stemmed = [[stemmer.stem(word) for word in text]for text in clean_tokens]
    data_stemmed = cl.stemming(clean_tokens)
    #Removing Stop words
    data_stemmed = [[token for token in tokens if token not in english_stopwords]for tokens in data_stemmed]
    
    #here iam counting the no of occurences of each word
    word_count = pd.Series(np.concatenate(data_stemmed)).value_counts()
    singular_words = set(word_count[pd.Series(np.concatenate(data_stemmed)).value_counts()==1].index)
    
    #Removing the singular words
    data_stemmed_nonsinglr = [[word for word in text if word not in singular_words]for text in data_stemmed]
    
    #Converting to array
    data_stemmed = np.asarray(data_stemmed_nonsinglr)
    
    
    clean_data = cl.converting_into_str(data_stemmed)
    clean_data = pd.DataFrame(clean_data, columns = ['clean_text'])
    clean_data = clean_data[:1000]
    
    #Converting the tone into categorical values
    from sklearn.preprocessing import LabelEncoder
    lb = LabelEncoder()
    data1['tone']=lb.fit_transform(data1['tone'])
    
    X = clean_data.iloc[:,0]
    y = data1.iloc[:,-1]
    
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)
    
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf = TfidfTransformer()
    X = tfidf.fit_transform(X)
    
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
    
    #Accuracy is like 87% which is pretty good
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    sc = clf.score(X_test,y_test)
    sc = sc*100
    print('---------------------------------------------------------------------------')
    print('The accuracy of naive bayes theorem {0:.2f}%.'.format(sc))
    print('---------------------------------------------------------------------------')
    #Here is Accuracy is around 86.5%  which is also not that bad
    from sklearn.linear_model import SGDClassifier
    text_svm = SGDClassifier(loss = 'hinge',alpha = 1e-2,n_iter = 6,random_state = 42)
    text_svm.fit(X_train,y_train)
    sc1 = text_svm.score(X_test,y_test)
    print('---------------------------------------------------------------------------')
    print('The accuracy of svm is {0:.2f}%.'.format(sc1))
    print('---------------------------------------------------------------------------')
    #Testing data
    test_data = pd.read_csv('C:\\Users\\Sai\\Downloads\\testing data.csv')
    data2 = test_data.iloc[:,3:4]
    
    #cleaning test data
    clean_links = cl.clean_text(data2)
    clean_puntuation = [cl.remove_puntuation(text) for text in clean_links]  
    clean_lower = cl.to_lower(clean_puntuation)
    clean_tokens = [nltk.word_tokenize(text) for text in clean_lower]
    data_stemmed = cl.stemming(clean_tokens)
    #Removing Stop words
    data_stemmed = [[token for token in tokens if token not in english_stopwords]for tokens in data_stemmed]

    #here iam counting the no of occurences of each word
    word_count = pd.Series(np.concatenate(data_stemmed)).value_counts()
    singular_words = set(word_count[pd.Series(np.concatenate(data_stemmed)).value_counts()==1].index)
    
    #Removing the singular words
    data_stemmed_nonsinglr = [[word for word in text if word not in singular_words]for text in data_stemmed]
    
    #Converting to array
    data_stemmed = np.asarray(data_stemmed_nonsinglr) 
    clean_data = cl.converting_into_str(data_stemmed)
    clean_data = pd.DataFrame(clean_data, columns = ['clean_text'])
    X = clean_data
    
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)
    
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf = TfidfTransformer()
    X = tfidf.fit_transform(X)
    
    predictions1 = clf.predict(X_test)
    for pred in predictions1:
        print('%d'%pred)
        
    #Wrinting into csv file
    predictions1.to_csv(analysed, sep='\t')
if __name__ == "__main__":
    # calling main function
    main()




test_data = pd.read_csv('C:\\Users\\Sai\\Downloads\\testing data.csv')
test_data.head()
