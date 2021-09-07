import time
start_time = time.time()
import os
import pyreadstat
from statsmodels.formula.api import ols
import pandas as pd
from collections import Counter
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import pickle
import sys



def train_binary():
    # os.chdir('C:\\Users\\Hiro Nagata\\Documents\\Python\\Scrapping') #TODO write relative paths

    data3 = pd.read_csv("covid-news.csv") #TODO change to the full processing algo
    data = data3

    # 1. Dataset preparation
    # load the dataset

    a = data['Title'].tolist()
    b = data['Label'].tolist()

    labels, texts = [], []

    for i in range(len(a)):
        content = a[i].split()
        #content = line.split()
        lab = b[i]
        labels.append(lab)
        texts.append(" ".join(content))
        
    # preprocessing
    
    import string
    #Lowering
    texts_lower = [t.lower() for t in texts]

    #Tokenizing
    import re
    WORD = re.compile(r'\w+')

    texts_tokenized = [WORD.findall(t) for t in texts_lower]
    """
    from nltk.tokenize import sent_tokenize
    texts_sent_toke = [sent_tokenize(t) for t in texts_lower]
    """
    #Removing punctuation
    import re
    regex = re.compile('[%s]'% re.escape(string.punctuation))

    texts_fully_tokenized = []

    for review in texts_tokenized:
        new_review = []
        for token in review:
            new_token = regex.sub(u'', token)
            if not new_token == u'':
                new_review.append(new_token)

        texts_fully_tokenized.append(new_review)


    #Removing stopwords
    from nltk.corpus import stopwords

    tokenized_docs_no_stopwords = []

    #This takes a little
    for doc in texts_fully_tokenized:
        new_term_vector = []
        for word in doc:
            if not word in stopwords.words('english'):
                new_term_vector.append(word)

        tokenized_docs_no_stopwords.append(new_term_vector)

    #Ainda não textei aqui embaixo
    #Stemming and Lemmatization
    from nltk.stem.wordnet import WordNetLemmatizer

    wordnet = WordNetLemmatizer()

    preprocessed_docs = []
    start_time = time.time()
    for doc in tokenized_docs_no_stopwords:
        final_doc = []
        for word in doc:
            final_doc.append(wordnet.lemmatize(word))

        preprocessed_docs.append(final_doc)

    # create a dataframe using texts and lables
    trainDF = pd.DataFrame()
    trainDF['text'] = texts
    trainDF['label'] = labels


    # 2. Feature Engineering

    # split the dataset into training and validation datasets 
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

    # label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)

    # 2.1 Count Vectors as features

    # create a count vectorizer object 
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(trainDF['text'])

    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x)


    # 3. Model Building

    results = []

    def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)
        
        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)
        
        if is_neural_net:
            predictions = predictions.argmax(axis=-1)
        
        return metrics.accuracy_score(predictions, valid_y), classifier

    def train_model2(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)
        
        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)
        
        if is_neural_net:
            predictions = predictions.argmax(axis=-1)
        
        return predictions





    # 3.1 Naive Bayes

    # Naive Bayes on Count Vectors
    accuracy, classificador = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
    print("NB, Count Vectors: ", accuracy)
    a = "NB, Count Vectors: ", accuracy
    results.append(a)


    #Export all the trained models!

    with open('classificador.pickle', 'wb') as handle:
        pickle.dump(classificador, handle)

    with open('xtrain_count.pickle', 'wb') as handle:
        pickle.dump(xtrain_count, handle)

    with open('train_y.pickle', 'wb') as handle:
        pickle.dump(train_y, handle)

    with open('count_vect.pickle', 'wb') as handle:
        pickle.dump(count_vect, handle)

    



###############################################################################################################################

# CLASSIFICATION

def classify(new_data):

    os.chdir('../MicrosoftEssexBinary/')

    with open('classificador.pickle', 'rb') as handle:
        classificador = pickle.load(handle)

    with open('xtrain_count.pickle', 'rb') as handle:
        xtrain_count = pickle.load( handle)

    with open('train_y.pickle', 'rb') as handle:
        train_y = pickle.load( handle)

    with open('count_vect.pickle', 'rb') as handle:
        count_vect = pickle.load( handle)

    
        
        
    #Classify here with the imported pickle
    texts = []

    texts.append(new_data)


    trainDF2 = pd.DataFrame()

    trainDF2['text'] = texts

    data2 = trainDF2.squeeze()
    data2 =  pd.Series(data2)
    xvalid_count2 = count_vect.transform(data2)

    # 3. Model Building
    results2 = []


    # Naive Bayes on Count Vectors
    # b = train_model2(classificador, xtrain_count, train_y, xvalid_count2)

    classificador.fit(xtrain_count, train_y)
    # predict the labels on validation dataset
    predictions = classificador.predict(xvalid_count2)
    b = predictions
    
    b = "NB, Count Vectors: ", b
    results2.append(b)

    return results2[0][1][0]


    #################################################################
# train_binary()
# new_data = 'Pfizer vaccine are causing troublesome effects on grannies'
# new_data = 'Pfizer vaccine saves lifes'
# result = classify(new_data)
# print(int(result))
# print("--- %s seconds ---" % (time.time() - start_time))
