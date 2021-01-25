import re
import pickle
import joblib
import numpy as np
import pandas as pd
from nltk.corpus import stopwords as sw
from nltk.stem.porter import PorterStemmer as ps
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix as cm, roc_auc_score as ras
from sklearn.feature_extraction.text import CountVectorizer as cv, TfidfVectorizer


def data_reading():
    global df
    
    df = pd.read_csv('balanced_reviews.csv')
    df.shape
    df.columns.tolist()
    df.dtypes
    df['overall'].value_counts()
    df.isnull().any(axis = 0)
    df.isnull().any(axis = 1)
    df[ df.isnull().any(axis = 1) ]
    df.dropna(inplace = True) # Removing null values from dataframe
    df = df[df['overall'] != 3] # Removing values corresponding to 3 star review
    df['positivity'] = np.where(df['overall'] > 3, 1, 0 ) # Creating a new column for specified condition
    

def train_test_split(features, labels,  random_state):
    global features_train, features_test, labels_train, labels_test
    
    features_train, features_test, labels_train, labels_test= tts( features, labels, random_state )


def version1(): # Logistic Regression Model
    train_test_split(df["reviewText"], df["Positivity"], 100)
    
    features_train_vectorized = cv().fit_transform(features_train)
    features_test_vectorized = cv().transform(features_test)

    model = lr().fit(features_train_vectorized, labels_train) # Model creation for logistic regression
    predictions = model.predict(features_test_vectorized)

    ras(labels_test, predictions) # Generating prediction score
    cm(labels_test, predictions)
    
    return model


def version2(): # Data cleaning in NLP Model
    corpus = []
    
    for i in range(0, 527383):
        review = re.sub( '[^a-zA-Z]', ' ', df.iloc[i, 1] ) # Removing all elements except words from all reviews
        review = review.lower()
        review = review.split()
        review = [ word for word in review if not word in set( sw.words('english') )]
        stammer = ps()
        review = [ stammer.stem(word) for word in review ]
        review = " ".join(review)
        corpus.append(review)
        
    features = cv().fit_transform(corpus)
    labels = df.iloc[:, -1]
    
    train_test_split(features, labels, 100)
    
    features_test_vectorized = cv().transform(features_test)
    features_train_vectorized = cv().fit_transform(features_train)

    model = lr().fit(features_train_vectorized, labels_train)
    predictions = model.predict(features_test_vectorized)
    ras(labels_test, predictions)
    cm(labels_test, predictions)
    
    return model


def version3(): # TF_IDF Model
    global vect
    
    train_test_split(df["reviewText"], df["Positivity"], 100)
    
    vect = TfidfVectorizer(min_df = 5)
    features_train_vectorized = vect.fit_transform(features_train)
    features_test_vectorized = vect.transform(features_test)
    
    model = lr().fit(features_train_vectorized, labels_train)
    predictions = model.predict(features_test_vectorized)
    ras(labels_test, predictions)
    cm(labels_test, predictions)
    
    return model


def saving_model(lib_name, model):
    if(lib_name == "pickle"):
        file = open("pickle_model.pkl", 'wb')
        pickle.dump(model, file)
        
        file2 = open("feature.pkl", 'wb')
        pickle.dump(vect.vocabulory_, file2)
    else:
        file = open("joblib_model.jlb", 'wb')
        joblib.dump(model, file)
        

def saved_model(lib_name):
    if(lib_name == "pickle"):
        file = open("pickle_model.pkl", 'rb')
        saved_model = pickle.load(file)
        
        global saved_model
    else:
        file = open("joblib_model.jlb", 'rb')
        saved_model = joblib.load(file)
        
        global saved_model
    
    file2 = open("feature.pkl", 'rb')
    saved_vocab = pickle.load(file2)
    global saved_vocab

def main():  # Main Function
    data_reading()
    
    version_name = input("Enter version number to be used { 1(Logistic Regression Model) / 2(Data cleaning in NLP Model) / 3( TF_IDF Model) }: ")
    
    if(version_name == 1):
        model  = version1()
    elif(version_name == 2):
        model  = version2()
    elif(version_name == 3):
        model  = version3()
    else:
        print("Wrong value for version name!!")
    
    lib_name1 = input("Enter library name to be used for saving model { pickle / joblib }: ")
    saving_model(lib_name1, model)
    
    lib_name2 = input("Enter library name to be used for using saved model { pickle / joblib }: ")
    saved_model(lib_name2)
    
    review_input = input("Enter review: ")
    saved_model.predict(review_input)    
    
    
if __name__ == '__main__':  # To call main function
    main()