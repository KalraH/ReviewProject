"""
Created on Sun Jan 24 12:32:33 2021

@author: Hariom Kalra
"""

# Importing Libraries
import pickle
import pandas as pd
import sqlite3 as sql
from time import sleep
from selenium import webdriver
from sklearn.feature_extraction.text import TfidfTransformer as tfidfT, TfidfVectorizer as tfidfV


def scrap_reviews():
    global reviews_list
    
    reviews_list = []
    url = 'https://www.etsy.com/in-en/listing/546737718/star-ear-crawler-earrings-925-sterling?ga_order=most_relevant&ga_search_type=all&ga_view_type=gallery&ga_search_query=&ref=sr_gallery-1-8&bes=1'
    
    browser = webdriver.Chrome( executable_path = "D:\Desktop\HK_CS_Labs\Python\Projects\ML & NLP Project\Project Stuf\chromedriver" )
    
    browser.get(url)
    sleep(2)
    
    for i in range(1,51):
        try:
            for j in range(4):
                h = browser.find_element_by_xpath( '//*[@id="review-preview-toggle-' + str(j) + '"]' )
                reviews_list.append(h.text)
                sleep(1)
        except:
            pass
        
        next_1 = browser.find_element_by_xpath( '//*[@id="reviews"]/div[2]/nav/ul/li[position() = last()]/a' )
        next_1.click()
        sleep(2)
        
    browser.quit()
    load_model()


def load_model():
    global pickle_model, vocab
    
    file = open("pickle_model.pkl", 'rb')
    pickle_model = pickle.load(file) # Importing saved model for using
    
    file1 = open("feature.pkl", 'rb')
    vocab = pickle.load(file1)
    
    positivity_column()


def positivity_column():
    global result_list
    
    transformer = tfidfT()
    loaded_vec = tfidfV( decode_error = "replace", vocabulary = vocab )
    result_list = []
    
    for i in range(0, len(reviews_list)):
        reviewText = transformer.fit_transform( loaded_vec.fit_transform( [reviews_list[i]] ) )
        
        if(pickle_model.predict(reviewText)[0] == 1):
            result_list.append(1)
        else:
            result_list.append(0)


def creating_CSV():
    df = pd.DataFrame()
    df["Reviews"] = reviews_list
    df["Positivity"] = result_list
    df.to_csv('scrapped_reviews.csv', index = False)


def creating_DB():
    df = pd.read_csv('scrapped_reviews.csv')
    conn = sql.connect('scrapped_reviews.db')
    df.to_sql('reviewstable', conn, index = False)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM reviewstable')
    
    for record in cursor: 
        print(record)
        
    
def main():  # Main Function

    # Calling Fuctions
    scrap_reviews()
    creating_CSV()
    creating_DB()
    

if __name__ == '__main__':  # Code to call main function
    main()