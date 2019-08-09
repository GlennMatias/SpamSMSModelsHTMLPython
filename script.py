# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 20:21:48 2019

@author: 10012191
"""

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import pandas, xgboost, numpy, textblob, string
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import pickle


from flask import Flask
from flask import request
from flask import render_template


from flask import current_app

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template("index.html") # this should be the name of your html file

@app.route('/', methods=['POST'])
def read_csv():
    dataset = request.files['dataset']
    

    ## read csv
    sms = pd.read_csv(dataset.stream, encoding = 'latin-1')
    
    
    ##
    valid_y = sms["Class"]
    valid_x = sms["Text"]
    
    # label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    valid_y = encoder.fit_transform(valid_y)
    
    ##COUNT VECTORS AS FEATURES

    # create a count vectorizer object 
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(sms['Text'])
    
    # transform the training and validation data using count vectorizer object
    xvalid_count =  count_vect.transform(valid_x)


    #    loaded_logistic_regression = pickle.load(open(str(app.root_path) + r'\NaiveBayes', "rb"))

    loaded_logistic_regression = pickle.load(open(r'C:\Users\10012191\Desktop\Machine learning class work\flash\NaiveBayes', "rb"))
    

    predictions = loaded_logistic_regression.predict(xvalid_count)
    
    output_dataframe = pd.concat([sms, pd.DataFrame(predictions)], axis =1)
    
    return render_template('table.html',  tables=[output_dataframe.to_html(classes='data')], titles=output_dataframe.columns.values)
          

    
  
    





if __name__ == '__main__':
    app.run(debug=True)
