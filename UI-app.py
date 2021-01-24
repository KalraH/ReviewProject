"""
Created on Thru Jan 7 17:32:33 2021

@author: Hariom Kalra
"""

# Importing Libraries
import dash
import pickle
import webbrowser
import pandas as pd
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.feature_extraction.text import TfidfTransformer as tfidfT, TfidfVectorizer as tfidfV


app = dash.Dash() # Creating Applicatiton for Web Interface


def load_model(): # Function for model loading
    global pickle_model, vocab, dropdown_options, name_PieChart, values_PieChart
    
    df = pd.read_csv('scrapped_reviews.csv')
    
    dropdown_options = [ { "label" : str(i), "value" : str(i) } for i in sorted( df['Reviews'].unique().tolist() ) ]
    
    values_PieChart = [0, 0]
    name_PieChart = ['Positive', 'Negative']
    
    for i in range( 0, len(df['Positivity']) ):
        
        if( df['Positivity'][i] == 1 ):
            values_PieChart[0] += 1
        
        else:
            values_PieChart[1] += 1
    
    
    file = open("pickle_model.pkl", 'rb')
    pickle_model = pickle.load(file) # Importing saved model
    
    file1 = open("feature.pkl", 'rb')
    vocab = pickle.load(file1) # Getting Vocabulary to predict given reviews
    
    print( df.sample(5) ) 


def open_browser(): #  Function to open Web-Browser 
    webbrowser.open_new( 'http://127.0.0.1:8050/' ) # Opening Browser


def create_app_ui(): # Function for UI creation
    dropdown_style = { 'text-align' : 'center',
                       'margin-left' : '100px',
                       'width' : '80%' }
    
    H1style = html.style = { 'textAlign' : 'center', 
                             'background-color' : '#80D8FF',
                             'color' : '#E53935',
                             'border' : '6px solid #1DE9B6',  
                             'font-style' : 'italic',
                             'font-weight' : 'bold',
                             'padding-top' : '10px', 
                             'padding-bottom' : '10px' }
    
    H2style = html.style = { 'background-color' : '#d3fcc5',
                             'color' : '#0091EA',
                             'width' : '80%',
                             'font-style' : 'italic', 
                             'font-weight' : 'bold',
                             'border' : '3px solid #FF0266',
                             'padding-top' : '10px',
                             'padding-left' : '10px',
                             'padding-bottom' : '10px' }
    
    TextAreaStyle = html.style = { 'border' : '6px solid #1DE9B6',
                                   'padding-top' : '15px',
                                   'width' : '80%',
                                   'padding-bottom' : '15px' }
    
    ButtonStyle = html.style = { 'border-color' : '6px solid violet',
                                 'background-color' : '#ABFAAA',
                                 'padding-top' : '10px',
                                 'width' : '300px',
                                 'padding-bottom' : '10px' }
    
    pie_fig = px.pie( names = name_PieChart, 
                      values = values_PieChart, 
                      title = "Data Status" )
    
    pie_style = { 'width' : '600px',
                  'height' : '450px' }


    main_layout = html.Div( html.Center([ html.H1( id = 'Main_title', 
                                                   children = 'Sentiments Analysis with Insights', 
                                                   style = H1style ),
                                          
                                          dcc.Graph( id = "pie_Chart",
                                                     style = pie_style,
                                                     figure = pie_fig ),
                                          
                                          dcc.Textarea( id = 'textarea_review', 
                                                        placeholder = "Enter the review here......", 
                                                        style = TextAreaStyle ), 
                                          
                                          html.Br(), html.Br(),
                                          
                                          html.Button( id = "Text_button", 
                                                       children = "Submit", 
                                                       n_clicks = 0, 
                                                       style = ButtonStyle ), 
                                          
                                          html.Br(), html.Br(),
                                          
                                          dcc.Dropdown( id = 'dropdown', 
                                                        options = dropdown_options, 
                                                        placeholder = 'Select Review Text', 
                                                        style = dropdown_style ),
                                          
                                          html.Br(), 
                                          
                                          html.Button( id = "DD_button",
                                                       children = "Submit",
                                                       n_clicks = 0,
                                                       style = ButtonStyle ), 
                                          
                                          html.Br(), html.Br(),
                                          
                                          html.H2( id = 'TextArea_result_display',
                                                   children = None,
                                                   style = H2style,
                                                   hidden = True ),
                                          
                                          html.H2( id = 'Dropodown_result_display',
                                                  children = None,
                                                  style = H2style,
                                                  hidden = True ),
                                          
                                          html.Br(), html.Br()
                                       ])
                           
                          )
    
    return main_layout


def check_review(reviewText): # Function for Checking the User Entered Reviews
    transformer = tfidfT()
    loaded_vec = tfidfV( decode_error = "replace", vocabulary = vocab )
    reviewText = transformer.fit_transform( loaded_vec.fit_transform( [reviewText] ) )

    return pickle_model.predict(reviewText)


# Using Callback mechaism for updating UI as per the requirements
@app.callback(
    Output('TextArea_result_display', 'hidden'),
    [
         Input('Text_button', 'n_clicks')
    ])
def update_TextArea_button(clicks):
    print("Clicked value = ", clicks)
    
    if(clicks > 0):
        return False
    else:
        return True


@app.callback(
    Output('Dropodown_result_display', 'hidden'),
    [
         Input('DD_button', 'n_clicks')
    ])
def update_DD_button(clicks):
    print("Clicked value = ", clicks)
    
    if(clicks > 0):
        return False
    else:
        return True


@app.callback(
    Output('TextArea_result_display', 'children'),
    [
        Input('textarea_review', 'value')
    ])
def update_TextArea_ui(textAreaReview):
    print("Data type = ", str(type(textAreaReview)))
    print("value = ", str(textAreaReview))
    
    result_list = check_review(textAreaReview)
    
    if (result_list[0] == 0 ):
        result = "Negative"

    elif (result_list[0] == 1 ):
        result = "Positive"

    else:
        result = "Unknown"
    
    return result

    
@app.callback(
    Output('Dropodown_result_display', 'children'),
    [
        Input('dropdown', 'value')
    ])
def update_DD_ui(DDAreaReview):
    print("Data type = ", str(type(DDAreaReview)))
    print("value = ", str(DDAreaReview))
    
    result_list = check_review(DDAreaReview)
    
    if (result_list[0] == 0 ):
        result = "Negative"
    
    elif (result_list[0] == 1 ):
        result = "Positive"
    
    else:
        result = "Unknown"
    
    return result


def main():  # Main Function
    global app

    # Calling Fuctions
    load_model()
    open_browser()
    
    app.title = "Sentiments Analysis with Insights"  # Setting title of Web-Page
    app.layout = create_app_ui()  # Creating Layout of Application
    app.run_server()  # Starting / Running Server of Web-Page

    print("This would be executed only after the script is closed") 
    app = None  # Making all Global variables None


if __name__ == '__main__':  # Code to call main function
    main()