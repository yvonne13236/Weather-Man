import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd



app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

def predict(to_predict_list):
    
    f = pd.read_csv('dk.csv')

    #get x and y 
    x=f[['uvIndex']]
    y=f[['guests']]


    #split into train and test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


    # Model initialization
    regression_model = LinearRegression()
    # Fit the data(train the model)
    regression_model.fit(X_train, y_train)
    
    #Predict form data
    to_predict_array = np.array(to_predict_list)
    to_predict_array=to_predict_array.reshape(-1, 1)
    result = regression_model.predict(to_predict_array)

    return result[0]
    
@app.route('/final',methods=['GET','POST'])
def final():

    data = request.get_json()
    
    print(data['uvindex'])
    uvindex=data['uvindex']
    
    # # #put everything into int
    uvindex=int(uvindex)
    
    to_predict_list = [uvindex]
    
    result = predict(to_predict_list)   
    result=int(result)

    return jsonify({ 
        'result': result
    })
    

if __name__== '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
