# import the necessary packages
import numpy as np
from flask import Flask, render_template, request
import uuid
import time
import json
import io
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


# initialize our Flask application and Redis server
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}

@app.route('/uploads', methods=['GET', 'POST'])
def uploadFile():
    if request.method == "POST": 
        file = request.files.get("file") 
        file_content = file.read() 
          
        # check if file loaded successfully or not 
        if file_content: 
            print("UPLOAD", file_content)
            return "Uploaded Successful"
        else: 
            return "Uploaded Unsuccessful"
  
    return render_template("index.html") 

@app.route("/")
def homepage():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def handle_predict():
    normal=pd.read_csv('normal.csv')
    LL1=pd.read_csv('LL1.csv')
    LL2=pd.read_csv('LL2.csv')
    Str1=pd.read_csv('Str1_3-Str2_2-LL1.csv')
    warning_data_wec=pd.read_csv('Partial_shading_edit.csv')
    total=pd.concat([normal,LL1,LL2,Str1,warning_data_wec])

    # Extract data field from file
    features=['V','I','P','G']
    x=total[features]
    y1=total['fault']
    y2=total['no_module_fault']
    y3=total['partial_shading']

    v = request.form.get("v")
    i = request.form.get("i") 
    p = request.form.get("p") 
    g = request.form.get("g") 
    predict_type = request.form.get("predict_type") 
    print("LOG", v, i, p, g, predict_type)

    re_data = [v, i, p, g]
    # Convert the list to a numpy array and reshape it to match the training data shape
    input_data = np.array(re_data).reshape(1, -1)
    x1train,x1test,y1train,y1test=train_test_split(x,y2,test_size=0.2,random_state=50)

    if predict_type == "DECISION":
        model1= DecisionTreeClassifier()
        model1.fit(x1train,y1train)
        prediction = model1.predict(input_data)
        if prediction == 0:
            prediction = "Normal State "
            print(prediction)
        elif prediction == 1:
            prediction= "Short circuit fault on one panel state"
            print(prediction)
        elif prediction ==2:
            prediction = "Short circuit fault on two panels state"
            print( prediction)
        elif prediction ==3:
            prediction = "Partial shading state"
            print(prediction)
        return render_template("index.html", result=prediction)
    elif predict_type == "RandomForest":
        model2 = RandomForestClassifier()
        model2.fit(x1train, y1train)
        prediction = model2.predict(input_data)
        if prediction == 0:
            prediction = "Normal State "
            print(prediction)
        elif prediction == 1:
            prediction= "Short circuit fault on one panel state"
            print(prediction)
        elif prediction ==2:
            prediction = "Short circuit fault on two panels state"
            print( prediction)
        elif prediction ==3:
            prediction = "Partial shading state"
            print(prediction)
        return render_template("index.html", result=prediction)
    elif predict_type == "KNN":
        model3 = KNeighborsClassifier()
        model3.fit(x1train, y1train)
        prediction = model3.predict(input_data)
        if prediction == 0:
            prediction = "Normal State "
            print(prediction)
        elif prediction == 1:
            prediction= "Short circuit fault on one panel state"
            print(prediction)
        elif prediction ==2:
            prediction = "Short circuit fault on two panels state"
            print( prediction)
        elif prediction ==3:
            prediction = "Partial shading state"
            print(prediction)
        return render_template("index.html", result=prediction)
    elif predict_type == "GA":
        from genetic_selection import GeneticSelectionCV
        selector = GeneticSelectionCV(
        estimator=model1,
        cv=5,
        verbose=1,
        scoring="accuracy",
        max_features=4,
        n_population=200,
        crossover_proba=0.5,
        mutation_proba=0.2,
        n_generations=40,
        crossover_independent_proba=0.5,
        mutation_independent_proba=0.05,
        tournament_size=3,
        n_gen_no_change=10,
        caching=True,
        n_jobs=-1,
    )
        model4 = selector.fit(x1train, y1train)
        prediction = model4.predict(input_data)
        if prediction == 0:
            prediction = "Normal State "
            print(prediction)
        elif prediction == 1:
            prediction= "Short circuit fault on one panel state"
            print(prediction)
        elif prediction ==2:
            prediction = "Short circuit fault on two panels state"
            print( prediction)
        elif prediction ==3:
            prediction = "Partial shading state"
            print(prediction)
        return render_template("index.html", result=prediction)



    
# for debugging purposes, it's helpful to start the Flask testing
# server (don't use this for production
if __name__ == "__main__":
	print("* Starting web service...")
	app.run()
