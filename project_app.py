from flask import Flask, request, jsonify
from mysklearn.myclassifiers import MyKNeighborsClassifier
from mysklearn.mypytable import MyPyTable
from mysklearn import myevaluation
import os

app = Flask(__name__)
# http://127.0.0.1:5000/predict?smoking_status=3&bmi=3&heart_disease=1&age=8&avg_glucose_level=2
#age 80s, smokes, bmi 30s, heart disease 1, glucose 2 (130-170)

@app.route("/")
def index():
    return "<h1>Welcome to Nelly & Maya's web app!!</h1>", 200

@app.route("/predict", methods=["GET"])
def predict():
    #parse query string to get instance attribute values from client URL
    #smoking status, BMI, heart disease, average glucose levels, age
    smoking_status = request.args.get("smoking_status", "") #"" is the default value
    bmi = request.args.get("bmi", "")
    heart_disease = request.args.get("heart_disease", "")
    avg_glucose_level = request.args.get("avg_glucose_level", "")
    age = request.args.get("age", "")
    
    print("smoking_status:", smoking_status)
    print("bmi:", bmi)
    print("heart_disease:", heart_disease)
    print("avg_glucose_level:", avg_glucose_level)
    print("age:", age)

    #ORDER: age,heart_disease,avg_glucose_level,bmi,smoking_status,stroke
    prediction = predict_stroke([age,heart_disease,avg_glucose_level,bmi,smoking_status]) #like a row in x_test
    if prediction is not None:
        result = {"Stroke prediction": prediction}
        return jsonify(result), 200
    return "Error making prediction", 400 #bad request = blame the client :)

def predict_stroke(instance):
    knn_clf = MyKNeighborsClassifier()
    stroke_data = MyPyTable()
    stroke_data.load_from_file("input_data/stroke_data_atts_selected.csv")
    X = [inst[:-1] for inst in stroke_data.data]
    y = [inst[-1] for inst in stroke_data.data]
    
    knn_clf.fit(X, y)
    prediction = knn_clf.predict([instance])
    return prediction

if __name__ == "__main__":
    app.run()