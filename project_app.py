from flask import Flask, request, jsonify
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

    #TODO fix hardcoding
    prediction = predict_stroke([smoking_status, bmi, heart_disease, avg_glucose_level, age]) #like a row in x_test
    #if anything goes wrong, return None
    if prediction is not None:
        result = {"Stroke prediction": prediction}
        return jsonify(result), 200
    return "Error making prediction", 400 #bad request = blame the client :)

def predict_stroke(instance):
    return 1.0 #HARDCODED TODO FIX

if __name__ == "__main__":
    app.run()