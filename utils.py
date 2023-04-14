import numpy as np 
import joblib 


def preprocessdata(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    test_data = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]  
    trained_model = joblib.load("model.pkl") 
    prediction = trained_model.predict(test_data) 

    return prediction 
