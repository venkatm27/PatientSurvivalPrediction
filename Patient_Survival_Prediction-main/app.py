import gradio
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Response
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

save_file_name = "xgboost-model.pkl"
model = joblib.load(save_file_name)


# FastAPI object
app = FastAPI()


# ################################# Prometheus related code START ######################################################
import prometheus_client as prom

acc_metric = prom.Gauge('Patient_Survival_accuracy_score', 'Accuracy score for few random 100 test samples')


# Load dataset
# df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
# X = df.iloc[:, :-1].values
# y = df['DEATH_EVENT'].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state= 123)
# print(y_test)
# test_data = X_test.copy()
# test_data['target'] = y_test

# Accuracy


# Function for updating metrics
r2_metric = prom.Gauge('patient_survial_r2_score', 'R2 score for random 100 test samples')
def update_metrics():
    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    df = df.sample(10)
    data_feat = df.drop('DEATH_EVENT', axis=1)
    data_target = df['DEATH_EVENT'].values
    r2 = r2_score(data_target, model.predict(data_feat))
    r2_metric.set(r2)
    

@app.get("/metrics")
async def get_metrics():
    update_metrics()
    return Response(media_type="text/plain", content= prom.generate_latest())
################################# Prometheus related code END ######################################################


# Function for prediction
def predict_death_event(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, 
                        high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame([{
        "age": age,
        "anaemia": anaemia,
        "creatinine_phosphokinase": creatinine_phosphokinase,
        "diabetes": diabetes,
        "ejection_fraction": ejection_fraction,
        "high_blood_pressure": high_blood_pressure,
        "platelets": platelets,
        "serum_creatinine": serum_creatinine,
        "serum_sodium": serum_sodium,
        "sex": sex,
        "smoking": smoking,
        "time": time
    }])
    # Predict using the trained model
    prediction = model.predict(input_data)
    
    # Optional: Return a human-readable result
    return "Death Event" if prediction[0] == 1 else "No Death Event"

# Gradio interface to generate UI link
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gradio.Interface(fn = predict_death_event,
                        inputs=[
                                    gradio.Number(label="Age"),
                                    gradio.Radio([0, 1], label="Anaemia (0=No, 1=Yes)"),
                                    gradio.Number(label="Creatinine Phosphokinase"),
                                    gradio.Radio([0, 1], label="Diabetes (0=No, 1=Yes)"),
                                    gradio.Number(label="Ejection Fraction"),
                                    gradio.Radio([0, 1], label="High Blood Pressure (0=No, 1=Yes)"),
                                    gradio.Number(label="Platelets"),
                                    gradio.Number(label="Serum Creatinine"),
                                    gradio.Number(label="Serum Sodium"),
                                    gradio.Radio([0, 1], label="Sex (0=Female, 1=Male)"),
                                    gradio.Radio([0, 1], label="Smoking (0=No, 1=Yes)"),
                                    gradio.Number(label="Follow-up Time")
                                ],
                        outputs = gradio.Textbox(type="text", label='Prediction', elem_id="out_textbox"),
                         title = title,
                         description = description,
                         allow_flagging='never'
                        )

# Mount gradio interface object on FastAPI app at endpoint = '/'
app = gradio.mount_gradio_app(app, iface, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)