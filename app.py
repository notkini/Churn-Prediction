# 1 is female 0 is male
#1 is yes 0 is no
# scaler is imported as scaler.pkl
# model is imported as model.pkl
# order of the x ==>age,gender,tenure,monthlycharges

import streamlit as st
import joblib
import numpy as np

scaler=joblib.load("scaler.pkl")
model=joblib.load("model.pkl")

st.title("Churn Prediction App")
st.divider()
st.write("Please enter the values and hit the predict button to get the prediction.")
st.divider()
age=st.number_input("ENter age",min_value=10,max_value=100,value=30)


tenure=st.number_input('Enter tenure',min_value=0,max_value=100,value=10)
monthlycharge=st.number_input('Enter monthly charged',min_value=30,max_value=150)
gender=st.selectbox("ENter the gender",["Male","Female"])
st.divider()
predictbutton=st.button("Predict")
st.divider()
if predictbutton:

    gender_selection=1 if gender=="Female" else 0
    x=[age,gender_selection,tenure,monthlycharge]
    x1=np.array(x)
    x_array=scaler.transform([x1])
    prediction=model.predict(x_array)[0]
    predicted="Yes" if prediction==1 else "No"
    st.balloons()
    st.write(f"The prediction is: {predicted}")
    st.write("Thank you for using the app!")





else:
    st.write("please enter the values and hit the predict button to get the prediction.")



