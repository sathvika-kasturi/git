import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
## load the model
model=tf.keras.models.load_model('model.h5')
##load the encoders and scalers
with open('label_encoded.pkl','rb') as file:
    label_encoded=pickle.load(file)
with open('Hot_encoded.pkl','rb') as file:
    Hot_encoded=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)
    
##stramlit app
st.title('Customer churn prediction')
## user input
geography=st.selectbox('Geography', Hot_encoded.categories_[0])
gender=st.selectbox('Gender',label_encoded.classes_)
age=st.slider('Age',18, 92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_number=st.selectbox('Is Active Member',[0,1])
## prepare input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[gender],
    'Geography':[geography],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_number],
    'EstimatedSalary':[estimated_salary],
    
})
input_data['Gender']=label_encoded.transform(input_data['Gender'])

##one-hot encoded 'Geography'
one_encoded_geo=Hot_encoded.transform(input_data[['Geography']])
features=Hot_encoded.get_feature_names_out(['Geography'])
one_encoded_geo_df=pd.DataFrame(one_encoded_geo.toarray(),columns=features)
## combine onehot encoded columnx with input data
input_data=pd.concat([input_data.drop(['Geography'],axis=1),one_encoded_geo_df],axis=1)
## scale the input data
input_data_scaled=scaler.transform(input_data)
#prediction
prediction=model.predict(input_data_scaled)
prediction_prob=prediction[0][0]
st.write(f'Churn probability:{prediction_prob:.2f}')

if prediction_prob>0.5:
    print("the customer is likely to churn")
    
else:
    print("the customer is not likely to churn")