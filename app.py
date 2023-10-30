import numpy as np
import joblib
import streamlit as st


model = joblib.load('./model.joblib')

input = (2022, .57, 30000, 1, 0, 1, 1)


def pred_car_price(input_data):
    np_arr = np.asarray(input).reshape(1, -1)
    return f"Car Price : ${model.predict(np_arr)[0] * 100000}",


def main():
    st.title('Car Prediction System')
    # Year, Present_price, Fuel_Type : 0 -> Petrol, Diesel: 1,  CNG: 2
    # Seller_Type : Dealer : 0, Individual : 1
    # Transmission -> Manual : 0, Automatic : 1
    Year = st.text_input('Year : ')
    Present_price = st.text_input('Present Price : ')
    Fuel_Type = st.text_input('Fuel Type \n Petrol : 0, Diesel: 1, CNG: 2: ')
    SellerType = st.text_input('Seller Type \n Dealer : 0, Individual: 1: ')
    Transmission = st.text_input('Transmission \n Manual: 0, Automatic: 1')
    Kms_Driven = st.text_input('KmsDriven \n')
    Owner = st.text_input('Owner: \n Yes : 1, No : 0')

    # code for prediction
    if (st.button('Car Price Result')):
        data = pred_car_price(
            [Year, Present_price, Kms_Driven, Fuel_Type, SellerType, Transmission, Owner])

    st.success(data)


if __name__ == '__main__':
    main()
