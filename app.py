import pickle

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Load the model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Load an example laptop image
laptop_image = Image.open("lap-top.jpg")  # Replace with the path to your image file

# Title and header image
st.image(laptop_image, use_column_width=True)
st.markdown("<h1 style='text-align: center; color: #ffff;'>Laptop Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #777;'>Predict the price of a laptop based on its configuration</h4>", unsafe_allow_html=True)
st.markdown("***")

# Sidebar for input fields
st.sidebar.header("Laptop Configuration")

# Brand
company = st.sidebar.selectbox('Brand', df['Company'].unique())

# Type of laptop
type = st.sidebar.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.sidebar.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.sidebar.number_input('Weight of the Laptop (in kg)', min_value=0.1, step=0.1, format="%.2f")

# Touchscreen
touchscreen = st.sidebar.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.sidebar.selectbox('IPS', ['No', 'Yes'])

# Screen size
screen_size = st.sidebar.number_input('Screen Size (in inches)', min_value=0.1, step=0.1, format="%.1f")

# Resolution
resolution = st.sidebar.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
    '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# CPU
cpu = st.sidebar.selectbox('CPU', df['Cpu brand'].unique())

# HDD
hdd = st.sidebar.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.sidebar.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.sidebar.selectbox('GPU', df['Gpu brand'].unique())

# OS
os = st.sidebar.selectbox('OS', df['os'].unique())

# Predict button
if st.sidebar.button('Predict Price'):
    # Prepare the query
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    # Avoid division by zero
    if screen_size > 0:
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
    else:
        st.error("Screen size must be greater than 0.")
        ppi = 0

    query = pd.DataFrame({
        'Company': [company],
        'TypeName': [type],
        'Ram': [ram],
        'Weight': [weight],
        'Touchscreen': [touchscreen],
        'Ips': [ips],
        'ppi': [ppi],
        'Cpu brand': [cpu],
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu brand': [gpu],
        'os': [os]
    })

    # Check if all columns are present
    expected_columns = ['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os']
    missing_columns = set(expected_columns) - set(query.columns)
    if missing_columns:
        st.error(f"An error occurred: columns are missing: {missing_columns}")
    else:
        # Predict the price
        try:
            predicted_price = int(np.exp(pipe.predict(query)[0]))
            st.markdown(f"<h2 style='text-align: center; color: #555;'>The predicted price of this configuration is Rs: {predicted_price}/-</h2>", unsafe_allow_html=True)

            # Display configuration details
            st.markdown("***")
            st.markdown("<h3 style='text-align: center; color: #555;'>Configuration Details</h3>", unsafe_allow_html=True)
            config_details = f"""
            <div style="display: flex; justify-content: center;">
                <table style="border-collapse: collapse; width: 70%; margin: auto;">
                    <tr style="border-bottom: 1px solid #ddd;">
                        <th style="text-align: left; padding: 8px;">Feature</th>
                        <th style="text-align: left; padding: 8px;">Details</th>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px;">Brand</td>
                        <td style="padding: 8px;">{company}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px;">Type</td>
                        <td style="padding: 8px;">{type}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px;">RAM</td>
                        <td style="padding: 8px;">{ram} GB</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px;">Weight</td>
                        <td style="padding: 8px;">{weight} kg</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px;">Touchscreen</td>
                        <td style="padding: 8px;">{'Yes' if touchscreen else 'No'}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px;">IPS</td>
                        <td style="padding: 8px;">{'Yes' if ips else 'No'}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px;">Screen Size</td>
                        <td style="padding: 8px;">{screen_size} inches</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px;">Resolution</td>
                        <td style="padding: 8px;">{resolution}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px;">CPU</td>
                        <td style="padding: 8px;">{cpu}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px;">HDD</td>
                        <td style="padding: 8px;">{hdd} GB</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px;">SSD</td>
                        <td style="padding: 8px;">{ssd} GB</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px;">GPU</td>
                        <td style="padding: 8px;">{gpu}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px;">OS</td>
                        <td style="padding: 8px;">{os}</td>
                    </tr>
                </table>
            </div>
            """
            st.markdown(config_details, unsafe_allow_html=True)


        except Exception as e:
            st.error(f"An error occurred: {e}")
