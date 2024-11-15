import streamlit as st
import pandas as pd
import plotly.express as px
from joblib import load
from sklearn.preprocessing import LabelEncoder
import time
from streamlit_extras.let_it_rain import rain

# --- Add Custom Styling ---
st.markdown(
    """
    <style>
    .st-header1{
        background-color: rgb(255, 75, 75);
        padding: 4px;
        border-radius: 8px;
        color: white;
        font-size: 20px;
        margin: 4px;
    }
    .st-header {
        background-color: #4CBB17;
        padding: 4px;
        border-radius: 8px;
        color: white;
        font-size: 20px;
        margin: 4px;
    }
    .st-sidebar {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .css-1wa3eu0-placeholder {
        color: #4CBB17;
    }
    body, .st-button > button {
        font-family: "Verdana", sans-serif;
    }
    .stProgress > div > div > div {
        background-color: #4CBB17;
    }
    .selected-button {
        background-color: #4CBB17 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- App Title ---
st.title("Energy Consumption Analysis & Prediction")
st.write("Please navigate to 'Dataset Upload' to upload your Excel file for visualizing or predicting energy consumption.")

# --- Main Navigation Menu ---
if "menu_option" not in st.session_state:
    st.session_state.menu_option = "Dataset Upload"

# Button navigation
if st.sidebar.button("Dataset Upload", key="dataset_button"):
    st.session_state.menu_option = "Dataset Upload"
if st.sidebar.button("Data Visualization", key="visualization_button"):
    st.session_state.menu_option = "Data Visualization"
if st.sidebar.button("ML Prediction", key="prediction_button"):
    st.session_state.menu_option = "ML Prediction"

# --- Page: Dataset Upload ---
if st.session_state.menu_option == "Dataset Upload":
    st.markdown('<div class="st-header1">📁 Dataset Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    
    if uploaded_file:
        # Store uploaded file in session state
        st.session_state['uploaded_file'] = pd.ExcelFile(uploaded_file)
        st.success("File uploaded successfully! You can now proceed to Data Visualization or ML Prediction.")
    
# Check if file is uploaded
if 'uploaded_file' in st.session_state:
    excel_data = st.session_state['uploaded_file']
    building_master_df = excel_data.parse('Building Master')
    energy_consumptions_df = excel_data.parse('Energy Consumptions')
    rates_df = excel_data.parse('Rates')
    
    # --- Page: Data Visualization ---
    if st.session_state.menu_option == "Data Visualization":
        st.markdown('<div class="st-header1">📊 Energy Consumption Data Visualization</div>', unsafe_allow_html=True)
        
        # Sidebar filter options
        with st.sidebar:
            building_filter = st.multiselect("Select Building(s):", options=building_master_df['Building'].unique(),
                                             default=building_master_df['Building'].unique())
            energy_type = st.selectbox("Select Energy Type:", ["Water", "Electricity", "Gas"])
        
        # Filter and preprocess data
        energy_consumptions_df['Date'] = pd.to_datetime(energy_consumptions_df['Date'])
        data = energy_consumptions_df.merge(building_master_df, on='Building', how='left')
        data = data[data['Building'].isin(building_filter)]
        
        # Map rates to data
        for energy, rate_dict in zip(["Water", "Electricity", "Gas"],
                                     [rates_df[rates_df['Energy Type'] == 'Water'].set_index('Year')['Price Per Unit'].to_dict(),
                                      rates_df[rates_df['Energy Type'] == 'Electricity'].set_index('Year')['Price Per Unit'].to_dict(),
                                      rates_df[rates_df['Energy Type'] == 'Gas'].set_index('Year')['Price Per Unit'].to_dict()]):
            data[f'{energy} Rate'] = data['Date'].dt.year.map(rate_dict)
            data[f'Overconsumed {energy}'] = (data[f'{energy} Consumption'] - data[f'Expected {energy} Consumption']).clip(lower=0)
            data[f'Total Cost {energy}'] = data[f'{energy} Consumption'] * data[f'{energy} Rate']
            data[f'Total Cost Overconsumed {energy}'] = data[f'Overconsumed {energy}'] * data[f'{energy} Rate']

        # Visualization sections with Plotly
        color_map = {"Water": "#1f77b4", "Electricity": "#ff7f0e", "Gas": "#2ca02c"}

        # Units Consumed by Date
        st.markdown(f"<div class='st-header'>📈 Units Consumed by Date - {energy_type}</div>", unsafe_allow_html=True)
        fig = px.line(data, x='Date', y=f'{energy_type} Consumption', labels={'y': f'{energy_type} Consumption'},
                      color_discrete_sequence=[color_map[energy_type]])
        st.plotly_chart(fig)

        # Energy Rates Over Time
        st.markdown(f"<div class='st-header'>💹 Energy Rates Over Time - {energy_type}</div>", unsafe_allow_html=True)
        fig = px.line(data, x='Date', y=f'{energy_type} Rate', labels={'y': f'{energy_type} Rate'},
                      color_discrete_sequence=[color_map[energy_type]])
        st.plotly_chart(fig)

        # Total Cost by Date
        st.markdown(f"<div class='st-header'>💰 Total Cost by Date - {energy_type}</div>", unsafe_allow_html=True)
        fig = px.line(data, x='Date', y=f'Total Cost {energy_type}', labels={'y': f'Total Cost {energy_type}'},
                      color_discrete_sequence=[color_map[energy_type]])
        st.plotly_chart(fig)

        # Total Cost of Overconsumed Units by Date
        st.markdown(f"<div class='st-header'>🔴 Total Cost of Overconsumed Units by Date - {energy_type}</div>", unsafe_allow_html=True)
        fig = px.line(data, x='Date', y=f'Total Cost Overconsumed {energy_type}', labels={'y': f'Total Cost Overconsumed {energy_type}'},
                      color_discrete_sequence=[color_map[energy_type]])
        st.plotly_chart(fig)

        # Overconsumed Units by Building
        st.markdown(f"<div class='st-header'>🏢 Overconsumed {energy_type} Units by Building</div>", unsafe_allow_html=True)
        overconsumed_by_building = data.groupby('Building')[f'Overconsumed {energy_type}'].sum().reset_index()
        fig = px.bar(overconsumed_by_building, x='Building', y=f'Overconsumed {energy_type}',
                     color_discrete_sequence=[color_map[energy_type]])
        st.plotly_chart(fig)

        # Total Cost by City
        st.markdown(f"<div class='st-header'>🌆 Total Cost by City - {energy_type}</div>", unsafe_allow_html=True)
        total_cost_by_city = data.groupby('City')[f'Total Cost {energy_type}'].sum().reset_index()
        fig = px.bar(total_cost_by_city, x='City', y=f'Total Cost {energy_type}',
                     color_discrete_sequence=[color_map[energy_type]])
        st.plotly_chart(fig)

    # --- Page: ML Prediction ---
    elif st.session_state.menu_option == "ML Prediction":
        st.markdown('<div class="st-header1">⚙️ Energy Consumption Prediction</div>', unsafe_allow_html=True)
        st.write("This app predicts whether energy consumption is legal or illegal based on historical data.")

        # Encode 'Building' column
        le_building = LabelEncoder()
        energy_consumptions_df['Building_Encoded'] = le_building.fit_transform(energy_consumptions_df['Building'])

        # Load models
        water_model = load('water_model.joblib')
        electricity_model = load('electricity_model.joblib')
        gas_model = load('gas_model.joblib')

        # Prediction input section
        input_date = st.date_input("Select a date for prediction")
        input_building = st.selectbox("Select a building", options=energy_consumptions_df['Building'].unique())

        # Perform prediction when button clicked
        if st.button("Predict"):
            input_building_encoded = le_building.transform([input_building])[0]
            input_data = pd.DataFrame({
                'Building_Encoded': [input_building_encoded],
                'Year': [input_date.year],
                'Month': [input_date.month],
                'Day': [input_date.day],
                'DayOfWeek': [input_date.weekday()],
                'Quarter': [(input_date.month - 1) // 3 + 1],
                'IsWeekend': [int(input_date.weekday() >= 5)],
                'Expected Electricity Consumption': [energy_consumptions_df['Expected Electricity Consumption'].mean()],
                'Expected Water Consumption': [energy_consumptions_df['Expected Water Consumption'].mean()],
                'Expected Gas Consumption': [energy_consumptions_df['Expected Gas Consumption'].mean()],
                '% Over Expected Electricity': [100],
                '% Over Expected Water': [100],
                '% Over Expected Gas': [100],
            })

            # Model predictions
            water_prediction = water_model.predict(input_data)[0]
            electricity_prediction = electricity_model.predict(input_data)[0]
            gas_prediction = gas_model.predict(input_data)[0]

            # Show predictions
            st.write("### Prediction Results:")
            st.write(f"Water Consumption Prediction: {'Legal' if water_prediction == 0 else 'Illegal'}")
            st.write(f"Electricity Consumption Prediction: {'Legal' if electricity_prediction == 0 else 'Illegal'}")
            st.write(f"Gas Consumption Prediction: {'Legal' if gas_prediction == 0 else 'Illegal'}")
            # time.sleep(1)
            # st.balloons()
            # rain(emoji="⚡", font_size=54, falling_speed=1.5, animation_length="0.01s")
else:
    st.warning("Please upload the dataset in 'Dataset Upload' first!")
