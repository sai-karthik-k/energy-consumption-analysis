# Energy Consumption Analysis & Prediction System

## Overview
This system is designed to analyze and predict energy consumption patterns across different buildings, helping identify potential illegal usage of utilities (water, electricity, and gas). It features an interactive web interface for data visualization and machine learning-based predictions.

## Features
- 📊 **Interactive Data Visualization**
  - Time-series analysis of energy consumption
  - Cost analysis by building and city
  - Rate trends visualization
  - Overconsumption analysis

- 🤖 **ML-based Prediction**
  - Illegal consumption detection
  - Separate models for water, electricity, and gas
  - Building-specific predictions
  - Date-based analysis

- 📈 **Data Analysis**
  - Comprehensive consumption metrics
  - Historical trend analysis
  - Cost calculations
  - Performance monitoring

## Technologies Used
- Python 3.8+
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- Joblib

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/energy-consumption-analysis.git
cd energy-consumption-analysis
```

2. Create and activate virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

## Project Structure
```
energy-consumption-analysis/
│ 
│── 1.py                  # ML model training script
│── ECA.py               # Streamlit web application
│── requirements.txt     # Project dependencies
│
│── water_model.joblib
│── electricity_model.joblib
│── gas_model.joblib
│
│── expanded_energy_consumptions_dataset.xlsx
│
└── README.md
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run ECA.py
```

2. Upload your dataset: (Energy Consumptions Dataset.xlsx)
   - Navigate to 'Dataset Upload'
   - Upload Excel file with required sheets:
     * Building Master
     * Energy Consumptions
     * Rates

3. Explore visualizations:
   - Select buildings and energy types
   - View different charts and analyses
   - Filter data by date ranges

4. Make predictions:
   - Choose a date and building
   - Get predictions for all utility types
   - View legal/illegal consumption predictions

## Data Format Requirements

### Excel File Structure
The system expects an Excel file with three sheets:

1. **Building Master**
```
- Building
- City
- Address
- Type
```

2. **Energy Consumptions**
```
- Date
- Building
- Water Consumption
- Expected Water Consumption
- Electricity Consumption
- Expected Electricity Consumption
- Gas Consumption
- Expected Gas Consumption
```

3. **Rates**
```
- Year
- Energy Type
- Price Per Unit
```

## Model Training

To train new models:
```bash
python 1.py
```
This will:
- Process the dataset
- Train Random Forest models
- Save models to the models/ directory
- Generate performance metrics


## Acknowledgments
- Thanks to all contributors who have helped with the project
