# energy-anomaly-detection


##  Project Overview

This project detects **abnormally high energy consumption** in a store using only **outside air enthalpy** and **energy usage** data. The model is trained on historical **normal** data and then applied to test data that may contain anomalies. 

The primary approach is based on **semi-supervised learning** using **Isolation Forest**, with additional logic from a **polynomial baseline** fit to ignore low consumption outliers.

See `Detecting Faulty Energy Consumption.pdf` more details.
---

## How to Run

### Clone repo:
git clone https://github.com/frsth/energy-anomaly-detection.git
cd energy-anomaly-detection


### Create and activate env
conda env create -f environment.yml

conda activate hd-energy

### Run Script with your data
python src/main.py --test_data_csv_path <your_path>
