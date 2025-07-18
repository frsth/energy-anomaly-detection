import pandas as pd
import numpy as np
import argparse
from utils import refine_anomalies
from models.load_model import load_model, load_polyfit

def main(test_data_csv_path):

    # Load test data
    df = pd.read_csv(test_data_csv_path)

    # Extract features and energy
    X = df[['enthalpy', 'energy']]

    # Load trained model and polynomial fit
    model = load_model()
    poly_fit = load_polyfit()

    # Predict
    y_pred = model.predict(X)

    # Post-process with poly fit (remove low energy consumption from anomalies)
    final_anomalies = refine_anomalies(y_pred, np.array(df['enthalpy']), np.array(df['energy']), poly_fit)

    # Save results
    df['prediction'] = final_anomalies

    output_path = "data/processed/test_data_with_prediction.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_csv_path', type=str, required=True, help="Path to test CSV file")
    args = parser.parse_args()

    main(args.test_data_csv_path)