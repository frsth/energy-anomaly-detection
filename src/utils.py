import numpy as np

def refine_anomalies(anomaly_pred, enthalpy, energy, poly_fit):
    """
    Only keep anomalies where the actual energy consumption is 
    significantly higher than expected.
    
    Parameters:
    - anomaly_pred: array of initial anomaly predictions from the model (-1 or 1)
    - enthalpy: array of enthalpy for each prediction
    - energy: array of energy consumption for each prediction
    - poly_fit: polynomial fit object
    
    Returns:
    - final_anomalies: array of refined anomaly flags (-1 = anomaly, 1 = normal)
    """
    final_anomalies = anomaly_pred.copy()
    poly_fit_energy = poly_fit(enthalpy)
    final_anomalies[energy < poly_fit_energy] = 1

    return final_anomalies