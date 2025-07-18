import pickle
import numpy as np



model_path = "C:\\Users\\fsthilaire\\Desktop\\hd_energy_test\\src\\models\\isolation_forest.pkl"
polyfit_path = "C:\\Users\\fsthilaire\\Desktop\\hd_energy_test\\src\\models\\poly_coeffs.npy"

def load_model(path=model_path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_polyfit(path = polyfit_path):
    coefs = np.load(path)
    return np.poly1d(coefs)