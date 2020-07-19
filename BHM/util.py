import json
import pickle
import numpy as np
import os

dir = os.path.dirname(__file__)

__locations = None
__data_columns = None
__model = None


def get_estimated_price(location, sqft, bhk, bath):

    try:
        loc_index = __data_columns.index(location.lower())

    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)


def get_location_names():
    return __locations


def load_saved_artifacts():
    print("loading saved artifacts...start")

    global __data_columns
    global __locations
    global __model

    filename = os.path.join(dir, "artifacts/columns.json")

    with open(filename, 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    filename = os.path.join(dir, "artifacts/banglore_home_price_model.pickle")

    with open(filename, 'rb') as f:
        __model = pickle.load(f)

    print("loading saved artifacts...done")


load_saved_artifacts()
if __name__ == '__main__':

    print(get_location_names())
    # print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    # print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    # print(get_estimated_price('Kalhalli', 1000, 2, 2))
    # print(get_estimated_price('Ejipura', 1000, 2, 2))
