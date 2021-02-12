import torch
import torch.nn as nn
import torch.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt

df = pd.read_csv('Data/NYCTaxiFares.csv')


def haversine_distance(df, lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles

    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])

    delta_phi = np.radians(df[lat2] - df[lat1])
    delta_lambda = np.radians(df[lon2] - df[lon1])


    a = np.sin(delta_phi/ 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return c * r


df['dist_km'] = haversine_distance(df, 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude')

print(df.info())

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

