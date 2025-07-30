import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from math import remainder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

def safe_log_transform(X):
    return np.log(X + 1e-5)

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ['ratio']

def build_preprocessor():
    ratio_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )

    log_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(safe_log_transform, feature_names_out='one-to-one', validate=False),
        StandardScaler()
    )

    default_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )

    preprocessor = ColumnTransformer([
        ('room_to_poverty_ratio', ratio_pipeline, ['average_number_of_rooms_per_dwelling', 'lower_population']),
        ('distance_to_crime_ratio', ratio_pipeline, ['weighted_distances_to_employment_centers', 'crime_rate']),
        ('pupil_teacher_ratio_to_room_ratio', ratio_pipeline, ['pupil_teacher_ratio', 'average_number_of_rooms_per_dwelling']),
        ('log', log_pipeline, ['weighted_distances_to_employment_centers', 'lower_population', 'room_to_poverty_ratio', 'distance_to_crime_ratio']),
    ], remainder=default_pipeline)

    return preprocessor
