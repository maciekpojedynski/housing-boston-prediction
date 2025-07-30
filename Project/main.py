from data_loader import loading_house_data
from model import build_model


def main():
    import os
    import pandas as pd

    # Load data
    housing = loading_house_data('/Users/maciek/Desktop/GitHub/boston/boston.csv')

    # Rename columns
    column_mapping = {
        'CRIM': 'crime_rate',
        'ZN': 'residential_zone_proportion',
        'INDUS': 'non-retailand_business_acres',
        'CHAS': 'river_dummy',
        'NOX': 'nitric_oxide_concentration',
        'RM': 'average_number_of_rooms_per_dwelling',
        'DIS': 'weighted_distances_to_employment_centers',
        'RAD': 'index_of_accessibility_to_radial_highways',
        'PTRATIO': 'pupil_teacher_ratio',
        'B': 'black_ppl_per',
        'LSTAT': 'lower_population',
        'MEDV': 'median_house_value'
    }
    housing = housing.rename(columns=column_mapping)

    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop("median_house_value", axis=1)


    model = build_model()
    model.fit(housing, housing_labels)

    predictions = model.predict(housing)
    print("Przykładowe predykcje:")
    print(predictions[:5].round(-2))
    print("Rzeczywiste wartości:")
    print(housing_labels.iloc[:5].values)

if __name__ == "__main__":
    main()