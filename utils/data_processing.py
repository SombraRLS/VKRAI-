import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_and_preprocess(file):
    df = pd.read_csv(file)

    # Удаляем ненужные колонки, если есть
    columns_to_drop = ['RecordID', 'SerialNumber']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Проверяем наличие столбцов для определения типа отказа
    failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    if all(col in df.columns for col in failure_cols):
        # Пример: выбираем тип отказа по максимальному значению среди указанных столбцов
        df['Failure Type'] = df[failure_cols].idxmax(axis=1)
    else:
        df['Failure Type'] = 'Unknown'

    return df

def determine_failure_type(row):
    """Определение типа отказа на основе булевых столбцов."""
    # Здесь предполагается, что в данных есть эти колонки
    if row.get('MechanicalFailure', 0) == 1:
        return 1  # Mechanical
    elif row.get('ElectricalFailure', 0) == 1:
        return 2  # Electrical
    elif row.get('ThermalFailure', 0) == 1:
        return 3  # Thermal
    elif row.get('OtherFailure', 0) == 1:
        return 4  # Other
    return 0  # No failure

def create_preprocessor():
    """Создаём ColumnTransformer для числовых и категориальных признаков."""
    numeric_features = ['Temperature', 'Vibration', 'Pressure', 'Runtime', 'Maintenance']
    categorical_features = ['Type']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    return preprocessor

def preprocess_input(input_df, preprocessor):
    """Предобработка входных данных с использованием уже обученного трансформера."""
    return preprocessor.transform(input_df)
