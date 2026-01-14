import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Metadata for Curated Datasets
CURATED_DATASETS = {
    "Titanic": {
        "url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "target": "Survived",
        "drop_cols": ["PassengerId", "Name", "Ticket", "Cabin"] # Drop high cardinality/null columns
    },
    "Heart Disease": {
        "url": "https://raw.githubusercontent.com/kcmillerst/Heart-Diesease-Classification/master/heart.csv",
        "target": "target",
        "drop_cols": []
    },
    "Iris (Flowers)": {
        "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        "target": "species",
        "drop_cols": []
    }
}

def load_data_from_source(source, target_column=None, drop_cols=None, test_size=0.3, random_state=42):
    """
    Loads data from a generic source (CSV file buffer or URL path).
    """
    if source is None:
        return None
        
    try:
        df = pd.read_csv(source)
    except Exception as e:
        raise ValueError(f"Failed to load data: {e}")
    
    # Pre-cleaning: Drop specified columns
    if drop_cols:
        existing_drop_cols = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=existing_drop_cols)

    # Clean basic nulls (fill with mode for categorical, mean for numeric) - simple strategy for demo
    for col in df.columns:
        if df[col].dtype == 'object':
             df[col] = df[col].fillna(df[col].mode()[0])
        else:
             df[col] = df[col].fillna(df[col].mean())

    # User Target Selection Logic
    if target_column is None:
        target_column = df.columns[-1]
    
    if target_column not in df.columns:
         # Fallback mechanism if exact target name match fails (e.g. case sensitivity)
         # Try case-insensitive match
        found = False
        for c in df.columns:
            if c.lower() == target_column.lower():
                target_column = c
                found = True
                break
        if not found:
            raise ValueError(f"Target column '{target_column}' not found in dataset. Available: {list(df.columns)}")

    # Separate X and y
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Preprocessing: Encode Categoricals
    # LightGBM handles categories well, but we need to ensure they are numeric/category dtype
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
    # Process Target if it's categorical
    if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype) or len(np.unique(y)) < 10: # Heuristic for classification
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
