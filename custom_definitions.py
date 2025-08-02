

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations
from sklearn.pipeline import make_pipeline

import re


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['Cement'] = df['Cement_component_1kg_in_a_m3_mixture'].replace(0, 0.0001)
        df['Water'] = df['Water_component_4kg_in_a_m3_mixture'].replace(0, 0.0001)
        df['Superplasticizer'] = df['Superplasticizer_component_5kg_in_a_m3_mixture'].replace(0, 0.0001)

        df['water_cement_ratio'] = df['Water'] / df['Cement']
        df['total_binder'] = (
            df['Cement_component_1kg_in_a_m3_mixture'] +
            df['Blast_Furnace_Slag_component_2kg_in_a_m3_mixture'] +
            df['Fly_Ash_component_3kg_in_a_m3_mixture']
        )
        df['total_aggregate'] = (
            df['Coarse_Aggregate_component_6kg_in_a_m3_mixture'] +
            df['Fine_Aggregate_component_7kg_in_a_m3_mixture']
        )
        df['plasticizer_per_cement'] = df['Superplasticizer'] / df['Cement']
        df['cement_agg_ratio'] = df['Cement'] / df['total_aggregate']
        df['water_binder_ratio'] = df['Water'] / df['total_binder']
        df['mix_density'] = (
            df['Cement_component_1kg_in_a_m3_mixture'] +
            df['Blast_Furnace_Slag_component_2kg_in_a_m3_mixture'] +
            df['Fly_Ash_component_3kg_in_a_m3_mixture'] +
            df['Water'] +
            df['Superplasticizer'] +
            df['Coarse_Aggregate_component_6kg_in_a_m3_mixture'] +
            df['Fine_Aggregate_component_7kg_in_a_m3_mixture']
        )
        return df

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.skewed_cols = [
            'Age_day',
            'water_cement_ratio',
            'Superplasticizer_component_5kg_in_a_m3_mixture',
            'plasticizer_per_cement',
            'Blast_Furnace_Slag_component_2kg_in_a_m3_mixture',
            'cement_agg_ratio',
            'Fly_Ash_component_3kg_in_a_m3_mixture',
            'Cement_component_1kg_in_a_m3_mixture'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.skewed_cols:
            if col in X.columns:
                if X[col].min() > 0:
                    X_transformed[col] = np.log1p(X[col])
                else:
                    X_transformed[col] = np.log1p(X[col] - X[col].min() + 1)
        return X_transformed

class LowercaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in X_copy.columns:
            if X_copy[col].dtype == 'object' or X_copy[col].dtype == 'string':
                X_copy[col] = X_copy[col].astype(str).str.lower()
        return X_copy


class RemoveUnusualChars(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, pattern=r'[^a-zA-Z0-9\s]'):
        self.columns = columns
        self.pattern = pattern

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        columns_to_clean = self.columns or X_copy.select_dtypes(include=['object', 'string']).columns

        for col in columns_to_clean:
            X_copy[col] = X_copy[col].astype(str).apply(lambda x: re.sub(self.pattern, '', x))

        return X_copy



class PipelineWithLabelDecoder:
    def __init__(self, pipeline, label_encoder):
        self.pipeline = pipeline
        self.label_encoder = label_encoder

    def fit(self, X, y):
        y_encoded = self.label_encoder.fit_transform(y)
        self.pipeline.fit(X, y_encoded)
        return self

    def predict(self, X):
        y_encoded_pred = self.pipeline.predict(X)
        return self.label_encoder.inverse_transform(y_encoded_pred)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

class Leela_Venkata_Sai_Nerella(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = self.featuressplit(X)
        X = self.remove(X)
        return X

    def featuressplit(self, X):
        feature1, feature6, feature8 = [], [], []
        feature10, feature15, feature21 = [], [], []

        for i in range(len(X)):
            # Last column: feature_1 and feature_6
            cell = X.iloc[i, -1]
            a, b = self.safe_split(cell)
            feature1.append(a)
            feature6.append(b)

            # Second last column: feature_21 and feature_10
            cell = X.iloc[i, -2]
            a, b = self.safe_split(cell)
            feature21.append(a)
            feature10.append(b)

            # Third last column: feature_8 and feature_15
            cell = X.iloc[i, -3]
            a, b = self.safe_split(cell)
            feature8.append(a)
            feature15.append(b)

        X["feature_1"] = feature1
        X["feature_6"] = feature6
        X["feature_8"] = feature8
        X["feature_10"] = feature10
        X["feature_15"] = feature15
        X["feature_21"] = feature21

        # Drop last 3 columns (which were split)
        X = X.drop(['feature_8,feature_15', 'feature_21,feature_10', 'feature_1,feature_6'], axis=1)

        return X

    

    def safe_split(self, cell):
        if pd.isna(cell):
            return np.nan, np.nan
        try:
            a, b = str(cell).split(",", 1)
            return a, b
        except:
            return np.nan, np.nan

    def remove(self, X):
        for col in X.columns:
            X[col] = X[col].astype(str).apply(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x))
            X[col] = pd.to_numeric(X[col], errors='ignore')  
        return X



def split_columns(df):
    df = df.copy()
    for col in df.columns:
        if ',' in col:
            parts = col.split(',')
            df[parts[0]] = df[col].str[0]
            df[parts[1]] = df[col].str[1]
            df.drop(columns=col, inplace=True)
    return df


def fill_missing(df):

    return df.fillna('')
    
def split_bycomma(X_df):
    X = X_df.copy()
    X[['feature_8', 'feature_15']] = X['feature_8,feature_15'].str.split(',', expand=True)
    X[['feature_21', 'feature_10']] = X['feature_21,feature_10'].str.split(',', expand=True)
    X[['feature_1', 'feature_6']] = X['feature_1,feature_6'].str.split(',', expand=True)
    X.drop(['feature_8,feature_15', 'feature_21,feature_10', 'feature_1,feature_6'], axis=1, inplace=True)
    return X
fillna_transformer = FunctionTransformer(fill_missing, validate=False)


def replacer(X):
   X_temp=X.copy()
   for col in X.columns:
      X_temp[f"{col}_letters"] = X_temp[col].str.replace(r'[^a-zA-Z]', "", regex=True)
      X_temp[f"{col}_symbols"] = X_temp[col].str.replace(r'[a-zA-Z0-9]', "", regex=True)
      X_temp.drop(columns=[col],inplace=True)
    
 
   return X_temp

custom_replacer=FunctionTransformer(replacer,validate=False)



class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, top_features=None):
        self.top_features = top_features if top_features else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[['feature_8', 'feature_15']] = X['feature_8,feature_15'].str.split(",", expand=True)
        X[['feature_21', 'feature_10']] = X['feature_21,feature_10'].str.split(",", expand=True)
        X[['feature_1', 'feature_6']] = X['feature_1,feature_6'].str.split(",", expand=True)
        X.drop(['feature_8,feature_15', 'feature_21,feature_10', 'feature_1,feature_6'], axis=1, inplace=True)

        string_cols = X.select_dtypes(include='object').columns
        for col in string_cols:
            X[col] = X[col].str.lower().str.strip()

        return X

class FeatureCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, top_features=None):
        self.top_features = top_features if top_features else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for f1, f2 in combinations(self.top_features, 2):
            new_col = f"{f1}_{f2}_comb"
            X[new_col] = X[f1].astype(str) + "_" + X[f2].astype(str)

        X["count_U"] = (X[self.top_features] == "U").sum(axis=1)
        X["unique_top_cats"] = X[self.top_features].nunique(axis=1)
        return X

class ImputerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = SimpleImputer(strategy="constant")

    def fit(self, X, y=None):
        self.imputer.fit(X)
        self.columns = X.columns
        return self

    def transform(self, X):
        return pd.DataFrame(self.imputer.transform(X), columns=self.columns, index=X.index)


class OneHotEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, X, y=None):
        self.obj_cols = X.select_dtypes(include="object").columns
        self.encoder.fit(X[self.obj_cols])
        return self

    def transform(self, X):
        X = X.copy()
        encoded = self.encoder.transform(X[self.obj_cols])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.obj_cols), index=X.index)
        return pd.concat([X.drop(columns=self.obj_cols), encoded_df], axis=1)


class KMeansClusterWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    def fit(self, X, y=None):
        self.kmeans.fit(X)
        return self

    def transform(self, X):
        cluster_labels = self.kmeans.predict(X)
        return pd.concat([X.reset_index(drop=True), pd.Series(cluster_labels, name="cluster_label")], axis=1)


class ScalerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)


# --- Full Preprocessing Pipeline ---
def create_pipeline():
    top_features = ["feature_18", "feature_1"]
    pipeline = Pipeline([
        ("custom_feature_engineering", CustomFeatureEngineer(top_features=top_features)),
        ("feature_combination", FeatureCombiner(top_features=top_features)),
        ("imputation", ImputerWrapper()),
        ("onehot_encode", OneHotEncoderWrapper()),
        ("clustering", KMeansClusterWrapper()),
        ("scaling", ScalerWrapper())
    ])

    return pipeline


    
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import pickle


class InfValueHandler(BaseEstimator, TransformerMixin):
    """Replace inf and -inf values with NaN"""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)

class LeakyFeatureRemover(BaseEstimator, TransformerMixin):
    """Drops predefined leaky features if they exist"""
    def __init__(self, target_col):
        self.target_col = target_col
        self.leaky_features = [
            'compreesive_strength_mpa','compressive_strength_duplicate',
            'compressive_strength_mpas'
        ]
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(
            columns=[f for f in self.leaky_features if f in X.columns],
            errors='ignore'
        )

class DerivedFeatureAdder(BaseEstimator, TransformerMixin):
    """Adds cement_water_ratio and agg_cement_ratio features"""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        if 'cement' in X.columns and 'water' in X.columns:
            X['cement_water_ratio'] = X['cement'] / (X['water'] + 1)
        if 'fine_aggregate' in X.columns and 'coarse_aggregate' in X.columns and 'cement' in X.columns:
            X['agg_cement_ratio'] = (X['fine_aggregate'] + X['coarse_aggregate']) / (X['cement'] + 1)
        return X



possible_targets = [c for c in df.columns if 'strength' in c.lower()]
if possible_targets:
    target_col = possible_targets[0]
else:
    raise ValueError("No target column found containing 'strength'.")

print(f" Using target column: {target_col}")
numeric_cols = df.drop(columns=[target_col], errors='ignore').select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.drop(columns=[target_col], errors='ignore').select_dtypes(include=['object']).columns
boolean_cols = df.drop(columns=[target_col], errors='ignore').select_dtypes(include=['bool']).columns



numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

boolean_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols),
    ('bool', boolean_transformer, boolean_cols)
])


final_pipeline = Pipeline([
    ('handle_inf', InfValueHandler()),
    ('drop_leaky', LeakyFeatureRemover(target_col=target_col)),
    ('add_features', DerivedFeatureAdder()),
    ('preprocessor', preprocessor),('model', LinearRegression())
])


X_full = df.drop(columns=[target_col], errors='ignore')
y_full = df[target_col]

final_pipeline.fit(X_full, y_full)

with open("dev_pathak.pkl", "wb") as f:
    pickle.dump(final_pipeline, f)

print("Pipeline created, trained, and saved successfully!")



