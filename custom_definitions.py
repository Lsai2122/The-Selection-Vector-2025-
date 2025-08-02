import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import FunctionTransformer , LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline as ImbPipeline   
from imblearn.over_sampling import SMOTE   
from sklearn.linear_model import LogisticRegression   
from splitter import split_columns
import imblearn
import re


#==============Your Name======================
# Your code
#==============Your Name=====================
#Dont remove the following snippet and follow the same

#==========Arjun E Naik==============
class YesNoConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        mapping = {'yes': 1, 'y': 1, 'no': 0, 'n': 0, '1': 1, '0': 0}

        return X.iloc[:, 0].apply(lambda x: mapping.get(str(x).lower(), 0)).values.reshape(-1, 1)

class ReplaceInfWithNan(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        return X.replace([np.inf, -np.inf], np.nan)

class ConvertToString(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.astype(str)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X_future = X.copy()
        X_future['new_feature_1'] = X_future['portland_cement_kg'] * X_future['mixing_water_kg']
        X_future['new_feature_2'] = X_future['ground_slag_kg'] + X_future['coal_ash_kg']
        return X_future
#==========Arjun E Naik==============

#=========Rudra======================
import re

from word2number import w2n
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

class ColumnDropper(BaseEstimator, TransformerMixin):
    def _init_(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop or []
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X.drop(columns=self.cols_to_drop, errors='ignore')

class InfinityRemover(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X.replace([np.inf, -np.inf], np.nan)

class CementGradeParserTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        def parse(val):
            try:
                return float(val)
            except:
                val = str(val).lower().strip()
                num_match = re.findall(r"\d+\.?\d*", val)
                if num_match:
                    return float(num_match[0])
                try:
                    return w2n.word_to_num(val)
                except:
                    return np.nan
        X = X.copy()
        X['cement_grade'] = X['cement_grade'].apply(parse)
        return X

class DatePreprocessorTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['last_modified'] = pd.to_datetime(X['last_modified'], errors='coerce')
        X['last_modified_year'] = X['last_modified'].dt.year.astype('Int64')
        X['last_modified_month'] = X['last_modified'].dt.month.astype('Int64')
        X['last_modified_day'] = X['last_modified'].dt.day.astype('Int64')
        X['last_modified_weekday'] = X['last_modified'].dt.dayofweek.astype('Int64')
        X.drop(columns=['last_modified'], inplace=True)

        X['inspection_timestamp'] = pd.to_datetime(X['inspection_timestamp'], errors='coerce')
        X['inspection_year'] = X['inspection_timestamp'].dt.year.astype('Int64')
        X['inspection_month'] = X['inspection_timestamp'].dt.month.astype('Int64')
        X['inspection_day'] = X['inspection_timestamp'].dt.day.astype('Int64')
        X['inspection_weekday'] = X['inspection_timestamp'].dt.dayofweek.astype('Int64')
        X.drop(columns=['inspection_timestamp'], inplace=True)
        return X

class CategoryConverterTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Map approval values to binary
        approval_mapping = {'yes': True, 'y': True, '1': True, 'no': False, 'n': False, '0': False}
        if 'is_approved' not in X.columns:
            X['is_approved'] = 0
        else:
            X['is_approved'] = X['is_approved'].astype(str).str.strip().str.lower().map(approval_mapping)
            X['is_approved'] = X['is_approved'].fillna(0)
        X['is_approved'] = X['is_approved'].astype(int)

        # Map supplier rating to numeric values
        rating_map = {'A ++': 6, 'A+': 5, 'AA': 4.5, 'A': 4, 'A-': 3.5, 'B': 3, 'C': 2}
        if 'supplier_rating' not in X.columns:
            X['supplier_rating'] = 0
        else:
            X['supplier_rating'] = X['supplier_rating'].map(rating_map).fillna(0)

        # Ensure certain columns are numeric ints
        for col in ['is_valid_strength', 'static_col']:
            if col not in X.columns:
                X[col] = 0
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            X[col] = X[col].astype(int)

        return X


class TimeDeltaAndRatiosTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['time_since_casting'] = pd.to_timedelta(X['time_since_casting'], errors='coerce')
        X['time_since_casting_days'] = X['time_since_casting'].dt.total_seconds() / (24 * 3600)
        X.drop('time_since_casting', axis=1, inplace=True)

        X['water_binder_ratio'] = X['mixing_water_kg'] / (X['total_binder_kg'] + 1e-6)
        X['admixture_binder_ratio'] = X['chemical_admixture_kg'] / (X['total_binder_kg'] + 1e-6)
        return X

class RedundantDropperTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X.drop(columns=[
            'time_since_casting_days', 'days_since_last_modified',
            'days_since_inspection', 'random_noise'
        ], errors='ignore')

class FinalCleanerTransformer(BaseEstimator, TransformerMixin):
    def _init_(self):
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.num_cols = None

    def fit(self, X, y=None):
        self.num_cols = X.select_dtypes(include=[np.number]).columns
        X_imputed = self.imputer.fit_transform(X[self.num_cols])
        self.scaler.fit(X_imputed)
        return self

    def transform(self, X):
        if self.num_cols is None:
            raise RuntimeError("You must call fit() before transform().")
        X = X.copy()
        X[self.num_cols] = self.imputer.transform(X[self.num_cols])
        X[self.num_cols] = self.scaler.transform(X[self.num_cols])
        return X
#===============Rudra=============

#==================================shrihari=====================================================================

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


class FinalFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.raw_features = [
            'portland_cement_kg', 'ground_slag_kg', 'coal_ash_kg', 'mixing_water_kg',
            'chemical_admixture_kg', 'gravel_aggregate_kg', 'sand_aggregate_kg', 'specimen_age_days'
        ]
        self.column_mapping = {
            'portland_cement_kg': 'Cement_component_1kg_in_a_m3_mixture',
            'ground_slag_kg': 'Blast_Furnace_Slag_component_2kg_in_a_m3_mixture',
            'coal_ash_kg': 'Fly_Ash_component_3kg_in_a_m3_mixture',
            'mixing_water_kg': 'Water_component_4kg_in_a_m3_mixture',
            'chemical_admixture_kg': 'Superplasticizer_component_5kg_in_a_m3_mixture',
            'gravel_aggregate_kg': 'Coarse_Aggregate_component_6kg_in_a_m3_mixture',
            'sand_aggregate_kg': 'Fine_Aggregate_component_7kg_in_a_m3_mixture',
            'specimen_age_days': 'Age_day'
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X[self.raw_features].copy()

        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
                df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        
        df = df.rename(columns=self.column_mapping)

        cement = df['Cement_component_1kg_in_a_m3_mixture'].replace(0, 0.0001)
        df['total_binder'] = (
            df['Cement_component_1kg_in_a_m3_mixture'] +
            df['Blast_Furnace_Slag_component_2kg_in_a_m3_mixture'] +
            df['Fly_Ash_component_3kg_in_a_m3_mixture']
        )
        total_binder_for_ratio = df['total_binder'].replace(0, 0.0001)
        df['water_cement_ratio'] = df['Water_component_4kg_in_a_m3_mixture'] / cement
        total_aggregate = (
            df['Coarse_Aggregate_component_6kg_in_a_m3_mixture'] +
            df['Fine_Aggregate_component_7kg_in_a_m3_mixture']
        )
        df['agg_binder_ratio'] = total_aggregate / total_binder_for_ratio

        safe_age = df['Age_day'].clip(lower=0)
        df['log_Age_day'] = np.log1p(safe_age)

        df = df.drop(columns=[
            'Cement_component_1kg_in_a_m3_mixture',
            'Blast_Furnace_Slag_component_2kg_in_a_m3_mixture',
            'Fly_Ash_component_3kg_in_a_m3_mixture',
            'Water_component_4kg_in_a_m3_mixture',
            'Coarse_Aggregate_component_6kg_in_a_m3_mixture',
            'Fine_Aggregate_component_7kg_in_a_m3_mixture'
        ])
        
        return df


submission_pipeline = Pipeline([
    ('feature_engineer', FinalFeatureEngineer()),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=10.0, random_state=42))
])

#===============Shrihari=============================================

#==============ADITYA_SENGER======================
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import ast

class CleanAndDropTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for cleaning and dropping columns in the concrete dataset.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Drop unwanted columns
        columns_to_remove = [
            'Unnamed: 0','Row_ID','weird_col','strength_category',
            'static_col','nan_col','last_modified','inspection_timestamp','batch_id',
            'compressive_strength_duplicate','compressive_strength_mpas',
            'is_valid_strength', 'dangerous_ratio','json_notes', 'mixing_water_kg',
            'leaky_strength_log', 'pseudo_target', 'time_since_casting',
            'total_binder_kg','water_cement_ratio'
        ]
        df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True, errors='ignore')

        # Clean supplier_rating
        if 'supplier_rating' in df.columns:
            df['supplier_rating'] = (
                df['supplier_rating']
                .astype(str)
                .str.strip()
                .str.replace(' ', '', regex=False)
                .str.upper()
            )

        # Clean is_approved column
        if 'is_approved' in df.columns:
            df['is_approved'] = df['is_approved'].astype(str).str.lower().str.strip()
            mapping = {'yes': 1, 'y': 1, '1': 1, 'no': 0, 'n': 0, '0': 0}
            df['is_approved'] = df['is_approved'].map(mapping)

        # Replace infinities
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Absolute values for _kg columns
        kg_cols = [col for col in df.columns if col.endswith('_kg')]
        df[kg_cols] = df[kg_cols].abs()

        # Drop numeric columns with all NaNs
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        all_nan_cols = [col for col in num_cols if df[col].isna().all()]
        df.drop(columns=all_nan_cols, inplace=True)

        # Final numeric cleanup: make all numbers positive
        for col in df.select_dtypes(include=['number']).columns:
            df[col] = df[col].abs()

        return df


#==============ADITYA_SENGER======================
#Dont remove the following snippet and follow the same


# ---------------------------------------Yash_Verdhan2----------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.linear_model import LinearRegression

class BestFeatureSelector(BaseEstimator, TransformerMixin):
    """
    A custom transformer that selects the single best feature from a DataFrame
    based on its absolute correlation with the target variable.
    """
    def __init__(self):
        self.best_feature_ = None

    def fit(self, X, y):
        """
        Learns which feature has the highest correlation with the target.
        """
        # Combine features and target to calculate correlation
        train_df = pd.concat([X, y], axis=1)
        target_col = y.name
        
        # Calculate correlations and find the best feature
        correlations = train_df.corr(numeric_only=True)[target_col].abs().sort_values(ascending=False)
        self.best_feature_ = correlations.index[1] # Index 0 is the target itself
        print(f"Identified best feature: '{self.best_feature_}' (Correlation: {correlations.iloc[1]:.3f})")
        return self

    def transform(self, X):
        """
        Transforms the DataFrame to include only the best feature.
        """
        if self.best_feature_ is None:
            raise RuntimeError("You must fit the selector before transforming.")
        # Return a DataFrame with only the selected feature
        return X[[self.best_feature_]]

class ClippedRegression(BaseEstimator, RegressorMixin):
    """
    A wrapper for a regression model that clips the final predictions to ensure
    they are within a physically realistic range (non-negative). This class
    also handles missing values internally before fitting and predicting.
    """
    def __init__(self, model=LinearRegression()):
        self.model = model
        self.y_train_max_ = None
        self.feature_mean_ = None # To store the mean for imputing

    def fit(self, X, y):
        """
        Fits the underlying regression model and stores the maximum value of y
        for clipping. It handles missing values before fitting.
        """
        # Ensure X is a DataFrame
        X_df = pd.DataFrame(X)
        
        # Combine features and target to handle NaNs consistently
        combined = pd.concat([X_df, pd.Series(y, name='target', index=X_df.index)], axis=1).dropna()
        
        # Separate cleaned data
        X_clean = combined.drop('target', axis=1)
        y_clean = combined['target']

        # Store the mean of the feature for later imputation during prediction
        self.feature_mean_ = X_clean.mean()

        # Fit the model on clean data
        self.model.fit(X_clean, y_clean)
        
        # Store the max value from the clean target data for clipping
        self.y_train_max_ = y_clean.max()
        return self

    def predict(self, X):
        """
        Makes predictions with the underlying model and then clips the output.
        """
        if self.y_train_max_ is None:
            raise RuntimeError("You must fit the model before making predictions.")

        # Ensure X is a DataFrame and impute missing values using the stored mean
        X_df = pd.DataFrame(X)
        X_imputed = X_df.fillna(self.feature_mean_)
            
        # Make raw predictions
        y_pred_raw = self.model.predict(X_imputed)
        
        # Clip predictions to be between a small positive number and a reasonable upper bound.
        y_pred_final = np.clip(y_pred_raw, 0.1, self.y_train_max_ * 1.5)
        return y_pred_final

#===================yash verdhan ====================

#==============Mouryagna=============================
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ConcretePreprocessor(BaseEstimator, TransformerMixin):

    def fit(self,X,y=None):
        return self
    
    def transform(self,X):

        X=pd.DataFrame(X).copy()

        # converting 0/1 form
        X['is_approved']=X['is_approved'].apply(self.convert_is_approved)

        # making inf to 0
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_cols = ['portland_cement_kg', 'ground_slag_kg', 'coal_ash_kg','mixing_water_kg','chemical_admixture_kg', 'gravel_aggregate_kg','sand_aggregate_kg','specimen_age_days','total_binder_kg','is_approved']
        X[numeric_cols] = X[numeric_cols].clip(-1e6, 1e6)
        X.fillna(0, inplace=True)

        # time extracting
        X['inspection_timestamp'] = pd.to_datetime(X['inspection_timestamp'], errors='coerce')
        X['inspection_hour'] = X['inspection_timestamp'].dt.hour
        X['inspection_day'] = X['inspection_timestamp'].dt.day
        X['inspection_month'] = X['inspection_timestamp'].dt.month


        #cement grade merging
        X['cement_grade'] = X['cement_grade'].astype(str).str.upper().str.replace('FIFTY THREE', '53')
        X['cement_grade'] = X['cement_grade'].astype(str)

        #drop unwanted columns
        X=self.dropColumns(X)

        return X

    @staticmethod
    def convert_is_approved(value):
        if value =='YES' or value == 'yes' or value =='y' or value== '1':
            return 1
        else:
            return 0
    @staticmethod
    def dropColumns(X):
        dropingcolumns=['compressive_strength_duplicate','compressive_strength_mpa_str','compreesive_strength_mpa','compressive_strength_mpas','pseudo_target','static_col','nan_col','mixing_water_kg_mislabeled','json_notes','is_valid_strength','Unnamed: 0','Row_ID','last_modified','inspection_timestamp','time_since_casting','strength_category','random_noise','weird_col','batch_id','leaky_strength_log', 'water_cement_ratio', 'aggregate_binder_ratio', 'supplementary_ratio','dangerous_ratio']
        X=X.drop(columns=dropingcolumns,errors='ignore')
        return X
        
#===============Mouryagna====================



#==================LeelaVenkataSai_Nerella=============
from datetime import datetime
import json
class Leela_Venkata_Sai_Nerella2(BaseEstimator, TransformerMixin):
    def fit(self, X,y=None):
        X = self.data_cols(X)
        self.X = X
        return self
    def transform(self,X):
        X = self.data_cols(X)
        return X
    def data_cols(self,X):
        def normalize_approval(x):
            if pd.isnull(x):
                return np.nan
            x = str(x).strip().lower()
            if x in ["yes", "y", "1"]:
                return 1
            elif x in ["no", "n", "0"]:
                return 0
            else:
                return np.nan
        def extract_water(x):
            if pd.isnull(x):
                return None
            if isinstance(x, str):
                try:
                    x = json.loads(x.replace("'", '"'))
                except:
                    return None
            return x.get("water") if isinstance(x, dict) else None
        cols_dropped = ["random_noise","batch_id","leaky_strength_log","static_col","nan_col","inspection_timestamp","pseudo_target","compressive_strength_mpas","compressive_strength_duplicate","compressive_strength_mpa_str","compressive_strength_mpa","strength_category","Unnamed: 0","Row_ID","weird_col",'total_kg','aggregates_kg','cementitious_kg','mixing_water_kg','chemical_admixture_kg','gravel_aggregate_kg','sand_aggregate_kg','portland_cement_kg','ground_slag_kg','coal_ash_kg',"is_valid_strength","w_c_ratio","total_binder_kg","aggregate_binder_ratio","cement_ratio_total","water_ratio_total","supplementary_ratio","compreesive_strength_mpa"]
        days = [
                (datetime.strptime(j, "%Y-%m-%d %H:%M:%S") - datetime.strptime(i, "%Y-%m-%d")).days
                for i, j in zip(X["last_modified"], X["inspection_timestamp"])
                ]
        grade_map = {
            '32': 32,
            '32.5': 33,   
            '43': 43,
            '43a': 43,
            '53': 53,
            'fifty three': 53
        }
        rating_map={
            'A++':6,
            'A-':3,
            'A+':5,
            'A':4,
            'AA':7,
            'C':1,
            'B':2
        }
        X["days_passed"]=days
        X["days_passed"]=X["days_passed"].apply(lambda x: x if x>0 else 0)
        X["time_since_casting"] = X["time_since_casting"].apply(
            lambda x: int(float(str(x).split(" ")[0])) if pd.notnull(x) else None
        )
        X["waiting_time"]=X["specimen_age_days"]-X["time_since_casting"]
        X = X.drop(["time_since_casting","specimen_age_days"],axis=1)
        X["is_approved"] = X["is_approved"].apply(normalize_approval)
        X["cement_grade"] = X["cement_grade"].astype(str).str.strip().str.lower()
        X["cement_grade"] = X["cement_grade"].map(grade_map)
        X["supplier_rating"] = X["supplier_rating"].map(rating_map)
        X["json_notes"] = X["json_notes"].apply(extract_water)

        """feature engineering"""
       # Step 1: Create intermediate totals
        X['cementitious_kg'] = (
            X['portland_cement_kg'] +
            X['ground_slag_kg'] +
            X['coal_ash_kg']
        )

        X['aggregates_kg'] = (
            X['gravel_aggregate_kg'] +
            X['sand_aggregate_kg']
        )

        X['total_kg'] = (
            X['portland_cement_kg'] +
            X['ground_slag_kg'] +
            X['coal_ash_kg'] +
            X['mixing_water_kg'] +
            X['chemical_admixture_kg'] +
            X['gravel_aggregate_kg'] +
            X['sand_aggregate_kg']
        )

        # Step 2: Create ratio-based features
        X['w_c_ratio'] = X['mixing_water_kg'] / X['cementitious_kg']
        X['admixture_ratio'] = X['chemical_admixture_kg'] / X['cementitious_kg']
        X['gravel_sand_ratio'] = X['gravel_aggregate_kg'] / X['sand_aggregate_kg']
        X['agg_cement_ratio'] = X['aggregates_kg'] / X['cementitious_kg']
        X['cement_ratio_total'] = X['cementitious_kg'] / X['total_kg']
        X['water_ratio_total'] = X['mixing_water_kg'] / X['total_kg']


        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.drop("last_modified",axis=1,errors='ignore')
        X = X.drop(cols_dropped,axis=1)
        
        return X

#======================LeelaVenktaSaiNerella======

from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import json
class Leela_Venkata_Sai_Nerella2(BaseEstimator, TransformerMixin):
    def fit(self, X,y=None):
        X = self.data_cols(X)
        self.X = X
        return self
    def transform(self,X):
        X = self.data_cols(X)
        return X
    def data_cols(self,X):
        def normalize_approval(x):
            if pd.isnull(x):
                return np.nan
            x = str(x).strip().lower()
            if x in ["yes", "y", "1"]:
                return 1
            elif x in ["no", "n", "0"]:
                return 0
            else:
                return np.nan
        def extract_water(x):
            if pd.isnull(x):
                return None
            if isinstance(x, str):
                try:
                    x = json.loads(x.replace("'", '"'))
                except:
                    return None
            return x.get("water") if isinstance(x, dict) else None
        cols_dropped = ["random_noise","batch_id","leaky_strength_log","static_col","nan_col","inspection_timestamp","pseudo_target","compreesive_strength_mpa","strength_category","Unnamed: 0","Row_ID","weird_col",'total_kg','aggregates_kg','cementitious_kg','mixing_water_kg','chemical_admixture_kg','gravel_aggregate_kg','sand_aggregate_kg','portland_cement_kg','ground_slag_kg','coal_ash_kg',"is_valid_strength","w_c_ratio","total_binder_kg","aggregate_binder_ratio","cement_ratio_total","water_ratio_total","supplementary_ratio"]
        from datetime import datetime
        import pandas as pd

        def try_parse(date_str):
            if pd.isna(date_str) or date_str == "":
                return None
            
            if isinstance(date_str, datetime):
                return date_str

            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%d-%m-%Y %H.%M",
                "%Y-%m-%d",
                "%d/%m/%Y",
                "%d-%m-%Y",
                "%Y/%m/%d",
                "%Y.%m.%d %H:%M",
                "%d %b %Y",
                "%d %B %Y",
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue

            raise ValueError(f"Date format not supported: {date_str}")

        days = [
            (try_parse(j) - try_parse(i)).days
            for i, j in zip(X["last_modified"], X["inspection_timestamp"])
        ]

        grade_map = {
            '32': 32,
            '32.5': 33,   
            '43': 43,
            '43a': 43,
            '53': 53,
            'fifty three': 53
        }
        rating_map={
            'A++':6,
            'A-':3,
            'A+':5,
            'A':4,
            'AA':7,
            'C':1,
            'B':2
        }
        X["days_passed"]=days
        X["days_passed"]=X["days_passed"].apply(lambda x: x if x>0 else 0)
        X["time_since_casting"] = X["time_since_casting"].apply(
            lambda x: int(float(str(x).split(" ")[0])) if pd.notnull(x) else None
        )
        X["waiting_time"]=X["specimen_age_days"]-X["time_since_casting"]
        X = X.drop(["time_since_casting","specimen_age_days"],axis=1)
        X["is_approved"] = X["is_approved"].apply(normalize_approval)
        X["cement_grade"] = X["cement_grade"].astype(str).str.strip().str.lower()
        X["cement_grade"] = X["cement_grade"].map(grade_map)
        X["supplier_rating"] = X["supplier_rating"].map(rating_map)
        X["json_notes"] = X["json_notes"].apply(extract_water)

        """feature engineering"""
       # Step 1: Create intermediate totals
        X['cementitious_kg'] = (
            X['portland_cement_kg'] +
            X['ground_slag_kg'] +
            X['coal_ash_kg']
        )

        X['aggregates_kg'] = (
            X['gravel_aggregate_kg'] +
            X['sand_aggregate_kg']
        )

        X['total_kg'] = (
            X['portland_cement_kg'] +
            X['ground_slag_kg'] +
            X['coal_ash_kg'] +
            X['mixing_water_kg'] +
            X['chemical_admixture_kg'] +
            X['gravel_aggregate_kg'] +
            X['sand_aggregate_kg']
        )

        # Step 2: Create ratio-based features
        X['w_c_ratio'] = X['mixing_water_kg'] / X['cementitious_kg']
        X['admixture_ratio'] = X['chemical_admixture_kg'] / X['cementitious_kg']
        X['gravel_sand_ratio'] = X['gravel_aggregate_kg'] / X['sand_aggregate_kg']
        X['agg_cement_ratio'] = X['aggregates_kg'] / X['cementitious_kg']
        X['cement_ratio_total'] = X['cementitious_kg'] / X['total_kg']
        X['water_ratio_total'] = X['mixing_water_kg'] / X['total_kg']


        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.drop("last_modified",axis=1,errors='ignore')
        X = X.drop(cols_dropped,axis=1)
        
        return X




