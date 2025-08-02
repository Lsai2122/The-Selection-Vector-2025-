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

#==================================shrihari=====================================================================

class FinalFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.columns_ = X.columns
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=self.columns_)

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
    
column_mapping = {
    'portland_cement_kg': 'Cement_component_1kg_in_a_m3_mixture',
    'ground_slag_kg': 'Blast_Furnace_Slag_component_2kg_in_a_m3_mixture',
    'coal_ash_kg': 'Fly_Ash_component_3kg_in_a_m3_mixture',
    'mixing_water_kg': 'Water_component_4kg_in_a_m3_mixture',
    'chemical_admixture_kg': 'Superplasticizer_component_5kg_in_a_m3_mixture',
    'gravel_aggregate_kg': 'Coarse_Aggregate_component_6kg_in_a_m3_mixture',
    'sand_aggregate_kg': 'Fine_Aggregate_component_7kg_in_a_m3_mixture',
    'specimen_age_days': 'Age_day'
}
df = df.rename(columns=column_mapping)

TARGET = 'compressive_strength_mpa'
CORE_FEATURES = list(column_mapping.values())
df_clean = df[CORE_FEATURES + [TARGET]]

df_clean.dropna(subset=[TARGET], inplace=True)

# 2. Handle Outliers
for column in df_clean.columns:
    if df_clean[column].dtype in ['int64', 'float64']:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean[column] = np.where(df_clean[column] > upper_bound, upper_bound, df_clean[column])
        df_clean[column] = np.where(df_clean[column] < lower_bound, lower_bound, df_clean[column])

# 3. Prepare Final Data and Pipeline
X = df_clean.drop(TARGET, axis=1)
y = df_clean[TARGET]

# This pipeline uses the best configuration we found
submission_pipeline = Pipeline([
    ('feature_engineer', FinalFeatureEngineer()),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=10.0, random_state=42)) # Using the best alpha
])
#============================================shrihari=======================================================================================================