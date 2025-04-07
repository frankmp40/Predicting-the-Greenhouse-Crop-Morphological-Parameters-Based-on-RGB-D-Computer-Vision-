import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load data
df = pd.read_csv('Processed_data/data_analysis.csv')

# Convert Variety from str to numbers
variety_mapping = {variety: idx for idx, variety in enumerate(df['Variety'].unique())}
df['Variety'] = df['Variety'].map(variety_mapping)

# Define features and target columns
features_cols = ['Leaf_pixel_ratio', 'Edge_pixel_ratio', 'EXG', 'EXR', 'VARI', 'Variety']
target_cols = ['FreshWeightShoot', 'DryWeightShoot', 'Height', 'Diameter', 'LeafArea', 'DMC']

# Split the dataset into training and evaluation sets
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialise
# Choose data processing method and prediction model
# Prediction model includes: RF-Random Forest, LR-Linear Regression, XGBOOST
# Data processing methods are PCA or None
model_list = ['RF', 'LR', 'XGBOOST']
# model_list = ['LR']
fs_list = ['PCA', 'NONE']
# fs_list = ['PCA']

list_fs = []
list_model = []
list_target = []
list_r2 = []
list_rmse = []
list_rrmse = []
list_nrmse = []
list_coef = []

for fs in fs_list:
    if fs == 'PCA':
        # Standardize the data
        scaler = StandardScaler()
        train_features_standardized = scaler.fit_transform(train_df[features_cols])
        eval_features_standardized = scaler.transform(eval_df[features_cols])

        # Apply PCA for feature selection
        pca = PCA(n_components=3)
        train_features = pca.fit_transform(train_features_standardized)
        eval_features = pca.transform(eval_features_standardized)
    else:
        train_features = train_df[features_cols]
        eval_features = eval_df[features_cols]

    for modelname in model_list:
        if modelname == 'RF':
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif modelname == 'LR':
            model = LinearRegression()
            param_grid = {}
        elif modelname == 'XGBOOST':
            model = XGBRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
            }
        else:
            print(model, ' is not found!')
            break

        # Initialize GridSearchCV
        grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)


        # Train the model for each target column using Grid Search
        for target_col in target_cols:
            # Fit the GridSearchCV object to find the best hyperparameters
            grid_search.fit(train_features, train_df[target_col])

            # Get the best parameters
            best_params = grid_search.best_params_
            print(f'Best hyperparameters for {target_col}: {best_params}')

            # Make predictions on the evaluation set using the best model
            predictions = grid_search.predict(eval_features)

            # Evaluate the model using R-squared (R2)
            r2 = r2_score(eval_df[target_col], predictions)
            print(f'R-squared for {target_col}: {r2}')

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(eval_df[target_col], predictions))
            print(f'RMSE for {target_col}: {rmse}')

            # Calculate RRMSE
            rrmse = rmse / eval_df[target_col].mean()

            # Calculate NRMSE
            target_range = eval_df[target_col].max() - eval_df[target_col].min()
            nrmse = rmse / target_range

            list_fs.append(fs)
            list_model.append(modelname)
            list_target.append(target_col)
            list_coef.append(best_params)
            list_r2.append(r2)
            list_rmse.append(rmse)
            list_rrmse.append(rrmse)
            list_nrmse.append(nrmse)

df_result = pd.DataFrame({'FS': list_fs, 'Model': list_model, 'Target': list_target,
                          'R2': list_r2, 'RMSE':list_rmse, 'RRMSE':list_rrmse, 'NRMSE':list_nrmse, 'Coef': list_coef})
df_result.to_csv('Processed_data/model_result.csv')
