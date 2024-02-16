import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from collections import defaultdict
from tqdm.auto import tqdm


class CurveParamPredictor: 
    def __init__(self, folds = 10):
        self.folds = folds
        self.all_x0_models = {}
        self.all_k_models = {}

    
    def predict(sample):
        if 'curve_L' in sample.columns:
            pred_X = sample.drop(['batch_id','sample_id','curve_L','curve_k','curve_x0'], axis = 1)
        else:
            pred_X = sample.drop(['batch_id','sample_id'], axis = 1)
        
        return [.60, self.best_model_x0.predict(pred_X), self.best_model_k.predict(pred_X)]
    
    def train(self, df):
        self.samples_to_predict = df[df['curve_L'].isnull()]
        self.samples_to_predict = self.samples_to_predict.drop(['batch_id','sample_id','curve_L','curve_k','curve_x0'], axis = 1)
        
        df = df[~df['curve_L'].isnull()]
        
        self.y_x0 = df['curve_x0']
        self.y_k = df['curve_k']
        self.X = df.drop(['batch_id','sample_id','curve_L','curve_k','curve_x0'], axis = 1)

        self.cat_features = self.X.select_dtypes(include=['object','bool']).columns
        self.num_features = self.X.select_dtypes(exclude=['object','bool']).columns
        
        self.cat_imp = SimpleImputer(missing_values= np.nan, strategy= 'most_frequent')
            
        self.num_imp = SimpleImputer(missing_values= np.nan, strategy= 'median')

        if len(self.cat_features) > 0:
            self.cat_imp.fit(self.X[self.cat_features])
        
        self.num_imp.fit(self.X[self.num_features])
        
        if len(self.cat_features) > 0:
            self.X[self.cat_features] = self.cat_imp.transform(self.X[self.cat_features])
            
        self.X[self.num_features] = self.num_imp.transform(self.X[self.num_features])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('OHE', OneHotEncoder(handle_unknown = 'ignore'), self.cat_features),
                ('Scalar', StandardScaler(), self.num_features)
            ],
        )
        regressors = {
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(max_iter=10000),
            'ElasticNet': ElasticNet(max_iter=10000),
            'Random Forest Regressor': RandomForestRegressor(),
            'SVR': SVR(),
        }
        
        hyperparameters = {}

        hyperparameters['Ridge Regression'] = {
            'Regressor__alpha' : [5, 2, 1, 0.1]
        }

        hyperparameters['Lasso Regression'] = {
            'Regressor__alpha' : [5, 2, 1, 0.1]
        }

        hyperparameters['ElasticNet'] = {
            'Regressor__alpha' : [5, 2, 1, 0.1],
            'Regressor__l1_ratio' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'Regressor__max_iter' : [10000]

        }

        hyperparameters['Random Forest Regressor'] = {
            'Regressor__max_depth': [3,5,10,25], 
            'Regressor__min_samples_split': [20, 25],
            'Regressor__n_estimators' : [30,40,50],
            'Regressor__max_features' : ['sqrt','log2',1.0]

        }

        hyperparameters['SVR'] = {
            'Regressor__kernel' : ['linear', 'rbf', 'sigmoid'],
            'Regressor__degree' : [2,3,4],
            'Regressor__C' : [0.1,0.2,0.5,1,2],
        }
        
        s = 0
        for i in hyperparameters:
            p = 1
            for j in hyperparameters[i]:
                p = p * len(hyperparameters[i][j])
            #print(f'{p*self.folds*2} {i} models will be trained')
            s += p * self.folds*2
        #print(f'{s} models in total will be trained\n-----')
        
        best_rmse = np.inf
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y_x0, test_size=0.2)

        for name, regressor in regressors.items():
            #print(f'Training {name} x0 models')
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('Regressor', regressor)
            ])


            grids = GridSearchCV(model,scoring= "neg_root_mean_squared_error", param_grid = hyperparameters[name], verbose = 0, cv = self.folds)
            grids.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = grids.predict(X_test)

            # Evaluate the model
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            #print(f"{name} - RMSE on the test set: {rmse}")
            
            self.all_x0_models[name] = rmse

            if rmse < best_rmse:
                self.best_model_x0 = grids
                best_rmse = rmse
            #print('-----')
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y_k, test_size=0.2)

            
        for name, regressor in regressors.items():
            #print(f'Training {name} k models')
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('Regressor', regressor)
            ])


            grids = GridSearchCV(model,scoring= "neg_root_mean_squared_error", param_grid = hyperparameters[name], verbose = 0, cv = self.folds)
            grids.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = grids.predict(X_test)

            # Evaluate the model
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            #print(f"{name} - RMSE on the test set: {rmse}")
            
            self.all_k_models[name] = rmse


            if rmse < best_rmse:
                self.best_model_k = grids
                best_rmse = rmse
            #print('-----')
            
    def performance_analysis(self, df):
        n = len(df)
        
        self.stats_dict_x0 = defaultdict(lambda :[])
        self.stats_dict_k = defaultdict(lambda :[])

        
        for i in range(13, n, 10):
            print(F'-----Analyzing models performance on {i} samples-----')
            self.stats_dict_k['n'].append(i)
            self.stats_dict_x0['n'].append(i)
            temp_stats_dict_k = defaultdict(lambda: [])
            temp_stats_dict_x0 = defaultdict(lambda: [])

            for _ in tqdm(range(10), desc = f'Running trials for {i} samples.'):
                temp_df = df.sample(i)

                self.train(temp_df)

                for k in self.all_k_models:
                    temp_stats_dict_k[k].append(self.all_k_models[k])

                for k in self.all_x0_models:
                    temp_stats_dict_x0[k].append(self.all_x0_models[k])
                    
            for k in self.all_k_models:
                self.stats_dict_k[k].append(np.mean(temp_stats_dict_k[k]))

            for k in self.all_x0_models:
                self.stats_dict_x0[k].append(np.mean(temp_stats_dict_x0[k]))
