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
import catboost as cb
from catboost import Pool
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import warnings 
import seaborn as sns

warnings.filterwarnings('ignore')


class CurveParamPredictor: 
    def __init__(self, folds = 10):
        self.folds = folds
        self.all_x0_models = {}
        self.all_k_models = {}
        self.overall_best_model_x0 = False
        self.overall_best_model_k = False
        self.overall_best_rmse_x0 = np.inf
        self.overall_best_rmse_k = np.inf


    
    def predict(self, sample):
        if 'curve_L' in sample.columns:
            sample = sample.drop(['batch_id','sample_id','curve_L','curve_k','curve_x0'], axis = 1)
        else:
            sample = sample.drop(['batch_id','sample_id'], axis = 1)
            
        if len(self.cat_features) > 0:
            sample[self.cat_features] = self.cat_imp.transform(sample[self.cat_features])
            
        sample[self.num_features] = self.num_imp.transform(sample[self.num_features])
        
        if type(self.overall_best_model_k) == type(False):
            return [1, self.best_model_x0.predict(sample)[0], self.best_model_k.predict(sample)[0]]
        else:
            return [1, self.overall_best_model_x0.predict(sample)[0], self.overall_best_model_k.predict(sample)[0]]
    
    def train(self, df):
        self.samples_to_predict = df[df['curve_L'].isnull()]
        self.samples_to_predict = self.samples_to_predict.drop(['batch_id','sample_id','curve_L','curve_k','curve_x0'], axis = 1)
        
        df = df[~df['curve_L'].isnull()]
        
        self.y = df[['curve_x0','curve_k']]
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
                ('OHE', OneHotEncoder(handle_unknown = 'ignore', sparse = False
                                     ), self.cat_features),
                ('Scalar', StandardScaler(), self.num_features)
            ],
        )
        regressors = {
            #'Ridge Regression': Ridge(),
            #'Lasso Regression': Lasso(max_iter=10000),
            'ElasticNet': ElasticNet(max_iter=10000),
            #'Random Forest Regressor': RandomForestRegressor(),
            'SVR': SVR(),
            'CatBoost' : cb.CatBoost()
        }
        
        hyperparameters = {}

        hyperparameters['Ridge Regression'] = {
            'Regressor__alpha' : [1, 0.5, 0.1]
        }

        hyperparameters['Lasso Regression'] = {
            'Regressor__alpha' : [1, 0.5, 0.1]
        }

        hyperparameters['ElasticNet'] = {
            'Regressor__alpha' : [1, 0.5, 0.1],
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
            'Regressor__C' : [1, 0.5, 0.1],
        }
        
        hyperparameters['CatBoost'] = {}
        
        s = 0
        for i in hyperparameters:
            p = 1
            for j in hyperparameters[i]:
                p = p * len(hyperparameters[i][j])
            #print(f'{p*self.folds*2} {i} models will be trained')
            s += p * self.folds*2
        #print(f'{s} models in total will be trained\n-----')
        
        self.best_rmse_x0 = np.inf
        self.best_rmse_k = np.inf
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

        for name, regressor in regressors.items():
            #print(f'Training {name} x0 models')
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('Regressor', regressor)
            ])


            grids = GridSearchCV(model,scoring= "neg_root_mean_squared_error", param_grid = hyperparameters[name], verbose = 0, cv = self.folds)
            
            if name == 'CatBoost':
                grids.fit(self.X_train, self.y_train['curve_x0'], Regressor__verbose = 0)
            else:
                grids.fit(self.X_train, self.y_train['curve_x0'])
                
            # Make predictions on the test set
            y_pred = grids.predict(self.X_test)

            # Evaluate the model
            rmse = np.sqrt(mean_squared_error(self.y_test['curve_x0'], y_pred))
            #print(f"{name} - RMSE on the test set: {rmse}")
            
            self.all_x0_models[name] = rmse

            if rmse < self.best_rmse_x0:
                self.best_model_x0 = grids
                self.best_rmse_x0 = rmse
            #print('-----')
        
            
        for name, regressor in regressors.items():
            #print(f'Training {name} k models')
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('Regressor', regressor)
            ])


            grids = GridSearchCV(model,scoring= "neg_root_mean_squared_error", param_grid = hyperparameters[name], verbose = 0, cv = self.folds)
            
            if name == 'CatBoost':
                grids.fit(self.X_train, self.y_train['curve_k'], Regressor__verbose = 0)
            else:
                grids.fit(self.X_train, self.y_train['curve_k'])


            # Make predictions on the test set
            y_pred = grids.predict(self.X_test)

            # Evaluate the model
            rmse = np.sqrt(mean_squared_error(self.y_test['curve_k'], y_pred))
            #print(f"{name} - RMSE on the test set: {rmse}")
            
            self.all_k_models[name] = rmse


            if rmse < self.best_rmse_k:
                self.best_model_k = grids
                self.best_rmse_k = rmse
            #print('-----')
            
    def sample_performance_analysis(self, df):
        n = len(df)
        
        self.stats_dict_x0 = defaultdict(lambda :[])
        self.stats_dict_k = defaultdict(lambda :[])

        
        for i in range(15, len(self.X), 5):
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
                self.stats_dict_k[k].append(temp_stats_dict_k[k])

            for k in self.all_x0_models:
                print(k)
                self.stats_dict_x0[k].append(temp_stats_dict_x0[k])
                
    def model_variance_analysis(self,df, samplings = 100):
        self.stats_dict_x0 = defaultdict(lambda :[])
        self.stats_dict_k = defaultdict(lambda :[])
        
        for i in [len(df)]:
            print(F'-----Analyzing models variance on {i} samples-----')
            self.stats_dict_k['n'].append(i)
            self.stats_dict_x0['n'].append(i)
            temp_stats_dict_k = defaultdict(lambda: [])
            temp_stats_dict_x0 = defaultdict(lambda: [])

            for _ in tqdm(range(samplings), desc = f'Running trials for {i} samples.'):
                temp_df = df.sample(i)

                self.train(temp_df)
                
                if self.best_rmse_k > self.overall_best_rmse_k:
                    self.overall_best_rmse_k = self.best_rmse_k
                    self.overall_best_model_k = self.best_model_k
                    
                if self.best_rmse_x0 > self.overall_best_rmse_x0:
                    self.overall_best_rmse_x0 = self.best_rmse_x0
                    self.overall_best_model_x0 = self.best_model_x0

                for k in self.all_k_models:
                    temp_stats_dict_k[k].append(self.all_k_models[k])

                for k in self.all_x0_models:
                    temp_stats_dict_x0[k].append(self.all_x0_models[k])
                    
            for k in self.all_k_models:
                self.stats_dict_k[k].append(temp_stats_dict_k[k])

            for k in self.all_x0_models:
                self.stats_dict_x0[k].append(temp_stats_dict_x0[k])
        cols = {}
                
        for i in range(len(self.stats_dict_x0['n'])):
            for j in self.stats_dict_k:
                if j == 'n':
                    continue

                temp = np.array(self.stats_dict_x0[j][i])
                temp_mean = np.mean(temp)
                temp_std = np.std(temp)
                temp_med = np.median(temp)
                cols[j] = temp

                print(f'{j} stats when predicting x0:')
                print(f'RMSE average: {temp_mean}')
                print(f'RMSE std: {temp_std}')
                print(f'RMSE median: {temp_med}')
                print(f'RMSE max: {np.max(temp)}')
                print(f'RMSE min: {np.min(temp)}')
                
            temp_df = pd.DataFrame(cols)
            plt.figure(figsize=(10,6))
            plt.yscale('log')
            plt.ylabel('RMSE')
            plt.title('Boxplot of models\' performance on test set when predicting x0')
            sns.boxplot(temp_df, color = 'red')
            plt.show()


        for i in range(len(self.stats_dict_k['n'])):
            for j in self.stats_dict_k:
                if j == 'n':
                    continue

                temp = np.array(self.stats_dict_k[j][i])
                temp_mean = np.mean(temp)
                temp_std = np.std(temp)
                temp_med = np.median(temp)
                cols[j] = temp

                print(f'{j} stats when predicting k:')
                print(f'RMSE average: {temp_mean}')
                print(f'RMSE std: {temp_std}')
                print(f'RMSE median: {temp_med}')
                print(f'RMSE max: {np.max(temp)}')
                print(f'RMSE min: {np.min(temp)}')
                
            temp_df = pd.DataFrame(cols)
            plt.figure(figsize=(10,6))
            plt.yscale('log')
            plt.ylabel('RMSE')
            plt.title('Boxplot of models\' performance on test set when predicting k')
            sns.boxplot(temp_df, color = 'red')
            plt.show()
                
    def generate_parity_plots(self):
        observed_x0 = []
        observed_k = []

        predicted_x0 = self.best_model_x0.predict(self.X_test)
        predicted_k = self.best_model_k.predict(self.X_test)

        obs = [self.y_test['curve_x0'], self.y_test['curve_k']]

            
        plt.scatter(obs[0], predicted_x0)
        plt.plot(obs[0], obs[0], c = 'red')
        plt.title('x0 Parity plot')
        plt.ylabel('Predicted x0')
        plt.xlabel('Real x0')
        plt.show()
        
        plt.scatter(obs[1], predicted_k)
        plt.plot(obs[1], obs[1], c = 'red')
        plt.title('k Parity plot')
        plt.ylabel('Predicted k')
        plt.xlabel('Real k')
        plt.show()
        
    def get_feature_importances(self):
        
        if type(self.best_model_x0.best_estimator_.named_steps['Regressor']) != type(cb.CatBoost()) and type(self.best_model_k.best_estimator_.named_steps['Regressor']) != type(cb.CatBoost()):
            return 'Neither best model is catboost.'
        df_temp = pd.DataFrame()

        df_temp['feature'] = self.best_model_x0.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
        if type(self.best_model_x0.best_estimator_.named_steps['Regressor']) != type(cb.CatBoost()):
            df_temp['importances_x0'] = self.best_model_x0.best_estimator_.named_steps["Regressor"].feature_importances_
        if type(self.best_model_k.best_estimator_.named_steps['Regressor']) != type(cb.CatBoost()):
            df_temp['importances_k'] = self.best_model_k.best_estimator_.named_steps["Regressor"].feature_importances_
        df_temp.sort_values('importances_x0', ascending =False)
        
        return df_temp
