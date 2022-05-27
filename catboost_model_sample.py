import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from category_encoders.ordinal import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from catboost import CatBoostClassifier, Pool



def train_model(train, test):
  n_est = 2000
  seed = 42
  n_class = 3
  # n_fold = 18
  n_fold = 3

  target = 'credit'
  X = train.drop(target, axis=1)
  y = train[target]
  X_test = test

  X = X.reset_index(drop=True)
  y = y.reset_index(drop=True)
  X_test = X_test.reset_index(drop=True)

  skfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
  folds=[]
  for train_idx, valid_idx in skfold.split(X, y):
          folds.append((train_idx, valid_idx))

  cat_pred = np.zeros((X.shape[0], n_class))
  cat_pred_test = np.zeros((X_test.shape[0], n_class))
  cat_cols = ['income_type', 'edu_type', 'family_type', 'house_type', 'occyp_type', 'ID']

  for fold in range(n_fold):
      # print(f'\n----------------- Fold {fold} -----------------\n')
      train_idx, valid_idx = folds[fold]
      X_train, X_valid, y_train, y_valid = X.iloc[train_idx], X.iloc[valid_idx], y[train_idx], y[valid_idx]
      train_data = Pool(data=X_train, label=y_train, cat_features=cat_cols)
      valid_data = Pool(data=X_valid, label=y_valid, cat_features=cat_cols)

      model_cat = CatBoostClassifier()
      model_cat.fit(train_data, eval_set=valid_data, use_best_model=True, early_stopping_rounds=100, verbose=100)
      
      cat_pred[valid_idx] = model_cat.predict_proba(X_valid)
      cat_pred_test += model_cat.predict_proba(X_test) / n_fold
      # print(f'CV Log Loss Score: {log_loss(y_valid, cat_pred[valid_idx]):.6f}')
  return model_cat, X_train

def result(model_cat, X_train):
  y_predict= model_cat.predict(X_train)
  # print(f'\tacc: {accuracy_score(y_train, y_predict):.6f}')     
  # print(f'\tLog Loss: {log_loss(y, cat_pred):.6f}')
  # print('='*60)

  return y_predict