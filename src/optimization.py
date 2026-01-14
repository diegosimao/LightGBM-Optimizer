import optuna
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class OptimizationManager:
    def __init__(self, data, search_space, callback_fn=None):
        self.X_train, self.X_test, self.y_train, self.y_test = data
        self.search_space = search_space
        self.callback_fn = callback_fn
        self.history = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'best_f1': []}
        
    def objective(self, trial):
        # 1. Hyperparameters for LightGBM
        params = {
            'objective': 'multiclass' if len(set(self.y_train)) > 2 else 'binary',
            'metric': 'multi_logloss' if len(set(self.y_train)) > 2 else 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_jobs': -1,
            'random_state': 42,
            
            # Tunable parameters
            'n_estimators': trial.suggest_int('n_estimators', 
                                            self.search_space.get('n_estimators', [50, 200])[0], 
                                            self.search_space.get('n_estimators', [50, 200])[1]),
            'learning_rate': trial.suggest_float('learning_rate', 
                                               self.search_space.get('learning_rate', [0.01, 0.3])[0], 
                                               self.search_space.get('learning_rate', [0.01, 0.3])[1], log=True),
            'num_leaves': trial.suggest_int('num_leaves', 
                                          self.search_space.get('num_leaves', [20, 100])[0], 
                                          self.search_space.get('num_leaves', [20, 100])[1]),
            'max_depth': trial.suggest_int('max_depth', 
                                         self.search_space.get('max_depth', [3, 12])[0], 
                                         self.search_space.get('max_depth', [3, 12])[1]),
        }
        
        # 2. Train Model
        clf = lgb.LGBMClassifier(**params)
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        
        # 3. Metrics
        metrics = {
            'acc': accuracy_score(self.y_test, preds),
            'prec': precision_score(self.y_test, preds, average='macro', zero_division=0),
            'rec': recall_score(self.y_test, preds, average='macro', zero_division=0),
            'f1': f1_score(self.y_test, preds, average='macro', zero_division=0)
        }
        
        # 4. Update Internal History
        for k, v in metrics.items():
            self.history[k].append(v)
            
        current_best = max(self.history['f1'])
        self.history['best_f1'].append(current_best)
        
        # 5. Invoke Callback
        if self.callback_fn:
            self.callback_fn(self.history, metrics, current_best)
            
        return metrics['f1']

    def run(self, n_trials=50):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        return study
