import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import config
from data_loader import load_and_preprocess, get_folds, save_predictions

def objective_xgb(trial, X, y, cat_features, cat_dtypes, X_original=None, y_original=None):
    params = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
        'eval_metric': 'auc',
        'enable_categorical': True,
        'random_state': config.SEED,
        'early_stopping_rounds': 30
    }
    
    # Use all folds for accurate evaluation (align with train_cat.py)
    folds = get_folds(X, y)
    scores = []
    
    for train_idx, val_idx in folds:
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Add original dataset to training if available (align with train_cat.py)
        if X_original is not None and y_original is not None:
            X_tr = pd.concat([X_tr, X_original], axis=0)
            y_tr = pd.concat([y_tr, y_original])
        
        for c in cat_features:
            X_tr[c] = X_tr[c].astype(cat_dtypes[c])
            X_val[c] = X_val[c].astype(cat_dtypes[c])
            
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        scores.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
        
    return np.mean(scores)

def run_xgb():
    result = load_and_preprocess()
    if len(result) == 8:
        train, test, original, num_cols, cat_cols, all_cat_cols, target_col, id_col = result
    else:
        # Fallback for old version
        train, test, num_cols, cat_cols, all_cat_cols, target_col, id_col = result
        original = None
    
    X = train.drop(columns=[target_col, id_col])
    y = train[target_col]
    X_test = test.drop(columns=[id_col])
    
    # Prepare original dataset if available
    if original is not None:
        X_original = original.drop(columns=[target_col], errors='ignore')
        y_original = original[target_col]
        print(f"✅ Original dataset: {len(original)} samples will be added to training")
    else:
        X_original = None
        y_original = None
    
    # Align feature set behavior with train_cat.py fixed mode:
    # - When using fixed hyperparameters + original dataset, use only base features (no *_cat copies)
    use_fixed_mode = bool(getattr(config, "USE_OWN_PARAMETERS", False))
    use_base_features_only = use_fixed_mode and bool(getattr(config, "USE_ORIGINAL_DATASET", False))
    if use_base_features_only:
        base_feature_cols = [c for c in X.columns if not str(c).endswith("_cat")]
        X = X[base_feature_cols].copy()
        X_test = X_test[base_feature_cols].copy()
        if X_original is not None:
            X_original = X_original[base_feature_cols].copy()
        cat_features = list(cat_cols)
        print(f"🔧 Fixed-mode feature set: using {len(base_feature_cols)} base features (no categorical copies)")
    else:
        cat_features = list(all_cat_cols)
    
    cat_dtypes = {}
    for c in cat_features:
        all_cats = set(train[c].astype(str).unique()) | set(test[c].astype(str).unique())
        if original is not None and c in original.columns:
            all_cats |= set(original[c].astype(str).unique())
        cat_dtypes[c] = pd.CategoricalDtype(categories=list(all_cats), ordered=False)
        
    xgb_params_list = [{
        'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 6, 
        'eval_metric': 'auc', 'enable_categorical': True, 'random_state': config.SEED,
        'early_stopping_rounds': 50
    }]

    # Fixed hyperparameters mode (align with train_cat.py)
    if use_fixed_mode:
        xgb_params_list = [{
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'auc',
            'enable_categorical': True,
            'random_state': 1,  # fixed seed
        }]
        print("📌 Using fixed hyperparameters")
    
    # Skip Optuna when using fixed hyperparameters (align with train_cat.py)
    if config.USE_OPTUNA and not use_fixed_mode:
        print(f"\nRunning Optuna Optimization for XGBoost (Timeout: {config.OPTUNA_TIMEOUT}s)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective_xgb(trial, X, y, cat_features, cat_dtypes, X_original, y_original),
                           n_trials=config.N_TRIALS, timeout=config.OPTUNA_TIMEOUT, show_progress_bar=True)
        
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        completed_trials.sort(key=lambda t: t.value, reverse=True)
        top_trials = completed_trials[:config.N_TOP_MODELS]
        
        xgb_params_list = []
        for t in top_trials:
            base = {
                'n_estimators': 1000, 'eval_metric': 'auc', 'enable_categorical': True, 
                'random_state': config.SEED, 'early_stopping_rounds': 50
            }
            base.update(t.params)
            xgb_params_list.append(base)
            
        print(f"Top {len(xgb_params_list)} XGB Params selected.")

    print("\nStarting Training XGBoost...")
    if config.USE_PSEUDO_LABELING:
        print(f"✅ Pseudo-labeling: ENABLED (using public test as training data)")
        print(f"   Pseudo-label threshold: {config.PSEUDO_THRESHOLD}")
    
    # Use fixed CV seed in fixed mode (align with train_cat.py)
    if use_fixed_mode:
        skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=1)
        folds = list(skf.split(X, y))
        print("📌 Using random_state=1 for CV splits (fixed seed)")
    else:
        folds = get_folds(train.drop(columns=[target_col, id_col]), train[target_col])
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    for train_idx, val_idx in tqdm(folds, desc="CV Folds"):
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        X_test_fold = X_test.copy()
        
        # Add original dataset to training if available
        if X_original is not None and y_original is not None:
            X_tr = pd.concat([X_tr, X_original], axis=0)
            y_tr = pd.concat([y_tr, y_original])
        
        for c in cat_features:
            X_tr[c] = X_tr[c].astype(cat_dtypes[c])
            X_val[c] = X_val[c].astype(cat_dtypes[c])
            X_test_fold[c] = X_test_fold[c].astype(cat_dtypes[c])
            
        fold_oof = np.zeros(len(val_idx))
        fold_test = np.zeros(len(test))
        
        for params in xgb_params_list:
            # Step 1: Train initial model (for pseudo-labeling if enabled)
            if config.USE_PSEUDO_LABELING:
                # Train initial model to get pseudo-labels
                initial_params = params.copy()
                initial_params['n_estimators'] = min(500, initial_params.get('n_estimators', 1000))  # Faster for pseudo-labeling
                model_initial = xgb.XGBClassifier(**initial_params)
                model_initial.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                
                # Get pseudo-labels from public test
                pseudo_proba = model_initial.predict_proba(X_test_fold)[:, 1]
                pseudo_labels = (pseudo_proba >= config.PSEUDO_THRESHOLD).astype(int)
                
                # Filter by confidence if threshold is not 0.5
                if config.PSEUDO_THRESHOLD != 0.5:
                    confidence_mask = (pseudo_proba >= 0.7) | (pseudo_proba <= 0.3)
                    X_test_pseudo = X_test_fold[confidence_mask]
                    y_test_pseudo = pd.Series(pseudo_labels[confidence_mask], index=X_test_pseudo.index)
                else:
                    # Use all pseudo-labels
                    X_test_pseudo = X_test_fold.copy()
                    y_test_pseudo = pd.Series(pseudo_labels, index=X_test_pseudo.index)
                
                # Add pseudo-labeled test to training
                X_tr_final = pd.concat([X_tr, X_test_pseudo], axis=0)
                y_tr_final = pd.concat([y_tr, y_test_pseudo])
            else:
                X_tr_final = X_tr
                y_tr_final = y_tr
            
            # Step 2: Train final model
            model = xgb.XGBClassifier(**params)
            if use_fixed_mode:
                # No early stopping in fixed mode (align with train_cat.py "full training" behavior)
                model.fit(X_tr_final, y_tr_final, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_tr_final, y_tr_final, eval_set=[(X_val, y_val)], verbose=False)
            fold_oof += model.predict_proba(X_val)[:, 1] / len(xgb_params_list)
            fold_test += model.predict_proba(X_test_fold)[:, 1] / len(xgb_params_list)
            
        oof_preds[val_idx] = fold_oof
        test_preds += fold_test / config.N_SPLITS
    
    print(f"XGB CV Score: {roc_auc_score(y, oof_preds):.5f}")
    if config.USE_PSEUDO_LABELING:
        print(f"✅ Pseudo-labeling: Used public test ({len(test):,} samples) as training data")
    save_predictions('xgb', oof_preds, test_preds)

if __name__ == "__main__":
    run_xgb()

