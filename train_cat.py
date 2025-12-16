import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings
import json
import os
import config
from data_loader import load_and_preprocess, get_folds, save_predictions

# Suppress all warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.ERROR)  # Only show errors, not warnings

BEST_PARAMS_FILE = 'best_cat_params.json'

def load_best_params():
    """Load best params from previous run if exists"""
    if os.path.exists(BEST_PARAMS_FILE):
        try:
            with open(BEST_PARAMS_FILE, 'r') as f:
                params = json.load(f)
            print(f"📂 Loaded best params from previous run: {BEST_PARAMS_FILE}")
            return params
        except Exception as e:
            print(f"⚠️  Could not load params from {BEST_PARAMS_FILE}: {e}")
            return None
    return None

def save_best_params(params):
    """Save best params to file"""
    try:
        with open(BEST_PARAMS_FILE, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"💾 Saved best params to {BEST_PARAMS_FILE}")
    except Exception as e:
        print(f"⚠️  Could not save params to {BEST_PARAMS_FILE}: {e}")

def proba_to_logits(proba):
    """Convert probabilities to logits (raw values for baseline)"""
    epsilon = 1e-15
    proba = np.clip(proba, epsilon, 1 - epsilon)
    return np.log(proba / (1 - proba))

def objective_cat(trial, X, y, cat_features_indices, X_original=None, y_original=None, 
                 use_string_categorical=False, feature_cols=None):
    params = {
        'iterations': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-2, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'eval_metric': 'AUC',
        'random_seed': config.SEED,
        'verbose': 0,
        'allow_writing_files': False
    }
    
    # Use all folds for accurate evaluation
    folds = get_folds(X, y)
    scores = []
    
    try:
        for train_idx, val_idx in folds:
            X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Ensure no NaN values remain (especially important for string categorical mode)
            if use_string_categorical:
                # Double-check: fill any remaining NaN
                for col in X_tr.columns:
                    if X_tr[col].isna().any():
                        X_tr[col] = X_tr[col].fillna('Missing')
                for col in X_val.columns:
                    if X_val[col].isna().any():
                        X_val[col] = X_val[col].fillna('Missing')
            
            # Add original dataset to training if available (for Optuna objective)
            if X_original is not None and y_original is not None:
                X_tr = pd.concat([X_tr, X_original], axis=0)
                y_tr = pd.concat([y_tr, y_original])
            
            # Use column names for cat_features if all columns are treated as categorical strings
            if use_string_categorical and feature_cols is not None:
                model = CatBoostClassifier(**params, cat_features=feature_cols)
            else:
                # cat_features_indices might be None if use_string_categorical is True
                if cat_features_indices is not None:
                    model = CatBoostClassifier(**params, cat_features=cat_features_indices)
                else:
                    # Fallback: use all columns as categorical
                    model = CatBoostClassifier(**params, cat_features=list(range(X_tr.shape[1])))
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=30)
            scores.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
            
        return np.mean(scores)
    except KeyboardInterrupt:
        # Re-raise to be caught by outer handler
        raise
    except Exception as e:
        # For any other error, return a bad score so Optuna skips this trial
        return 0.0

def run_cat():
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
        
        # Заполняем NaN в категориальных признаках для CatBoost
        # CatBoost требует, чтобы категориальные признаки были string или int, не NaN
        for c in all_cat_cols:
            if c in X_original.columns:
                X_original[c] = X_original[c].fillna('Missing').astype(str)
        
        print(f"✅ Original dataset: {len(original)} samples will be added to training")
    else:
        X_original = None
        y_original = None
    
    # Check if we should use fixed parameters
    if config.USE_OWN_PARAMETERS:
        # Use fixed hyperparameters (high-performance configuration)
        # IMPORTANT: No early stopping, use full iterations (like train_cat_2nd.py)
        cat_params_list = [{
            'iterations': 1800,
            'learning_rate': 0.05,
            'depth': 8,
            'random_strength': 0.79,
            'bagging_temperature': 0.6,
            'l2_leaf_reg': 4,
            'rsm': 0.6,
            'min_data_in_leaf': 5,
            'eval_metric': 'AUC',
            'random_seed': 1,  # Fixed seed for reproducibility
            'verbose': 100,
            'allow_writing_files': False,
            # No early_stopping_rounds - train full 1800 iterations
        }]
        print("📌 Using fixed hyperparameters (no early stopping, full 1800 iterations)")
    else:
        # Try to load from JSON, otherwise use defaults
        loaded_params = load_best_params() if config.USE_SAVED_PARAMS else None
        if loaded_params:
            cat_params_list = [{
                'iterations': 1000,
                'eval_metric': 'AUC',
                'random_seed': config.SEED,
                'verbose': 0,
                'allow_writing_files': False,
                **loaded_params
            }]
            print("📌 Using parameters from best_cat_params.json")
        else:
            cat_params_list = [{
                'iterations': 1000, 'learning_rate': 0.05, 'depth': 6, 
                'eval_metric': 'AUC', 'random_seed': config.SEED, 'verbose': 0,
                'allow_writing_files': False
            }]
            print("📌 Using default parameters")
    
    # All columns as categorical (convert all to string) - high-performance mode
    # This approach treats all features as categorical, which can improve performance
    if config.USE_OWN_PARAMETERS and config.USE_ORIGINAL_DATASET:
        print("🔧 Data preparation: all columns as categorical (string conversion)")
        # Convert all columns to string for categorical treatment
        feature_cols = [col for col in X.columns if col not in [id_col, target_col]]
        
        # Convert train, test, and original to string
        # IMPORTANT: Fill NaN before converting to string, otherwise NaN becomes 'nan' string
        X = X.copy()
        X_test = X_test.copy()
        for col in feature_cols:
            # Fill NaN first, then convert to string
            X[col] = X[col].fillna('Missing').astype(str)
            X_test[col] = X_test[col].fillna('Missing').astype(str)
        
        if X_original is not None:
            X_original = X_original.copy()
            for col in feature_cols:
                if col in X_original.columns:
                    X_original[col] = X_original[col].fillna('Missing').astype(str)
        
        # All columns are categorical for CatBoost
        cat_features_indices = list(range(len(feature_cols)))
        all_cat_cols = feature_cols  # Update for consistency
    else:
        # Standard approach: use categorical copies
        cat_features_indices = [X.columns.get_loc(c) for c in all_cat_cols if c in X.columns]

    # Skip Optuna if using fixed parameters
    if config.USE_OPTUNA and not config.USE_OWN_PARAMETERS:
        # Use CatBoost-specific settings
        timeout = config.CAT_OPTUNA_TIMEOUT if config.USE_CAT_TIMEOUT else None
        n_trials = config.CAT_OPTUNA_TRIALS if not config.USE_CAT_TIMEOUT else config.N_TRIALS
        
        if timeout:
            print(f"\nRunning Optuna Optimization for CatBoost (Timeout: {timeout}s = {timeout/60:.1f} mins)...")
        else:
            print(f"\nRunning Optuna Optimization for CatBoost (Max {n_trials} trials)...")
        print(f"Using all {config.N_SPLITS} folds for accurate evaluation")
        print("Press Ctrl+C to stop early and use best found params")
        
        study = optuna.create_study(direction='maximize')
        
        # Load best params from previous run and use as starting point (if enabled)
        previous_best = None
        if config.USE_SAVED_PARAMS:
            previous_best = load_best_params()
            if previous_best:
                print(f"🚀 Using previous best params as starting point (trial 0)")
                study.enqueue_trial(previous_best)
            else:
                print(f"🎲 Starting with random parameters (no saved params found)")
        else:
            print(f"🎲 Starting with random parameters (USE_SAVED_PARAMS=False)")
        
        try:
            # Determine if we should use string categorical mode for Optuna
            use_string_cat_mode = config.USE_OWN_PARAMETERS and config.USE_ORIGINAL_DATASET
            feature_cols_for_optuna = feature_cols if use_string_cat_mode else None
            
            if timeout:
                study.optimize(
                    lambda trial: objective_cat(trial, X, y, cat_features_indices, X_original, y_original, 
                                               use_string_categorical=use_string_cat_mode, feature_cols=feature_cols_for_optuna),
                    n_trials=config.N_TRIALS, 
                    timeout=timeout, 
                    show_progress_bar=True
                )
            else:
                study.optimize(
                    lambda trial: objective_cat(trial, X, y, cat_features_indices, X_original, y_original,
                                               use_string_categorical=use_string_cat_mode, feature_cols=feature_cols_for_optuna),
                    n_trials=n_trials,
                    show_progress_bar=True
                )
        except KeyboardInterrupt:
            print("\n\n⚠️  Optimization interrupted by user (Ctrl+C)")
            print("Using best params found so far...")
        
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials) == 0:
            print("⚠️  No completed trials found! Using default params.")
        else:
            completed_trials.sort(key=lambda t: t.value, reverse=True)
            
            # If we have fewer completed trials than N_TOP_MODELS, duplicate the best one
            if len(completed_trials) < config.N_TOP_MODELS:
                # Use all available trials and duplicate the best one to reach N_TOP_MODELS
                top_trials = completed_trials.copy()
                while len(top_trials) < config.N_TOP_MODELS:
                    top_trials.append(completed_trials[0])  # Duplicate best
            else:
                top_trials = completed_trials[:config.N_TOP_MODELS]
            
            print(f"\n✅ Completed {len(completed_trials)} trials")
            print(f"🏆 Best CV Score: {completed_trials[0].value:.5f}")
            print(f"📊 Best params: {completed_trials[0].params}")
            
            cat_params_list = []
            for t in top_trials:
                base = {
                    'iterations': 1000, 'eval_metric': 'AUC', 'random_seed': config.SEED, 
                    'verbose': 0, 'allow_writing_files': False
                }
                base.update(t.params)
                cat_params_list.append(base)
                
            print(f"✅ Top {len(cat_params_list)} Cat Params selected (using {len(completed_trials)} unique + {len(cat_params_list) - len(completed_trials)} duplicates).")
            
            # Save best params for next run and refinement step
            best_params_dict = completed_trials[0].params.copy()
            save_best_params(best_params_dict)
            np.save('best_cat_params.npy', cat_params_list[0])  # For stacking.py
    elif config.USE_OWN_PARAMETERS:
        print("⏭️  Skipping Optuna (using fixed hyperparameters)")
    
    # Train final model with selected params (Optuna or default)
    print("\n" + "="*60)
    print(f"Starting Final Training CatBoost (Full CV)...")
    print(f"Using {len(cat_params_list)} model(s) in ensemble")
    print(f"Feature engineering: {'Categorical copies ENABLED' if config.ADD_CATEGORICAL_COPIES else 'Categorical copies DISABLED (raw features only)'}")
    print(f"Total categorical features: {len(all_cat_cols)}")
    if config.USE_PSEUDO_LABELING:
        print(f"✅ Pseudo-labeling: ENABLED (using public test as training data)")
        print(f"   Pseudo-label threshold: {config.PSEUDO_THRESHOLD}")
    if len(cat_params_list) > 0:
        first_params = cat_params_list[0]
        print(f"Key params: learning_rate={first_params.get('learning_rate', 'N/A'):.4f}, "
              f"depth={first_params.get('depth', 'N/A')}, "
              f"l2_leaf_reg={first_params.get('l2_leaf_reg', 'N/A'):.4f}")
    print("="*60)
    
    # Use fixed random_state for reproducibility when using fixed parameters
    if config.USE_OWN_PARAMETERS:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=1)  # Fixed seed for reproducibility
        folds = list(skf.split(X, y))
        print(f"📌 Using random_state=1 for CV splits (fixed seed)")
    else:
        folds = get_folds(train.drop(columns=[target_col, id_col]), train[target_col])
    
    final_oof = np.zeros(len(train))
    final_test = np.zeros(len(test))
    final_fold_scores = []
    
    for fold_num, (train_idx, val_idx) in enumerate(tqdm(folds, desc="CV Folds"), 1):
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Ensure no NaN values remain (especially important for string categorical mode)
        if config.USE_OWN_PARAMETERS and config.USE_ORIGINAL_DATASET:
            # Double-check: fill any remaining NaN (shouldn't happen, but safety check)
            for col in X_tr.columns:
                if X_tr[col].isna().any():
                    X_tr[col] = X_tr[col].fillna('Missing')
            for col in X_val.columns:
                if X_val[col].isna().any():
                    X_val[col] = X_val[col].fillna('Missing')
        
        # Add original dataset to training if available
        if X_original is not None and y_original is not None:
            # X_original уже обработан (NaN заполнены) при подготовке в run_cat()
            X_tr = pd.concat([X_tr, X_original], axis=0)
            y_tr = pd.concat([y_tr, y_original])
            
            # Final safety check: ensure no NaN after concatenation
            if config.USE_OWN_PARAMETERS and config.USE_ORIGINAL_DATASET:
                for col in X_tr.columns:
                    if X_tr[col].isna().any():
                        X_tr[col] = X_tr[col].fillna('Missing')
            
            # Final safety check: ensure no NaN after concatenation
            if config.USE_OWN_PARAMETERS and config.USE_ORIGINAL_DATASET:
                for col in X_tr.columns:
                    if X_tr[col].isna().any():
                        X_tr[col] = X_tr[col].fillna('Missing')
        
        fold_oof = np.zeros(len(val_idx))
        fold_test = np.zeros(len(test))
        
        for params in cat_params_list:
            # Step 1: Train initial model (for pseudo-labeling if enabled)
            if config.USE_PSEUDO_LABELING:
                # Train initial model to get pseudo-labels
                initial_params = params.copy()
                initial_params['iterations'] = min(500, initial_params.get('iterations', 1000))  # Faster for pseudo-labeling
                initial_params['verbose'] = False
                # Use column names for cat_features if all columns are treated as categorical strings
                if config.USE_OWN_PARAMETERS and config.USE_ORIGINAL_DATASET and feature_cols is not None:
                    model_initial = CatBoostClassifier(**initial_params, cat_features=feature_cols)
                else:
                    model_initial = CatBoostClassifier(**initial_params, cat_features=cat_features_indices)
                model_initial.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
                
                # Get pseudo-labels from public test
                pseudo_proba = model_initial.predict_proba(X_test)[:, 1]
                pseudo_labels = (pseudo_proba >= config.PSEUDO_THRESHOLD).astype(int)
                
                # Filter by confidence if threshold is not 0.5
                if config.PSEUDO_THRESHOLD != 0.5:
                    confidence_mask = (pseudo_proba >= 0.7) | (pseudo_proba <= 0.3)
                    X_test_pseudo = X_test[confidence_mask]
                    y_test_pseudo = pd.Series(pseudo_labels[confidence_mask], index=X_test_pseudo.index)
                else:
                    # Use all pseudo-labels
                    X_test_pseudo = X_test.copy()
                    y_test_pseudo = pd.Series(pseudo_labels, index=X_test_pseudo.index)
                
                # Add pseudo-labeled test to training
                X_tr_final = pd.concat([X_tr, X_test_pseudo], axis=0)
                y_tr_final = pd.concat([y_tr, y_test_pseudo])
                
                # Final safety check: ensure no NaN after concatenation with pseudo-labels
                if config.USE_OWN_PARAMETERS and config.USE_ORIGINAL_DATASET:
                    for col in X_tr_final.columns:
                        if X_tr_final[col].isna().any():
                            X_tr_final[col] = X_tr_final[col].fillna('Missing')
            else:
                X_tr_final = X_tr
                y_tr_final = y_tr
            
            # Step 2: Train final model
            # Use column names for cat_features if all columns are treated as categorical strings
            if config.USE_OWN_PARAMETERS and config.USE_ORIGINAL_DATASET and feature_cols is not None:
                model = CatBoostClassifier(**params, cat_features=feature_cols)
            else:
                # Use indices for cat_features (standard approach)
                model = CatBoostClassifier(**params, cat_features=cat_features_indices)
            # No early stopping when using fixed parameters (like train_cat_2nd.py)
            if config.USE_OWN_PARAMETERS:
                # Train full iterations without early stopping
                model.fit(X_tr_final, y_tr_final, eval_set=(X_val, y_val), verbose=100)
            else:
                # Use early stopping for Optuna-found parameters
                model.fit(X_tr_final, y_tr_final, eval_set=(X_val, y_val), early_stopping_rounds=50)
            fold_oof += model.predict_proba(X_val)[:, 1] / len(cat_params_list)
            fold_test += model.predict_proba(X_test)[:, 1] / len(cat_params_list)
        
        final_oof[val_idx] = fold_oof
        final_test += fold_test / config.N_SPLITS
        
        fold_score = roc_auc_score(y_val, fold_oof)
        final_fold_scores.append(fold_score)
    
    final_score = roc_auc_score(y, final_oof)
    print("\n" + "="*60)
    print(f"🎯 First CatBoost CV Score: {final_score:.5f}")
    print(f"📈 Fold scores: {[f'{s:.5f}' for s in final_fold_scores]}")
    print(f"📊 Mean fold score: {np.mean(final_fold_scores):.5f} ± {np.std(final_fold_scores):.5f}")
    print(f"🔧 Features: {'With categorical copies' if config.ADD_CATEGORICAL_COPIES else 'Raw features only'} ({len(all_cat_cols)} cat features)")
    if config.USE_PSEUDO_LABELING:
        print(f"✅ Pseudo-labeling: Used public test ({len(test):,} samples) as training data")
    print("="*60)
    
    # CatBoost on CatBoost (refinement with baseline)
    if config.USE_CATBOOST_ON_CATBOOST:
        print("\n" + "="*60)
        print("🔧 CatBoost on CatBoost: Training second CatBoost with first CatBoost as baseline...")
        print("="*60)
        
        # Convert probabilities to logits (raw values for baseline)
        baseline_oof_logits = proba_to_logits(final_oof)
        baseline_test_logits = proba_to_logits(final_test)
        
        # Get best params for refinement (use same as first model or load from file)
        try:
            refine_params = np.load('best_cat_params.npy', allow_pickle=True).item()
        except:
            refine_params = cat_params_list[0] if len(cat_params_list) > 0 else {
                'iterations': 1000, 'learning_rate': 0.05, 'depth': 6, 
                'eval_metric': 'AUC', 'random_seed': config.SEED, 'verbose': 0,
                'allow_writing_files': False
            }
        
        refined_oof = np.zeros(len(train))
        refined_test = np.zeros(len(test))
        refined_fold_scores = []
        
        for fold_num, (train_idx, val_idx) in enumerate(tqdm(folds, desc="Refinement CV"), 1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Prepare baselines for this fold
            train_baseline = baseline_oof_logits[train_idx]
            val_baseline = baseline_oof_logits[val_idx]
            test_baseline = baseline_test_logits
            
            # Create pools with baseline
            # Use column names for cat_features if all columns are treated as categorical strings
            if config.USE_OWN_PARAMETERS and config.USE_ORIGINAL_DATASET and feature_cols is not None:
                cat_features_for_pool = feature_cols
            else:
                cat_features_for_pool = cat_features_indices
            
            train_pool = Pool(X_tr, y_tr, cat_features=cat_features_for_pool, baseline=train_baseline)
            val_pool = Pool(X_val, y_val, cat_features=cat_features_for_pool, baseline=val_baseline)
            test_pool = Pool(X_test, cat_features=cat_features_for_pool, baseline=test_baseline)
            
            # Train refinement model
            refine_model = CatBoostClassifier(**refine_params, cat_features=cat_features_for_pool)
            refine_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)
            
            # Get refined predictions (model returns residuals, need to add baseline)
            refined_oof[val_idx] = refine_model.predict_proba(val_pool)[:, 1]
            refined_test += refine_model.predict_proba(test_pool)[:, 1] / config.N_SPLITS
            
            fold_score = roc_auc_score(y_val, refined_oof[val_idx])
            refined_fold_scores.append(fold_score)
        
        refined_score = roc_auc_score(y, refined_oof)
        improvement = refined_score - final_score
        improvement_pct = (improvement / final_score) * 100
        
        print("\n" + "="*60)
        print(f"🎯 Refined CatBoost CV Score: {refined_score:.5f}")
        print(f"📈 Fold scores: {[f'{s:.5f}' for s in refined_fold_scores]}")
        print(f"📊 Mean fold score: {np.mean(refined_fold_scores):.5f} ± {np.std(refined_fold_scores):.5f}")
        print(f"\n{'✅' if improvement > 0 else '❌'} Improvement: {improvement:+.5f} ({improvement_pct:+.3f}%)")
        if improvement > 0:
            print(f"🎉 CatBoost on CatBoost improved the model!")
        else:
            print(f"⚠️  CatBoost on CatBoost did not improve (using first model results)")
        print("="*60)
        
        # Use refined predictions if better
        if improvement > 0:
            final_oof = refined_oof
            final_test = refined_test
            final_score = refined_score
            final_fold_scores = refined_fold_scores
            print("✅ Using refined CatBoost predictions")
        else:
            print("✅ Using first CatBoost predictions (refinement didn't help)")
    
    save_predictions('cat', final_oof, final_test)
    
    # Save CSV submission file
    try:
        sub = pd.read_csv('sample_submission.csv')
        sub[target_col] = final_test
        sub.to_csv('submission_cat.csv', index=False)
        print(f"\n✅ Submission saved to submission_cat.csv")
    except FileNotFoundError:
        print("\n⚠️  sample_submission.csv not found, skipping CSV export")

if __name__ == "__main__":
    run_cat()

