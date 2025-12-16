# Configuration settings for the project

N_SPLITS = 5
SEED = 42
USE_OPTUNA = True
N_TRIALS = 1000    # High number so timeout controls the duration
OPTUNA_TIMEOUT = 1000 # Seconds per model optimization (~10 mins)
N_TOP_MODELS = 3   # Average top N models from Optuna study

# CatBoost specific settings (for quick testing)
CAT_OPTUNA_TIMEOUT = 1800  # 30 minutes for CatBoost
CAT_OPTUNA_TRIALS = 50    # Or limit by number of trials instead
USE_CAT_TIMEOUT = True    # If True, use timeout; if False, use N_TRIALS limit

# Feature engineering settings
ADD_CATEGORICAL_COPIES = True  # If True, create categorical copies of numerical features
HANDLE_OUTLIERS = False  # If True, clip outliers using IQR method (1.5 * IQR)
ADD_POLYNOMIAL_FEATURES = False  # If True, create polynomial features (interactions, squares, ratios)
ADD_KAGGLE_FEATURES = False  # If True, add features from Kaggle comments (interest_payment_to_income, loan_to_income_precise)

# CatBoost on CatBoost (refinement with baseline)
USE_CATBOOST_ON_CATBOOST = False  # If True, train second CatBoost using first CatBoost predictions as baseline

# Hyperparameter loading
USE_SAVED_PARAMS = False  # If True, load best params from JSON file (warm start); if False, start with default/random params

# Pseudo-labeling (using public test as training data)
USE_PSEUDO_LABELING = False  # If True, use pseudo-labeling from public test (valid for Kaggle with public/private split)
PSEUDO_THRESHOLD = 0.7  # Confidence threshold for pseudo-labels (0.5 = use all, 0.7 = only confident)

# 2nd place solution specific settings
USE_OWN_PARAMETERS = True  # If True, use fixed hyperparameters from script; if False, load from best_cat_params.json

# Original dataset (hack dataset) usage
USE_ORIGINAL_DATASET = False  # If True, add credit_risk_dataset.csv to training data in each fold (improves metrics)

# Ensemble settings
ENSEMBLE_MIN_SCORE_THRESHOLD = 0.95  # Minimum CV score for models to be included in ensemble (filters out weak models)
USE_MULTI_SEED_CAT = True  # If True, use multi-seed CatBoost ensemble (requires train_cat_multi_seed.py to be run first)

# If set, keep only top-K models by individual CV score (after threshold filter).
# Example: 3 means keep only the best 3 models (e.g. CAT, CAT_MULTI_SEED, LGBM).
# Set to None to disable.
ENSEMBLE_TOP_K = None


