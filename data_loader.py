import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import config

def handle_outliers(train, test, num_cols):
    """
    Обработка выбросов методом IQR (Interquartile Range)
    Обрезает значения за пределами [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    """
    print("🔧 Outlier handling: Clipping outliers using IQR method (1.5 * IQR)...")
    
    # Объединяем train и test для расчета границ на всех данных
    combined = pd.concat([train[num_cols], test[num_cols]], axis=0)
    
    clip_bounds = {}
    for col in num_cols:
        Q1 = combined[col].quantile(0.25)
        Q3 = combined[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Сохраняем границы
        clip_bounds[col] = {'lower': lower_bound, 'upper': upper_bound}
        
        # Подсчитываем выбросы до обработки
        outliers_before_train = ((train[col] < lower_bound) | (train[col] > upper_bound)).sum()
        outliers_before_test = ((test[col] < lower_bound) | (test[col] > upper_bound)).sum()
        
        # Обрезаем выбросы
        train[col] = train[col].clip(lower=lower_bound, upper=upper_bound)
        test[col] = test[col].clip(lower=lower_bound, upper=upper_bound)
        
        if outliers_before_train > 0 or outliers_before_test > 0:
            print(f"   {col}: clipped {outliers_before_train} train + {outliers_before_test} test outliers "
                  f"(bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")
    
    return train, test

def add_kaggle_features(train, test, num_cols):
    """
    Добавляет фичи из Kaggle комментариев (430th и 268th place)
    - interest_payment_to_income: сколько процентов от дохода уходит на проценты
    - loan_to_income_precise: более точная версия loan_percent_income
    """
    print("🔧 Feature engineering: Adding Kaggle features...")
    
    new_num_cols = []
    
    # 1. interest_rate * loan_amount / income (Björn, 430th place)
    # Сколько процентов от дохода уходит на проценты
    if 'loan_int_rate' in num_cols and 'loan_amnt' in num_cols and 'person_income' in num_cols:
        train['interest_payment_to_income'] = (train['loan_int_rate'] * train['loan_amnt']) / (train['person_income'] + 1e-8)
        test['interest_payment_to_income'] = (test['loan_int_rate'] * test['loan_amnt']) / (test['person_income'] + 1e-8)
        new_num_cols.append('interest_payment_to_income')
        print("   ✅ interest_payment_to_income (interest_rate * loan_amount / income)")
    
    # 2. loan_to_income_precise (Turabi, 268th place)
    # Более точная версия loan_percent_income (оригинал имеет только 2 знака после запятой)
    if 'loan_amnt' in num_cols and 'person_income' in num_cols:
        train['loan_to_income_precise'] = train['loan_amnt'] / (train['person_income'] + 1e-8)
        test['loan_to_income_precise'] = test['loan_amnt'] / (test['person_income'] + 1e-8)
        new_num_cols.append('loan_to_income_precise')
        print("   ✅ loan_to_income_precise (более точная версия loan_percent_income)")
    
    # 3. loan_amount / (65 - age) (Björn, 430th place)
    # Сколько нужно платить в год до пенсии (опционально, показало нейтральный результат)
    # Закомментировано, т.к. не показало улучшения
    # if 'loan_amnt' in num_cols and 'person_age' in num_cols:
    #     remaining_years = (65 - train['person_age']).clip(lower=1)
    #     train['loan_per_remaining_year'] = train['loan_amnt'] / remaining_years
    #     remaining_years_test = (65 - test['person_age']).clip(lower=1)
    #     test['loan_per_remaining_year'] = test['loan_amnt'] / remaining_years_test
    #     new_num_cols.append('loan_per_remaining_year')
    #     print("   ✅ loan_per_remaining_year (loan_amount / (65 - age))")
    
    return train, test, new_num_cols

def add_polynomial_features(train, test, num_cols):
    """
    Генерация полиномиальных признаков на основе EDA анализа
    Создает взаимодействия, квадраты и отношения важных признаков
    """
    print("🔧 Feature engineering: Creating polynomial features...")
    
    new_num_cols = []
    
    # 1. Взаимодействия (умножение) - высокий приоритет из EDA
    interaction_pairs = [
        ('loan_percent_income', 'loan_int_rate'),
        ('loan_percent_income', 'person_income'),
        ('loan_percent_income', 'loan_amnt'),
        ('loan_int_rate', 'person_income'),
        ('loan_int_rate', 'loan_amnt'),
        ('person_income', 'loan_amnt'),
        ('loan_amnt', 'person_emp_length'),
        ('person_income', 'person_emp_length'),
    ]
    
    for feat1, feat2 in interaction_pairs:
        if feat1 in num_cols and feat2 in num_cols:
            new_col = f'{feat1}_x_{feat2}'
            train[new_col] = train[feat1] * train[feat2]
            test[new_col] = test[feat1] * test[feat2]
            new_num_cols.append(new_col)
            print(f"   ✅ {new_col} (interaction)")
    
    # 2. Отношения (деление) - высокий приоритет из EDA
    ratio_pairs = [
        ('loan_amnt', 'person_income'),  # Улучшенная версия loan_percent_income
        ('person_age', 'person_emp_length'),  # Стабильность карьеры
        ('cb_person_cred_hist_length', 'person_age'),  # Раннее начало кредитной истории
    ]
    
    for feat1, feat2 in ratio_pairs:
        if feat1 in num_cols and feat2 in num_cols:
            new_col = f'{feat1}_div_{feat2}'
            # Избегаем деления на ноль
            train[new_col] = train[feat1] / (train[feat2] + 1e-8)
            test[new_col] = test[feat1] / (test[feat2] + 1e-8)
            new_num_cols.append(new_col)
            print(f"   ✅ {new_col} (ratio)")
    
    # 3. Квадраты важных признаков (топ-3 по важности из EDA)
    square_features = [
        'loan_percent_income',
        'loan_int_rate',
        'person_income',
    ]
    
    for feat in square_features:
        if feat in num_cols:
            new_col = f'{feat}_squared'
            train[new_col] = train[feat] ** 2
            test[new_col] = test[feat] ** 2
            new_num_cols.append(new_col)
            print(f"   ✅ {new_col} (square)")
    
    print(f"   📊 Created {len(new_num_cols)} new polynomial features")
    
    # Обновляем список числовых признаков
    updated_num_cols = num_cols + new_num_cols
    
    return train, test, updated_num_cols

def load_original_dataset():
    """Загружает original dataset (credit_risk_dataset.csv) если доступен"""
    try:
        original = pd.read_csv('credit_risk_dataset.csv')
        print(f"✅ Original dataset loaded: {len(original)} samples")
        return original
    except FileNotFoundError:
        print("⚠️  Original dataset not found (credit_risk_dataset.csv)")
        return None

def load_and_preprocess():
    print("Loading data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    # Load original dataset if flag is enabled
    original = None
    if config.USE_ORIGINAL_DATASET:
        original = load_original_dataset()
    
    # Identify columns
    target_col = 'loan_status'
    id_col = 'id'
    
    # Separate features
    num_cols = [
        'person_age', 'person_income', 'person_emp_length', 
        'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
        'cb_person_cred_hist_length'
    ]
    cat_cols = [
        'person_home_ownership', 'loan_intent', 'loan_grade', 
        'cb_person_default_on_file'
    ]
    
    # Step 1: Handle outliers (before feature engineering)
    if config.HANDLE_OUTLIERS:
        train, test = handle_outliers(train, test, num_cols)
    else:
        print("🔧 Outlier handling: Disabled (using raw values)")
    
    # Step 2: Add Kaggle features (before polynomial features)
    if config.ADD_KAGGLE_FEATURES:
        train, test, kaggle_features = add_kaggle_features(train, test, num_cols)
        num_cols = num_cols + kaggle_features
    else:
        print("🔧 Kaggle features: Disabled")
    
    # Step 3: Add polynomial features (after Kaggle features)
    if config.ADD_POLYNOMIAL_FEATURES:
        train, test, num_cols = add_polynomial_features(train, test, num_cols)
    else:
        print("🔧 Polynomial features: Disabled")
    
    # Step 4: Create categorical copies of numerical features (if enabled)
    if config.ADD_CATEGORICAL_COPIES:
        print("🔧 Feature engineering: Creating categorical copies of numerical features...")
        for col in num_cols:
            train[f'{col}_cat'] = train[col].astype(str)
            test[f'{col}_cat'] = test[col].astype(str)
        
        # Update feature lists
        all_cat_cols = cat_cols + [f'{c}_cat' for c in num_cols]
    else:
        print("🔧 Feature engineering: Using raw features only (no categorical copies)...")
        # Use only original categorical columns
        all_cat_cols = cat_cols
    
    # Return original dataset if loaded
    if config.USE_ORIGINAL_DATASET and original is not None:
        return train, test, original, num_cols, cat_cols, all_cat_cols, target_col, id_col
    else:
        return train, test, None, num_cols, cat_cols, all_cat_cols, target_col, id_col

def get_folds(train, target):
    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    return list(skf.split(train, target))

def save_predictions(model_name, oof_preds, test_preds):
    np.save(f'oof_{model_name}.npy', oof_preds)
    np.save(f'test_{model_name}.npy', test_preds)
    print(f"Saved predictions for {model_name} to oof_{model_name}.npy and test_{model_name}.npy")

