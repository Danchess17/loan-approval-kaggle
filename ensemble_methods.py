"""
Ансамблирование моделей различными методами:
1. Блендинг (простое среднее)
2. Стекинг (мета-модель)
3. Взвешенное среднее (оптимизация весов)
4. Hill Climbing (оптимизация весов)

Загружает предсказания из сохраненных файлов и применяет различные методы ансамбля
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import time
import config
from data_loader import load_and_preprocess
from scipy.optimize import minimize

def load_predictions(include_nn=False, include_multi_seed_cat=True):
    """Загружает OOF и test predictions из сохраненных файлов"""
    print("📂 Загрузка предсказаний моделей...")
    
    models = ['lgbm', 'xgb', 'cat']
    if include_nn:
        models.append('nn')
    
    oof_preds = {}
    test_preds = {}
    
    for m in models:
        try:
            oof_file = f'oof_{m}.npy'
            test_file = f'test_{m}.npy'
            oof_preds[m] = np.load(oof_file)
            test_preds[m] = np.load(test_file)
            print(f"   ✅ {m.upper()}: загружено")
        except FileNotFoundError:
            print(f"   ⚠️  {m.upper()}: файлы не найдены (oof_{m}.npy, test_{m}.npy)")
    
    # Load multi-seed CatBoost if available
    if include_multi_seed_cat:
        try:
            oof_multi = np.load('oof_cat_multi_seed.npy')
            test_multi = np.load('test_cat_multi_seed.npy')
            oof_preds['cat_multi_seed'] = oof_multi
            test_preds['cat_multi_seed'] = test_multi
            print(f"   ✅ CAT_MULTI_SEED: загружено (multi-seed ensemble)")
        except FileNotFoundError:
            print(f"   ⚠️  CAT_MULTI_SEED: не найдено (запустите train_cat_multi_seed.py)")
    
    if not oof_preds:
        raise ValueError("Не найдено ни одного файла с предсказаниями! Сначала обучите модели.")
    
    return oof_preds, test_preds

def simple_blending(oof_preds_dict, test_preds_dict):
    """Простое блендинг (среднее арифметическое)"""
    print("\n📊 Метод 1: Simple Blending (среднее арифметическое)")
    
    oof_blend = np.mean(list(oof_preds_dict.values()), axis=0)
    test_blend = np.mean(list(test_preds_dict.values()), axis=0)
    
    return oof_blend, test_blend

def weighted_average(oof_preds_dict, test_preds_dict, y):
    """Взвешенное среднее с оптимизацией весов"""
    print("\n📊 Метод 2: Weighted Average (оптимизация весов)")
    
    def objective(weights):
        weights = weights / np.sum(weights)  # Нормализация
        blended = np.zeros(len(y))
        for i, (model_name, preds) in enumerate(oof_preds_dict.items()):
            blended += weights[i] * preds
        return -roc_auc_score(y, blended)  # Минимизируем отрицательный AUC
    
    # Начальные веса (равномерные)
    n_models = len(oof_preds_dict)
    initial_weights = np.ones(n_models) / n_models
    
    # Ограничения: веса >= 0, сумма = 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_models)]
    
    print("   Оптимизация весов...")
    result = minimize(objective, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints, options={'maxiter': 100})
    
    optimal_weights = result.x / np.sum(result.x)
    
    print(f"   Оптимальные веса:")
    for i, (model_name, _) in enumerate(oof_preds_dict.items()):
        print(f"      {model_name.upper()}: {optimal_weights[i]:.4f}")
    
    # Применяем веса
    oof_weighted = np.zeros(len(y))
    test_weighted = np.zeros(len(test_preds_dict[list(test_preds_dict.keys())[0]]))
    
    for i, (model_name, preds) in enumerate(oof_preds_dict.items()):
        oof_weighted += optimal_weights[i] * preds
        test_weighted += optimal_weights[i] * test_preds_dict[model_name]
    
    return oof_weighted, test_weighted, optimal_weights

def stacking(oof_preds_dict, test_preds_dict, y, meta_model_type='lgbm'):
    """
    Честный стекинг: мета-модель обучается out-of-fold.
    Иначе (если обучить мету на всех OOF и оценить на них же) CV завышается и часто падает на private.
    """
    meta_model_name = "LightGBM" if meta_model_type == 'lgbm' else "Logistic Regression"
    print(f"\n📊 Метод 3: Stacking (мета-модель: {meta_model_name}, OOF для меты)")
    
    model_names = list(oof_preds_dict.keys())
    meta_features = np.column_stack([oof_preds_dict[m] for m in model_names])
    test_meta_features = np.column_stack([test_preds_dict[m] for m in model_names])
    
    skf = StratifiedKFold(n_splits=getattr(config, "N_SPLITS", 5), shuffle=True, random_state=1)
    meta_oof = np.zeros(len(y))
    meta_test = np.zeros(test_meta_features.shape[0])
    
    importance_sum = np.zeros(len(model_names), dtype=float)
    
    print("   Обучение мета-модели по фолдам...")
    for tr_idx, va_idx in skf.split(meta_features, y):
        X_tr, X_va = meta_features[tr_idx], meta_features[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        
        if meta_model_type == 'lgbm':
            meta_model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=5,
                reg_lambda=1.0,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                metric='auc',
                verbosity=-1,
                random_state=1
            )
            meta_model.fit(X_tr, y_tr)
            importance_sum += meta_model.feature_importances_
        else:
            meta_model = LogisticRegression(max_iter=2000, random_state=1, C=0.5)
            meta_model.fit(X_tr, y_tr)
            importance_sum += meta_model.coef_[0]
        
        meta_oof[va_idx] = meta_model.predict_proba(X_va)[:, 1]
        meta_test += meta_model.predict_proba(test_meta_features)[:, 1] / skf.get_n_splits()
    
    print("   Важность/коэффициенты мета-модели (сумма по фолдам):")
    for i, name in enumerate(model_names):
        print(f"      {name.upper():<20}: {importance_sum[i]:.2f}")
    
    return meta_oof, meta_test

def hill_climbing_blend(oof_preds_dict, test_preds_dict, y):
    """Hill Climbing для оптимизации весов"""
    print("\n📊 Метод 4: Hill Climbing Blend")
    
    n_models = len(oof_preds_dict)
    weights = np.ones(n_models) / n_models
    
    best_score = -roc_auc_score(y, np.mean(list(oof_preds_dict.values()), axis=0))
    best_weights = weights.copy()
    
    # Hill climbing
    step_size = 0.01
    max_iterations = 1000
    no_improvement = 0
    
    print("   Оптимизация весов (Hill Climbing)...")
    for iteration in tqdm(range(max_iterations), desc="   Итерации", leave=False):
        improved = False
        
        for i in range(n_models):
            # Пробуем увеличить вес i-й модели
            new_weights = weights.copy()
            new_weights[i] += step_size
            
            # Перераспределяем остальные веса
            other_weights_sum = np.sum(new_weights) - new_weights[i]
            if other_weights_sum > 0:
                for j in range(n_models):
                    if j != i:
                        new_weights[j] = new_weights[j] * (1 - new_weights[i]) / other_weights_sum
            
            new_weights = new_weights / np.sum(new_weights)  # Нормализация
            
            # Вычисляем score
            blended = np.zeros(len(y))
            for k, (_, preds) in enumerate(oof_preds_dict.items()):
                blended += new_weights[k] * preds
            
            score = -roc_auc_score(y, blended)
            
            if score < best_score:
                best_score = score
                best_weights = new_weights.copy()
                weights = new_weights.copy()
                improved = True
                no_improvement = 0
                break
        
        if not improved:
            no_improvement += 1
            if no_improvement >= 10:
                break
    
    print(f"   Оптимальные веса (Hill Climbing):")
    for i, model_name in enumerate(oof_preds_dict.keys()):
        print(f"      {model_name.upper()}: {best_weights[i]:.4f}")
    
    # Применяем веса
    oof_hill = np.zeros(len(y))
    test_hill = np.zeros(len(test_preds_dict[list(test_preds_dict.keys())[0]]))
    
    for i, (model_name, preds) in enumerate(oof_preds_dict.items()):
        oof_hill += best_weights[i] * preds
        test_hill += best_weights[i] * test_preds_dict[model_name]
    
    return oof_hill, test_hill, best_weights

def run_ensemble():
    """Основная функция для ансамблирования"""
    print("\n" + "="*70)
    print("🎯 АНСАМБЛИРОВАНИЕ МОДЕЛЕЙ")
    print("="*70)
    
    start_time = time.time()
    
    # Загружаем данные для получения y
    result = load_and_preprocess()
    if len(result) == 8:
        train, test, original, num_cols, cat_cols, all_cat_cols, target_col, id_col = result
    else:
        # Fallback for old version
        train, test, num_cols, cat_cols, all_cat_cols, target_col, id_col = result
    y = train[target_col]
    
    # Загружаем предсказания (без NN, с multi-seed CatBoost если доступен)
    use_multi_seed = getattr(config, 'USE_MULTI_SEED_CAT', True)
    oof_preds_dict, test_preds_dict = load_predictions(include_nn=False, include_multi_seed_cat=use_multi_seed)
    
    # Вычисляем качество отдельных моделей
    print("\n" + "="*70)
    print("📊 РЕЗУЛЬТАТЫ ОТДЕЛЬНЫХ МОДЕЛЕЙ")
    print("="*70)
    
    model_scores = {}
    for model_name, oof_preds in oof_preds_dict.items():
        score = roc_auc_score(y, oof_preds)
        model_scores[model_name.upper()] = score
        print(f"   {model_name.upper():<20}: {score:.5f}")
    
    # Filter out weak models (optional - can be enabled via config)
    if hasattr(config, 'ENSEMBLE_MIN_SCORE_THRESHOLD'):
        min_score_threshold = config.ENSEMBLE_MIN_SCORE_THRESHOLD
    else:
        min_score_threshold = 0.95  # Default threshold
    
    filtered_models = {k: v for k, v in oof_preds_dict.items() 
                      if model_scores[k.upper()] >= min_score_threshold}
    
    if len(filtered_models) < len(oof_preds_dict):
        removed = set(oof_preds_dict.keys()) - set(filtered_models.keys())
        print(f"\n⚠️  Отфильтрованы слабые модели (CV < {min_score_threshold}): {', '.join(removed)}")
        print(f"   Используется {len(filtered_models)} из {len(oof_preds_dict)} моделей")
        oof_preds_dict = filtered_models
        test_preds_dict = {k: test_preds_dict[k] for k in filtered_models.keys()}

    # Keep only top-K models by score (optional)
    top_k = getattr(config, "ENSEMBLE_TOP_K", None)
    if top_k is not None:
        try:
            top_k = int(top_k)
        except Exception:
            top_k = None
    if top_k is not None and top_k > 0 and len(oof_preds_dict) > top_k:
        # Sort by individual model CV (desc), tie-break by name for determinism
        sorted_models = sorted(
            oof_preds_dict.keys(),
            key=lambda m: (model_scores[m.upper()], m),
            reverse=True
        )
        keep = sorted_models[:top_k]
        removed = [m for m in oof_preds_dict.keys() if m not in keep]
        print(f"\n📌 Выбраны TOP-{top_k} модели по CV: {', '.join([m.upper() for m in keep])}")
        if removed:
            print(f"   Удалены: {', '.join([m.upper() for m in removed])}")
        oof_preds_dict = {k: oof_preds_dict[k] for k in keep}
        test_preds_dict = {k: test_preds_dict[k] for k in keep}
    
    # Применяем различные методы ансамбля
    ensemble_results = {}
    
    # 1. Simple Blending
    oof_blend, test_blend = simple_blending(oof_preds_dict, test_preds_dict)
    score_blend = roc_auc_score(y, oof_blend)
    ensemble_results['Simple Blending'] = {'oof': oof_blend, 'test': test_blend, 'score': score_blend}
    print(f"   CV Score: {score_blend:.5f}")
    
    # 2. Weighted Average
    oof_weighted, test_weighted, weights = weighted_average(oof_preds_dict, test_preds_dict, y)
    score_weighted = roc_auc_score(y, oof_weighted)
    ensemble_results['Weighted Average'] = {'oof': oof_weighted, 'test': test_weighted, 'score': score_weighted, 'weights': weights}
    print(f"   CV Score: {score_weighted:.5f}")
    
    # 3. Stacking (LightGBM)
    oof_stack, test_stack = stacking(oof_preds_dict, test_preds_dict, y, meta_model_type='lgbm')
    score_stack = roc_auc_score(y, oof_stack)
    ensemble_results['Stacking (LightGBM)'] = {'oof': oof_stack, 'test': test_stack, 'score': score_stack}
    print(f"   CV Score: {score_stack:.5f}")
    
    # 3b. Stacking (Logistic Regression) - for comparison
    oof_stack_lr, test_stack_lr = stacking(oof_preds_dict, test_preds_dict, y, meta_model_type='lr')
    score_stack_lr = roc_auc_score(y, oof_stack_lr)
    ensemble_results['Stacking (LR)'] = {'oof': oof_stack_lr, 'test': test_stack_lr, 'score': score_stack_lr}
    print(f"   CV Score: {score_stack_lr:.5f}")
    
    # 4. Hill Climbing
    oof_hill, test_hill, hill_weights = hill_climbing_blend(oof_preds_dict, test_preds_dict, y)
    score_hill = roc_auc_score(y, oof_hill)
    ensemble_results['Hill Climbing'] = {'oof': oof_hill, 'test': test_hill, 'score': score_hill, 'weights': hill_weights}
    print(f"   CV Score: {score_hill:.5f}")
    
    # Итоговые результаты
    print("\n" + "="*70)
    print("🏆 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*70)
    
    all_results = {**model_scores, **{k: v['score'] for k, v in ensemble_results.items()}}
    
    print(f"\n{'Модель/Метод':<25} {'CV Score':<12} {'Улучшение':<12}")
    print("-" * 50)
    
    best_single = max(model_scores.values())
    for name, score in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
        improvement = score - best_single
        status = "✅" if improvement > 0.0001 else "➖" if improvement > -0.0001 else ""
        print(f"{name:<25} {score:.5f}      {improvement:+.5f} {status}")
    
    best_method = max(all_results.items(), key=lambda x: x[1])
    print(f"\n🏆 Лучший метод: {best_method[0]} (CV Score: {best_method[1]:.5f})")
    
    # Сохраняем предсказания для КАЖДОГО ансамбль-метода (в т.ч. стекинг), плюс лучший как submission_ensemble.csv
    def _slug(name: str) -> str:
        return (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("__", "_")
        )
    
    print("\n💾 Сохранение submission-файлов для всех ансамблей...")
    for method_name, payload in ensemble_results.items():
        sub = pd.DataFrame({'id': test[id_col], 'loan_status': payload['test']})
        out_path = f"submission_{_slug(method_name)}.csv"
        sub.to_csv(out_path, index=False)
        print(f"   ✅ {method_name}: {out_path}")
    
    # Отдельно сохраняем лучший ансамбль по CV в submission_ensemble.csv (как и раньше)
    best_ensemble = max(ensemble_results.items(), key=lambda x: x[1]['score'])
    best_test_preds = best_ensemble[1]['test']
    submission = pd.DataFrame({'id': test[id_col], 'loan_status': best_test_preds})
    submission.to_csv('submission_ensemble.csv', index=False)
    print(f"\n🏆 Лучшие предсказания сохранены в: submission_ensemble.csv")
    print(f"   Метод: {best_ensemble[0]}")
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Общее время: {elapsed_time:.1f}s")
    
    return ensemble_results, model_scores

if __name__ == "__main__":
    run_ensemble()

