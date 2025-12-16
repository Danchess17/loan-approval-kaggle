"""
Exploratory Data Analysis (EDA) для Loan Approval Prediction
Анализ датасета, выбросов, важности признаков, корреляций и предложения по feature engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Загрузка данных"""
    print("="*80)
    print("📊 ЗАГРУЗКА ДАННЫХ")
    print("="*80)
    
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"\nTrain columns: {list(train.columns)}")
    print(f"\nTrain info:")
    print(train.info())
    
    return train, test

def basic_statistics(train, test):
    """Базовая статистика по датасету"""
    print("\n" + "="*80)
    print("📈 БАЗОВАЯ СТАТИСТИКА")
    print("="*80)
    
    # Целевая переменная
    print("\n🎯 Распределение целевой переменной (loan_status):")
    print(train['loan_status'].value_counts())
    print(f"Процент положительного класса: {train['loan_status'].mean()*100:.2f}%")
    
    # Пропущенные значения
    print("\n❓ Пропущенные значения в train:")
    missing_train = train.isnull().sum()
    missing_test = test.isnull().sum()
    missing_df = pd.DataFrame({
        'Train': missing_train,
        'Test': missing_test,
        'Train_%': (missing_train / len(train) * 100).round(2),
        'Test_%': (missing_test / len(test) * 100).round(2)
    })
    print(missing_df[missing_df['Train'] > 0])
    
    # Числовые признаки
    num_cols = [
        'person_age', 'person_income', 'person_emp_length', 
        'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
        'cb_person_cred_hist_length'
    ]
    
    print("\n📊 Описательная статистика числовых признаков:")
    print(train[num_cols].describe())
    
    # Категориальные признаки
    cat_cols = [
        'person_home_ownership', 'loan_intent', 'loan_grade', 
        'cb_person_default_on_file'
    ]
    
    print("\n📊 Распределение категориальных признаков:")
    for col in cat_cols:
        print(f"\n{col}:")
        print(train[col].value_counts())
        print(f"Уникальных значений: {train[col].nunique()}")
    
    return num_cols, cat_cols

def analyze_outliers(train, num_cols):
    """Анализ выбросов"""
    print("\n" + "="*80)
    print("🔍 АНАЛИЗ ВЫБРОСОВ")
    print("="*80)
    
    outlier_results = {}
    
    for col in num_cols:
        print(f"\n📌 {col}:")
        data = train[col].dropna()
        
        # Базовые статистики
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Z-score метод
        z_scores = np.abs(stats.zscore(data))
        outliers_zscore = np.sum(z_scores > 3)
        
        # IQR метод
        outliers_iqr = np.sum((data < lower_bound) | (data > upper_bound))
        
        print(f"  Min: {data.min():.2f}, Max: {data.max():.2f}")
        print(f"  Mean: {data.mean():.2f}, Median: {data.median():.2f}")
        print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"  Lower bound (IQR): {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
        print(f"  Выбросы (IQR метод): {outliers_iqr} ({outliers_iqr/len(data)*100:.2f}%)")
        print(f"  Выбросы (Z-score > 3): {outliers_zscore} ({outliers_zscore/len(data)*100:.2f}%)")
        
        # Анализ экстремальных значений
        extreme_low = data[data < lower_bound]
        extreme_high = data[data > upper_bound]
        
        if len(extreme_low) > 0:
            print(f"  ⚠️  Экстремально низкие значения: {extreme_low.min():.2f} (всего {len(extreme_low)})")
        if len(extreme_high) > 0:
            print(f"  ⚠️  Экстремально высокие значения: {extreme_high.max():.2f} (всего {len(extreme_high)})")
        
        outlier_results[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outliers_iqr': outliers_iqr,
            'outliers_iqr_pct': outliers_iqr/len(data)*100,
            'outliers_zscore': outliers_zscore,
            'outliers_zscore_pct': outliers_zscore/len(data)*100
        }
    
    # Визуализация выбросов
    print("\n📊 Создание графиков выбросов...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(num_cols):
        ax = axes[idx]
        data = train[col].dropna()
        
        # Box plot
        bp = ax.boxplot(data, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax.set_title(f'{col}\nВыбросов (IQR): {outlier_results[col]["outliers_iqr"]} ({outlier_results[col]["outliers_iqr_pct"]:.1f}%)')
        ax.set_ylabel('Значение')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outliers_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ График сохранен: outliers_analysis.png")
    
    return outlier_results

def feature_importance_analysis(train, num_cols, cat_cols):
    """Анализ важности признаков"""
    print("\n" + "="*80)
    print("⭐ АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
    print("="*80)
    
    # Подготовка данных
    X = train.drop(columns=['id', 'loan_status'])
    y = train['loan_status']
    
    # Кодирование категориальных признаков
    X_encoded = X.copy()
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str).fillna('Missing'))
        le_dict[col] = le
    
    # Обучение Random Forest для оценки важности
    print("\n🌲 Обучение Random Forest для оценки важности признаков...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_encoded, y)
    
    # Важность признаков
    feature_importance = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Важность признаков (Top 15):")
    print(feature_importance.head(15).to_string(index=False))
    
    # Визуализация
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title('Top 15 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    print("✅ График сохранен: feature_importance.png")
    
    return feature_importance

def correlation_analysis(train, num_cols):
    """Анализ корреляций"""
    print("\n" + "="*80)
    print("🔗 АНАЛИЗ КОРРЕЛЯЦИЙ")
    print("="*80)
    
    # Корреляция числовых признаков с целевой переменной
    print("\n📊 Корреляция числовых признаков с целевой переменной:")
    correlations = {}
    for col in num_cols:
        corr = train[col].corr(train['loan_status'])
        correlations[col] = corr
        print(f"  {col}: {corr:.4f}")
    
    # Матрица корреляций между числовыми признаками
    corr_matrix = train[num_cols + ['loan_status']].corr()
    
    print("\n📊 Матрица корреляций между числовыми признаками:")
    print(corr_matrix.round(3))
    
    # Поиск сильно коррелированных пар
    print("\n🔍 Сильно коррелированные пары признаков (|corr| > 0.5):")
    high_corr_pairs = []
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            corr_val = corr_matrix.loc[num_cols[i], num_cols[j]]
            if abs(corr_val) > 0.5:
                high_corr_pairs.append((num_cols[i], num_cols[j], corr_val))
                print(f"  {num_cols[i]} <-> {num_cols[j]}: {corr_val:.3f}")
    
    if not high_corr_pairs:
        print("  Нет сильно коррелированных пар (|corr| > 0.5)")
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, mask=mask)
    plt.title('Correlation Matrix (Numerical Features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
    print("✅ График сохранен: correlation_matrix.png")
    
    return corr_matrix, high_corr_pairs

def analyze_categorical_target_relationship(train, cat_cols):
    """Анализ связи категориальных признаков с целевой переменной"""
    print("\n" + "="*80)
    print("📊 АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ И ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
    print("="*80)
    
    target_rates = {}
    
    for col in cat_cols:
        print(f"\n📌 {col}:")
        grouped = train.groupby(col)['loan_status'].agg(['mean', 'count']).round(4)
        grouped.columns = ['Default_Rate', 'Count']
        grouped = grouped.sort_values('Default_Rate', ascending=False)
        print(grouped)
        
        target_rates[col] = grouped['Default_Rate'].to_dict()
        
        # Визуализация
        plt.figure(figsize=(10, 6))
        grouped['Default_Rate'].plot(kind='bar', color='coral')
        plt.title(f'Default Rate by {col}', fontsize=12, fontweight='bold')
        plt.xlabel(col, fontsize=10)
        plt.ylabel('Default Rate', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'default_rate_{col}.png', dpi=150, bbox_inches='tight')
        print(f"  ✅ График сохранен: default_rate_{col}.png")
    
    return target_rates

def suggest_polynomial_features(train, num_cols, cat_cols, feature_importance, corr_matrix, target_rates):
    """Предложения по полиномиальным признакам на основе анализа"""
    print("\n" + "="*80)
    print("💡 ПРЕДЛОЖЕНИЯ ПО ПОЛИНОМИАЛЬНЫМ ПРИЗНАКАМ")
    print("="*80)
    
    suggestions = []
    
    # 1. Анализ на основе важности признаков
    print("\n1️⃣  На основе важности признаков:")
    top_features = feature_importance.head(10)['feature'].tolist()
    top_numeric = [f for f in top_features if f in num_cols]
    
    if len(top_numeric) >= 2:
        print(f"   Топ числовые признаки: {top_numeric[:5]}")
        # Предлагаем взаимодействия между топ признаками
        for i in range(min(3, len(top_numeric))):
            for j in range(i+1, min(4, len(top_numeric))):
                f1, f2 = top_numeric[i], top_numeric[j]
                suggestions.append({
                    'type': 'interaction',
                    'features': [f1, f2],
                    'operation': 'multiply',
                    'reason': f'Оба признака в топ-{len(top_numeric)} по важности',
                    'priority': 'high'
                })
                print(f"   ✅ {f1} * {f2} (оба в топ-{len(top_numeric)} по важности)")
    
    # 2. Анализ на основе корреляций
    print("\n2️⃣  На основе корреляций:")
    # Ищем признаки с умеренной корреляцией (0.3-0.7) - они могут дать хорошие взаимодействия
    moderate_corr_pairs = []
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            corr_val = corr_matrix.loc[num_cols[i], num_cols[j]]
            if 0.3 <= abs(corr_val) <= 0.7:
                moderate_corr_pairs.append((num_cols[i], num_cols[j], corr_val))
    
    if moderate_corr_pairs:
        print(f"   Найдено {len(moderate_corr_pairs)} пар с умеренной корреляцией (0.3-0.7):")
        for f1, f2, corr in moderate_corr_pairs[:5]:  # Показываем топ-5
            suggestions.append({
                'type': 'interaction',
                'features': [f1, f2],
                'operation': 'multiply',
                'reason': f'Умеренная корреляция ({corr:.3f}) - может выявить нелинейные зависимости',
                'priority': 'medium'
            })
            print(f"   ✅ {f1} * {f2} (корреляция: {corr:.3f})")
    
    # 3. Анализ на основе бизнес-логики
    print("\n3️⃣  На основе бизнес-логики:")
    
    # Отношения, которые имеют смысл для кредитного скоринга
    business_logic_pairs = [
        (['loan_amnt', 'person_income'], 'divide', 
         'Отношение суммы займа к доходу (уже есть loan_percent_income, но можно улучшить)',
         'high'),
        (['loan_amnt', 'person_emp_length'], 'multiply',
         'Взаимодействие суммы займа и стажа работы',
         'medium'),
        (['person_income', 'person_emp_length'], 'multiply',
         'Взаимодействие дохода и стажа работы (опытный работник с высоким доходом)',
         'medium'),
        (['loan_int_rate', 'loan_amnt'], 'multiply',
         'Взаимодействие процентной ставки и суммы займа',
         'medium'),
        (['person_age', 'person_emp_length'], 'divide',
         'Отношение возраста к стажу (может показать стабильность карьеры)',
         'low'),
        (['cb_person_cred_hist_length', 'person_age'], 'divide',
         'Отношение кредитной истории к возрасту (показывает раннее начало кредитной истории)',
         'medium'),
    ]
    
    for features, operation, reason, priority in business_logic_pairs:
        if all(f in num_cols for f in features):
            suggestions.append({
                'type': 'interaction',
                'features': features,
                'operation': operation,
                'reason': reason,
                'priority': priority
            })
            op_symbol = '*' if operation == 'multiply' else '/'
            print(f"   ✅ {features[0]} {op_symbol} {features[1]} ({reason})")
    
    # 4. Полиномиальные признаки для важных признаков
    print("\n4️⃣  Полиномиальные признаки (степени):")
    top_3_numeric = top_numeric[:3] if len(top_numeric) >= 3 else top_numeric
    for feat in top_3_numeric:
        suggestions.append({
            'type': 'polynomial',
            'features': [feat],
            'operation': 'square',
            'reason': f'Квадрат важного признака (топ-3 по важности)',
            'priority': 'medium'
        })
        print(f"   ✅ {feat}^2 (топ-3 по важности)")
    
    # 5. Категориальные взаимодействия
    print("\n5️⃣  Взаимодействия категориальных признаков:")
    # Анализируем target rates для категориальных признаков
    high_variance_cats = []
    for col, rates in target_rates.items():
        if len(rates) > 1:
            variance = np.var(list(rates.values()))
            if variance > 0.01:  # Если есть значительная вариация в default rate
                high_variance_cats.append(col)
    
    if len(high_variance_cats) >= 2:
        for i in range(min(2, len(high_variance_cats))):
            for j in range(i+1, min(3, len(high_variance_cats))):
                f1, f2 = high_variance_cats[i], high_variance_cats[j]
                suggestions.append({
                    'type': 'categorical_interaction',
                    'features': [f1, f2],
                    'operation': 'concat',
                    'reason': f'Оба признака показывают значительную вариацию в default rate',
                    'priority': 'medium'
                })
                print(f"   ✅ {f1} + {f2} (комбинация категорий)")
    
    # Сортировка по приоритету
    priority_order = {'high': 3, 'medium': 2, 'low': 1}
    suggestions.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
    
    # Сохранение предложений
    print("\n" + "="*80)
    print("📋 ИТОГОВЫЕ ПРЕДЛОЖЕНИЯ (отсортированы по приоритету):")
    print("="*80)
    
    high_priority = [s for s in suggestions if s['priority'] == 'high']
    medium_priority = [s for s in suggestions if s['priority'] == 'medium']
    low_priority = [s for s in suggestions if s['priority'] == 'low']
    
    print(f"\n🔴 ВЫСОКИЙ ПРИОРИТЕТ ({len(high_priority)} предложений):")
    for idx, s in enumerate(high_priority, 1):
        op_symbol = '*' if s['operation'] == 'multiply' else '/' if s['operation'] == 'divide' else '^2'
        features_str = f" {op_symbol} ".join(s['features']) if len(s['features']) > 1 else f"{s['features'][0]}^2"
        print(f"   {idx}. {features_str}")
        print(f"      Обоснование: {s['reason']}")
    
    print(f"\n🟡 СРЕДНИЙ ПРИОРИТЕТ ({len(medium_priority)} предложений):")
    for idx, s in enumerate(medium_priority, 1):
        op_symbol = '*' if s['operation'] == 'multiply' else '/' if s['operation'] == 'divide' else '^2'
        features_str = f" {op_symbol} ".join(s['features']) if len(s['features']) > 1 else f"{s['features'][0]}^2"
        print(f"   {idx}. {features_str}")
        print(f"      Обоснование: {s['reason']}")
    
    if low_priority:
        print(f"\n🟢 НИЗКИЙ ПРИОРИТЕТ ({len(low_priority)} предложений):")
        for idx, s in enumerate(low_priority, 1):
            op_symbol = '*' if s['operation'] == 'multiply' else '/' if s['operation'] == 'divide' else '^2'
            features_str = f" {op_symbol} ".join(s['features']) if len(s['features']) > 1 else f"{s['features'][0]}^2"
            print(f"   {idx}. {features_str}")
            print(f"      Обоснование: {s['reason']}")
    
    return suggestions

def create_summary_report(train, test, outlier_results, feature_importance, corr_matrix, 
                         high_corr_pairs, target_rates, suggestions):
    """Создание итогового отчета"""
    print("\n" + "="*80)
    print("📄 СОЗДАНИЕ ИТОГОВОГО ОТЧЕТА")
    print("="*80)
    
    report = []
    report.append("="*80)
    report.append("ИТОГОВЫЙ ОТЧЕТ EDA - LOAN APPROVAL PREDICTION")
    report.append("="*80)
    report.append("")
    
    # 1. Общая информация
    report.append("1. ОБЩАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ")
    report.append("-" * 80)
    report.append(f"Размер train: {train.shape}")
    report.append(f"Размер test: {test.shape}")
    report.append(f"Целевая переменная: loan_status")
    report.append(f"Процент положительного класса: {train['loan_status'].mean()*100:.2f}%")
    report.append("")
    
    # 2. Выбросы
    report.append("2. АНАЛИЗ ВЫБРОСОВ")
    report.append("-" * 80)
    for col, results in outlier_results.items():
        report.append(f"{col}:")
        report.append(f"  - Выбросы (IQR): {results['outliers_iqr']} ({results['outliers_iqr_pct']:.2f}%)")
        report.append(f"  - Выбросы (Z-score): {results['outliers_zscore']} ({results['outliers_zscore_pct']:.2f}%)")
    report.append("")
    
    # 3. Важность признаков
    report.append("3. ТОП-10 ВАЖНЫХ ПРИЗНАКОВ")
    report.append("-" * 80)
    for idx, row in feature_importance.head(10).iterrows():
        report.append(f"{idx+1}. {row['feature']}: {row['importance']:.4f}")
    report.append("")
    
    # 4. Корреляции
    report.append("4. СИЛЬНО КОРРЕЛИРОВАННЫЕ ПАРЫ (|corr| > 0.5)")
    report.append("-" * 80)
    if high_corr_pairs:
        for f1, f2, corr in high_corr_pairs:
            report.append(f"{f1} <-> {f2}: {corr:.3f}")
    else:
        report.append("Нет сильно коррелированных пар")
    report.append("")
    
    # 5. Предложения
    report.append("5. ПРЕДЛОЖЕНИЯ ПО ПОЛИНОМИАЛЬНЫМ ПРИЗНАКАМ")
    report.append("-" * 80)
    high_priority = [s for s in suggestions if s['priority'] == 'high']
    report.append(f"\nВысокий приоритет ({len(high_priority)}):")
    for s in high_priority:
        op_symbol = '*' if s['operation'] == 'multiply' else '/' if s['operation'] == 'divide' else '^2'
        features_str = f" {op_symbol} ".join(s['features']) if len(s['features']) > 1 else f"{s['features'][0]}^2"
        report.append(f"  - {features_str}: {s['reason']}")
    
    report.append("")
    report.append("="*80)
    
    # Сохранение отчета
    report_text = "\n".join(report)
    with open('eda_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("✅ Отчет сохранен: eda_report.txt")
    print("\n" + report_text)

def main():
    """Главная функция для запуска EDA"""
    print("\n" + "🚀"*40)
    print("НАЧАЛО EDA АНАЛИЗА")
    print("🚀"*40 + "\n")
    
    # 1. Загрузка данных
    train, test = load_data()
    
    # 2. Базовая статистика
    num_cols, cat_cols = basic_statistics(train, test)
    
    # 3. Анализ выбросов
    outlier_results = analyze_outliers(train, num_cols)
    
    # 4. Важность признаков
    feature_importance = feature_importance_analysis(train, num_cols, cat_cols)
    
    # 5. Корреляции
    corr_matrix, high_corr_pairs = correlation_analysis(train, num_cols)
    
    # 6. Анализ категориальных признаков
    target_rates = analyze_categorical_target_relationship(train, cat_cols)
    
    # 7. Предложения по полиномиальным признакам
    suggestions = suggest_polynomial_features(
        train, num_cols, cat_cols, feature_importance, 
        corr_matrix, target_rates
    )
    
    # 8. Итоговый отчет
    create_summary_report(
        train, test, outlier_results, feature_importance,
        corr_matrix, high_corr_pairs, target_rates, suggestions
    )
    
    print("\n" + "✅"*40)
    print("EDA АНАЛИЗ ЗАВЕРШЕН!")
    print("✅"*40)
    print("\nСозданные файлы:")
    print("  - outliers_analysis.png")
    print("  - feature_importance.png")
    print("  - correlation_matrix.png")
    print("  - default_rate_*.png (для каждого категориального признака)")
    print("  - eda_report.txt")
    print("\n")

if __name__ == "__main__":
    main()

