import os
import pandas as pd
import numpy as np
from pathlib import Path

def load_and_merge_data(data_root: str = '.'):
    print("Загрузка данных...")
    
    # Получаем абсолютный путь к корневой папке
    data_root_abs = Path(data_root).absolute()
    print(f"  Корневая папка: {data_root_abs}\n")
    
    # 1. Читаем Лист1 из dataset_with_artefacts.xlsx
    print("  Читаем Лист1 из dataset_with_artefacts.xlsx...")
    df_artefacts = pd.read_excel('dataset_with_artefacts.xlsx', sheet_name='Лист1', dtype=str)
    
    print(f"  Колонки в Лист1: {df_artefacts.columns.tolist()}")
    
    # Проверяем наличие нужных колонок
    required_cols = ['study_id', 'rel_path']
    for col in required_cols:
        if col not in df_artefacts.columns:
            raise ValueError(f"В Лист1 не найдена колонка '{col}'. Доступные: {df_artefacts.columns.tolist()}")
    
    # 2. Фильтруем только исследования позвоночника
    spine_col = None
    for col in df_artefacts.columns:
        if 'spine' in col.lower() or 'позвоночн' in col.lower():
            spine_col = col
            break
    
    if spine_col:
        print(f"  Найдена колонка для фильтрации позвоночника: '{spine_col}'")
        df_spine = df_artefacts[df_artefacts[spine_col] == '1'].copy()
        print(f"  Отфильтровано {len(df_spine)} исследований позвоночника из {len(df_artefacts)}")
    else:
        print("   Колонка для фильтрации позвоночника не найдена, используем все исследования")
        df_spine = df_artefacts.copy()
    
    # 3. Загружаем разметку
    print("  Читаем разметка.xlsx...")
    df_labels = pd.read_excel('разметка.xlsx', header=[0, 1], dtype=str)
    
    study_cols = [col for col in df_labels.columns if str(col[0]).lower() == 'study']
    if not study_cols:
        raise ValueError("Не удалось найти колонку 'study' в файле разметка.xlsx")
    study_col = study_cols[0]
    
    target_col = ('Позвоночник', 'корректная укладка')
    if target_col not in df_labels.columns:
        raise ValueError(f"Не найдена колонка {target_col} в файле разметка.xlsx")
    
    df_clean = pd.DataFrame({
        'study': df_labels[study_col].str.strip(),
        'is_incorrect': df_labels[target_col]
    })
    
    df_clean['is_incorrect'] = df_clean['is_incorrect'].fillna('0').astype(float).astype(int)
    df_clean = df_clean[
        df_clean['study'].notna() & 
        (df_clean['study'] != '') & 
        (df_clean['study'] != 'nan') & 
        (df_clean['study'] != 'None')
    ]
    
    # 4. Объединение
    df_spine['study_id'] = df_spine['study_id'].str.strip()
    
    merged_df = pd.merge(
        df_spine,
        df_clean,
        left_on='study_id',
        right_on='study',
        how='inner'
    ).drop(columns=['study'])
    
    # 5. Создаём АБСОЛЮТНЫЕ полные пути
    print("\n  Создаём абсолютные пути к файлам...")
    result_rows = []
    not_found = 0
    
    for idx, row in merged_df.iterrows():
        rel_path = row['rel_path']
        # Создаём абсолютный путь
        full_path = (data_root_abs / rel_path).absolute()
        
        if full_path.exists():
            new_row = row.copy()
            new_row['full_path'] = str(full_path)  # Сохраняем как строку
            result_rows.append(new_row)
        else:
            not_found += 1
            if not_found <= 3:
                print(f"  ⚠ Файл не найден: {full_path}")
    
    result_df = pd.DataFrame(result_rows)
    
    print(f"\n✅ Успешно объединено {len(result_df)} DICOM файлов (только позвоночник).")
    print(f"   Не найдено файлов: {not_found}")
    
    if len(result_df) > 0:
        counts = result_df['is_incorrect'].value_counts().to_dict()
        print(f"   Распределение классов (0 - норма, 1 - некорректная укладка): {counts}")
    else:
        print("    ВНИМАНИЕ: Ни один файл не найден! Проверьте путь data_root.")
    
    return result_df

if __name__ == '__main__':
    try:
        # Увеличиваем ширину вывода pandas, чтобы пути не обрезались
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', 200)
        
        df = load_and_merge_data()
        print("\nПервые 5 строк результата (полные пути):")
        print(df[['study_id', 'full_path', 'is_incorrect']].head().to_string())
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()