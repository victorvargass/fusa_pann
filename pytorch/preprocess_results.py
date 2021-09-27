import pandas as pd
from sklearn.metrics import classification_report
import numpy as np

def get_target_names(json_file):
    a = pd.read_json(json_file).to_dict()
    target_names = list(a.keys())
    return target_names

def get_label_transforms(dataset_name, original_label):
    taxonomy_path = "fusa_taxonomy.json"
    s1 = pd.Series(name='label', dtype=object)
    a = pd.read_json(taxonomy_path).T[dataset_name].to_dict()
    transforms = {}
    for key, values in a.items():
        for value in values:
            transforms[value] = key
    for i, e in enumerate(original_label):
        try:
            transformed_label = transforms[e]
        except:
            transformed_label = ''
        s2 = pd.Series([transformed_label], name='label', dtype=object)
        s1 = s1.append(s2, ignore_index=True)
    return s1

esc_results_csv = 'fusa_ESC_results.csv'
us_results_csv = 'fusa_UrbanSound_results.csv'
columns = [
    'audio_name',
    'dataset',
    'label',
    'output_1',
    'output_2',
    'output_3',
    'fusa_output_1',
    'fusa_output_2',
    'fusa_output_3',
    'acc_1',
    'acc_2',
    'acc_3']

target_names = get_target_names("fusa_taxonomy.json")

df_esc = pd.read_csv(esc_results_csv, usecols=columns)
df_us = pd.read_csv(us_results_csv, usecols=columns)

labels_esc = df_esc['label']
labels_us = df_us['label']

df_esc = df_esc[get_label_transforms('ESC', df_esc.label) != '']
df_us = df_us[get_label_transforms('UrbanSound', df_us.label) != '']

esc_results_csv_res = 'fusa_ESC_results_fusa_labels.csv'
us_results_csv_res = 'fusa_UrbanSound_results_fusa_labels.csv'
df_esc.to_csv(esc_results_csv_res, index = False, header=True)
df_us.to_csv(us_results_csv_res, index = False, header=True)
