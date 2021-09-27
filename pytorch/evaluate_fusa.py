import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd

def get_target_names(json_file):
    a = pd.read_json(json_file).to_dict()
    target_names = list(a.keys())
    return target_names

def get_label_transforms(dataset_name):
    taxonomy_path = "fusa_taxonomy.json"
    a = pd.read_json(taxonomy_path).T[dataset_name].to_dict()
    transforms = {}
    for key, values in a.items():
        for value in values:
            transforms[value] = key
    return transforms

audioset_transforms = get_label_transforms("AudioSet")
esc_transforms = get_label_transforms("ESC")
us_transforms = get_label_transforms("UrbanSound")

esc_results_csv = 'fusa_ESC_results_fusa_labels.csv'
us_results_csv = 'fusa_UrbanSound_results_fusa_labels.csv'
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
fusa_output_1_esc = df_esc['fusa_output_1'].fillna('Not-in-FUSA')
fusa_output_2_esc = df_esc['fusa_output_2'].fillna('Not-in-FUSA')
fusa_output_3_esc = df_esc['fusa_output_3'].fillna('Not-in-FUSA')

labels_us = df_us['label']
fusa_output_1_us = df_us['fusa_output_1'].fillna('Not-in-FUSA')
fusa_output_2_us = df_us['fusa_output_2'].fillna('Not-in-FUSA')
fusa_output_3_us = df_us['fusa_output_3'].fillna('Not-in-FUSA')

esc_fusa_labels = []
us_fusa_labels = []

metrics_folder = 'metrics/'

print('---------WITH BEST SCORE OUTPUT----------')
for i in range(len(labels_esc)):
    esc_fusa_labels.append(esc_transforms[labels_esc[i]])

for j in range(len(labels_us)):
    us_fusa_labels.append(us_transforms[labels_us[j]])


print('------------------ESC50------------------')
report = classification_report(esc_fusa_labels, fusa_output_1_esc, zero_division=0, output_dict=True)
on_labels = list((report.keys()))[:-3]
matrix = confusion_matrix(esc_fusa_labels, fusa_output_1_esc, normalize='all')
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,
                               display_labels=on_labels)
fig, ax = plt.subplots(figsize=(14,14))
disp.plot(xticks_rotation='vertical', ax=ax, values_format='.2f')
plt.savefig(metrics_folder + "esc_best_acc_norm.png")

df_report = pd.DataFrame(report).transpose()
df_report = df_report[df_report.support != 0]
#df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
df_report['precision'] = np.round(df_report['precision'], decimals = 3)
df_report['recall'] = np.round(df_report['recall'], decimals = 3)
df_report['f1-score'] = np.round(df_report['f1-score'], decimals = 3)
print(df_report)
df_report.to_csv(metrics_folder + 'esc_metrics_best_acc.csv', header=True)

print('------------------URBANSOUND------------------')
report = classification_report(us_fusa_labels, fusa_output_1_us, zero_division=0, output_dict=True)
on_labels = list((report.keys()))[:-3]
matrix = confusion_matrix(us_fusa_labels, fusa_output_1_us, normalize='all')
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,
                               display_labels=on_labels)
fig, ax = plt.subplots(figsize=(14,14))
disp.plot(xticks_rotation='vertical', ax=ax, values_format='.2f')
plt.savefig(metrics_folder + "us_best_acc_norm.png")

df_report = pd.DataFrame(report).transpose()
df_report = df_report[df_report.support != 0]
#df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
df_report['precision'] = np.round(df_report['precision'], decimals = 3)
df_report['recall'] = np.round(df_report['recall'], decimals = 3)
df_report['f1-score'] = np.round(df_report['f1-score'], decimals = 3)
print(df_report)
df_report.to_csv(metrics_folder + 'us_metrics_best_acc.csv', header=True)

print('--------WITH 3 BEST SCORE OUTPUTS--------')
print('------------------ESC50------------------')
fusa_best_output_esc = []
for i in range(len(labels_esc)):
    outputs_array = [fusa_output_1_esc[i], fusa_output_2_esc[i], fusa_output_3_esc[i]]
    if (esc_fusa_labels[i] in outputs_array):
        index = outputs_array.index(esc_fusa_labels[i])
        fusa_best_output_esc.append(outputs_array[index])
    elif (esc_fusa_labels[i] not in outputs_array):
        if ('Not-in-FUSA' in outputs_array):
            if (outputs_array[2] != 'Not-in-FUSA'):
                fusa_best_output_esc.append(outputs_array[2])
            elif (outputs_array[1] != 'Not-in-FUSA'):
                fusa_best_output_esc.append(outputs_array[1])
            else:
                fusa_best_output_esc.append(outputs_array[0])
        else:
            fusa_best_output_esc.append(outputs_array[0])

report = classification_report(esc_fusa_labels, fusa_best_output_esc, zero_division=0, output_dict=True)
on_labels = list((report.keys()))[:-3]
matrix = confusion_matrix(esc_fusa_labels, fusa_best_output_esc, normalize='all')
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,
                               display_labels=on_labels)
fig, ax = plt.subplots(figsize=(14,14))
disp.plot(xticks_rotation='vertical', ax=ax, values_format='.2f')
plt.savefig(metrics_folder + "esc_best_3_acc_norm.png")

df_report = pd.DataFrame(report).transpose()
df_report = df_report[df_report.support != 0]
#df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
df_report['precision'] = np.round(df_report['precision'], decimals = 3)
df_report['recall'] = np.round(df_report['recall'], decimals = 3)
df_report['f1-score'] = np.round(df_report['f1-score'], decimals = 3)
print(df_report)
df_report.to_csv(metrics_folder + 'esc_metrics_3_best_accs.csv', header=True)

print('------------------URBANSOUND------------------')
fusa_best_output_us = []
for i in range(len(labels_us)):
    outputs_array = [fusa_output_1_us[i], fusa_output_2_us[i], fusa_output_3_us[i]]
    if (us_fusa_labels[i] in outputs_array):
        index = outputs_array.index(us_fusa_labels[i])
        fusa_best_output_us.append(outputs_array[index])
    elif (us_fusa_labels[i] not in outputs_array):
        if ('Not-in-FUSA' in outputs_array):
            if (outputs_array[2] != 'Not-in-FUSA'):
                fusa_best_output_us.append(outputs_array[2])
            elif (outputs_array[1] != 'Not-in-FUSA'):
                fusa_best_output_us.append(outputs_array[1])
            else:
                fusa_best_output_us.append(outputs_array[0])
        else:
            fusa_best_output_us.append(outputs_array[0])

report = classification_report(us_fusa_labels, fusa_best_output_us, zero_division=0, output_dict=True)
on_labels = list((report.keys()))[:-3]
matrix = confusion_matrix(us_fusa_labels, fusa_best_output_us, normalize='all')
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,
                               display_labels=on_labels)
fig, ax = plt.subplots(figsize=(14,14))
disp.plot(xticks_rotation='vertical', ax=ax, values_format='.2f')
plt.savefig(metrics_folder + "us_best_3_acc_norm.png")

df_report = pd.DataFrame(report).transpose()
df_report = df_report[df_report.support != 0]
#df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
df_report['precision'] = np.round(df_report['precision'], decimals = 3)
df_report['recall'] = np.round(df_report['recall'], decimals = 3)
df_report['f1-score'] = np.round(df_report['f1-score'], decimals = 3)
print(df_report)
df_report.to_csv(metrics_folder + 'us_metrics_3_best_accs.csv', header=True)