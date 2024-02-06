import pandas as pd
import os
from datetime import datetime
import numpy as np

### Parameters ###

#If using absolute sum filter
top_filter = 30

### Italy ###

#Adding header entry
file_path_cases = 'data/Italy/italy_labels.csv'

with open(file_path_cases, 'r') as f:
    content = f.readlines()

if content[0][0:5] != 'index':
    content[0] = 'index' + content[0]

with open(file_path_cases, 'w') as f:
    f.writelines(content)
    f.close()

#Cleaning columns
cases_ita_df = pd.read_csv(file_path_cases)
cases_ita_df = cases_ita_df.drop(columns=['index', 'id'])

#Fix Bergamo outlier at 2020-05-12 with average of week
cases_ita_df.at[12, '2020-05-12'] = int(cases_ita_df.iloc[12, 72:79].mean())

#Selecting subset of cities by percentile.
#upper_quant_ita = cases_ita_df.iloc[:, 1:].max(axis=1).quantile(0.75)
#cases_ita_df = cases_ita_df[cases_ita_df.iloc[:, 1:].ge(upper_quant_ita).any(axis=1)]

#Selecting subset of cities by number of absoluet cases.
top_ita = cases_ita_df.iloc[:, 1:].sum(axis=1).sort_values(ascending=False).head(top_filter).index
cases_ita_df = cases_ita_df.loc[top_ita]
selected_cities_ita = cases_ita_df['name'].to_numpy(dtype='str')

### Spain ###

#Importing CSV
cases_spa_df = pd.read_csv('data/Spain/spain_labels.csv')
cases_spa_df[cases_spa_df.columns[1:]] = cases_spa_df.iloc[:, 1:].astype('int64')

#Outlier due to mis-reporting
cases_spa_df = cases_spa_df.drop(index = 51)

#Selecting subset of cities by quantile
#upper_quant_spa = cases_spa_df.iloc[:, 1:].mean(axis=1).quantile(0.80)
#cases_spa_df = cases_spa_df[cases_spa_df.iloc[:, 1:].ge(upper_quant_spa).any(axis=1)]

#Selecting subset by absolute number of cases.
top_spa = cases_spa_df.iloc[:, 1:].sum(axis=1).sort_values(ascending=False).head(top_filter).index
cases_spa_df = cases_spa_df.loc[top_spa]
selected_cities_spa = cases_spa_df['name'].to_numpy(dtype='str')

### France ###

#Importing CSV
cases_fra_df = pd.read_csv('data/France/france_labels.csv')

#Selecting subset of cities by quantiles
#upper_quant_fra = cases_fra_df.iloc[:, 1:].mean(axis=1).quantile(0.8)
#cases_fra_df = cases_fra_df[cases_fra_df.iloc[:, 1:].ge(upper_quant_fra).any(axis=1)]

#Selecting cities by absolute number of cases
top_fra = cases_fra_df.iloc[:, 1:].sum(axis=1).sort_values(ascending=False).head(top_filter).index
cases_fra_df = cases_fra_df.loc[top_fra]
selected_cities_fra = cases_fra_df['name'].to_numpy(dtype='str')

### England ###
cases_eng_df = pd.read_csv('data/England/england_labels.csv')

#Selecting cities by quantiles
#upper_quant_eng = cases_eng_df.iloc[:, 1:].mean(axis=1).quantile(0.75)
#cases_eng_df = cases_eng_df[cases_eng_df.iloc[:, 1:].ge(upper_quant_eng).any(axis=1)]

#Selecting subset of cities by absolute value
top_eng = cases_eng_df.iloc[:, 1:].sum(axis=1).sort_values(ascending=True).head(top_filter).index
cases_eng_df = cases_eng_df.loc[top_eng]
selected_cities_eng = cases_eng_df['name'].to_numpy(dtype='str')

### Aggregating ###

selected_cities_list = [selected_cities_ita, selected_cities_spa, selected_cities_fra, selected_cities_eng]
cases_dfs_list = [cases_ita_df, cases_spa_df, cases_fra_df, cases_eng_df]
countries = ['ita', 'spa', 'fra', 'eng']
folder_path_dict = {
    'ita' : 'data/Italy/graphs',
    'spa' : 'data/Spain/graphs',
    'fra' : 'data/France/graphs',
    'eng' : 'data/England/graphs',
}
movement_dfs_list = []

for i in range(len(countries)):
    movement_df = pd.DataFrame(columns=['src', 'trg', 'movement', 'date'])
    movement_df = movement_df.astype({'src':'str', 'trg':'str', 'movement':'int64'})
    movement_df['date'] = pd.to_datetime(movement_df['date'])

    folder_path = folder_path_dict[countries[i]]

    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as f:
            content = f.readlines()

        if content[0].strip() != ','.join(['src', 'trg', 'movement']):
            content.insert(0, ','.join(['src', 'trg', 'movement']) + '\n')
        
        with open(file_path, 'w') as f:
            f.writelines(content)
            f.close()

        df = pd.read_csv(file_path)
        df = df.astype({'src':'str', 'trg':'str', 'movement':'int64'})
        df = df[(df['src'].isin(selected_cities_list[i])) & (df['trg'].isin(selected_cities_list[i]))]

        year, month, day = int(filename[3:-10]), int(filename[8:-7]), int(filename[11:-4])
        date = datetime(year, month, day)
        df['date'] = date   
        df['date'] = pd.to_datetime(df['date'])

        df = df.sort_values(by=['src', 'trg'])

        movement_df = pd.concat([movement_df, df], axis = 0)

    movement_df = movement_df.reset_index(drop=True)
    movement_dfs_list.append(movement_df)

movement_ita_df = movement_dfs_list[0]
movement_spa_df = movement_dfs_list[1]
movement_fra_df = movement_dfs_list[2]
movement_eng_df = movement_dfs_list[3]

### Indexing ###

selected_cities_ita = np.sort(selected_cities_ita)
selected_cities_spa = np.sort(selected_cities_spa)
selected_cities_fra = np.sort(selected_cities_fra)
selected_cities_eng = np.sort(selected_cities_eng)

mapping_ita = {city: i for i, city in enumerate(selected_cities_ita, start=1)}
mapping_spa = {city: i for i, city in enumerate(selected_cities_spa, start=1)}
mapping_fra = {city: i for i, city in enumerate(selected_cities_fra, start=1)}
mapping_eng = {city: i for i, city in enumerate(selected_cities_eng, start=1)}

movement_ita_df['src'] = movement_ita_df['src'].map(mapping_ita)
movement_ita_df['trg'] = movement_ita_df['trg'].map(mapping_ita)
movement_spa_df['src'] = movement_spa_df['src'].map(mapping_spa)
movement_spa_df['trg'] = movement_spa_df['trg'].map(mapping_spa)
movement_fra_df['src'] = movement_fra_df['src'].map(mapping_fra)
movement_fra_df['trg'] = movement_fra_df['trg'].map(mapping_fra)
movement_eng_df['src'] = movement_eng_df['src'].map(mapping_eng)
movement_eng_df['trg'] = movement_eng_df['trg'].map(mapping_eng)