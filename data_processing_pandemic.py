
import pandas as pd
import os
from datetime import datetime
import numpy as np
import itertools

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

 
## Italy

#Adding header
file_path_cases = 'data_pandemic/Italy/italy_labels.csv'

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
cases_ita_df = cases_ita_df.iloc[:, :80]
cases_ita_df[cases_ita_df.columns[1:]] = cases_ita_df.iloc[:, 1:].astype('float64')


#Fix Bergamo outlier at 2020-05-12 with average of week
cases_ita_df.at[12, '2020-05-12'] = int(cases_ita_df.iloc[12, 72:79].mean())

#Selecting subset of cities by number of absoluet cases.
top_30_ita = cases_ita_df.iloc[:, 1:].sum(axis=1).sort_values(ascending=False).head(30).index
cases_ita_df = cases_ita_df.loc[top_30_ita]
selected_cities_ita = cases_ita_df['name'].to_numpy(dtype='str')

 
## Spain
cases_spa_df = pd.read_csv('data_pandemic/Spain/spain_labels.csv')
cases_spa_df[cases_spa_df.columns[1:]] = cases_spa_df.iloc[:, 1:].astype('float64')

#Outlier due to mis-reporting
cases_spa_df = cases_spa_df.drop(index = 51)

#Selecting subset by absolute number of cases.
top_30_spa = cases_spa_df.iloc[:, 1:].sum(axis=1).sort_values(ascending=False).head(30).index
cases_spa_df = cases_spa_df.loc[top_30_spa]
selected_cities_spa = cases_spa_df['name'].to_numpy(dtype='str')


## France
cases_fra_df = pd.read_csv('data_pandemic/France/france_labels.csv')
cases_fra_df[cases_fra_df.columns[1:]] = cases_fra_df.iloc[:, 1:].astype('float64')

#Selecting cities by absolute number of cases
top_30_fra = cases_fra_df.iloc[:, 1:].sum(axis=1).sort_values(ascending=False).head(30).index
cases_fra_df = cases_fra_df.loc[top_30_fra]
selected_cities_fra = cases_fra_df['name'].to_numpy(dtype='str')

 
## England
cases_eng_df = pd.read_csv('data_pandemic/England/england_labels.csv')
cases_eng_df[cases_eng_df.columns[1:]] = cases_eng_df.iloc[:, 1:].astype('float64')

#Selecting subset of cities by absolute value
top_30_eng = cases_eng_df.iloc[:, 1:].sum(axis=1).sort_values(ascending=True).head(30).index
cases_eng_df = cases_eng_df.loc[top_30_eng]
selected_cities_eng = cases_eng_df['name'].to_numpy(dtype='str')

 
## Aggregating
selected_cities_list = [selected_cities_ita, selected_cities_spa, selected_cities_fra, selected_cities_eng]
cases_dfs_list = [cases_ita_df, cases_spa_df, cases_fra_df, cases_eng_df]
countries = ['ita', 'spa', 'fra', 'eng']
folder_path_dict = {
    'ita' : 'data_pandemic/Italy/graphs',
    'spa' : 'data_pandemic/Spain/graphs',
    'fra' : 'data_pandemic/France/graphs',
    'eng' : 'data_pandemic/England/graphs',
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
        df = df.astype({'src':'str', 'trg':'str', 'movement':'float64'})
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

## Indexing
selected_cities_ita = np.sort(selected_cities_ita)
selected_cities_spa = np.sort(selected_cities_spa)
selected_cities_fra = np.sort(selected_cities_fra)
selected_cities_eng = np.sort(selected_cities_eng)

mapping_ita = {city: i for i, city in enumerate(selected_cities_ita, start=0)}
mapping_spa = {city: i for i, city in enumerate(selected_cities_spa, start=0)}
mapping_fra = {city: i for i, city in enumerate(selected_cities_fra, start=0)}
mapping_eng = {city: i for i, city in enumerate(selected_cities_eng, start=0)}

def indexing_data(movement_df, cases_df, mapping):
    movement_df['src'] = movement_df['src'].map(mapping)
    movement_df['trg'] = movement_df['trg'].map(mapping)
    cases_df['name'] = cases_df['name'].map(mapping)
    cases_df = cases_df.sort_values(by=['name']).reset_index(drop=True)
    return movement_df, cases_df

movement_ita_df, cases_ita_df = indexing_data(movement_ita_df, cases_ita_df, mapping_ita)
movement_spa_df, cases_spa_df = indexing_data(movement_spa_df, cases_spa_df, mapping_spa)
movement_fra_df, cases_fra_df = indexing_data(movement_fra_df, cases_fra_df, mapping_fra)
movement_eng_df, cases_eng_df = indexing_data(movement_eng_df, cases_eng_df, mapping_eng)


## Engineering
# Function for generating fully connected dataframes for fully connected graphs. 
def generate_df_connected_movement(movement_df):
    dates = movement_df['date'].unique()
    cities = list(range(0, 30))
    all_combinations = pd.DataFrame(list(itertools.product(dates, cities, cities)), columns=['date', 'src', 'trg'])
    connected_movement_df = all_combinations.merge(movement_df, on=['date', 'src', 'trg'], how='left')
    connected_movement_df['movement'] = connected_movement_df['movement'].fillna(0)
    connected_movement_df['movement'] = connected_movement_df['movement'].astype(int)
    return connected_movement_df

movement_ita_df = generate_df_connected_movement(movement_ita_df)
movement_spa_df = generate_df_connected_movement(movement_spa_df)
movement_fra_df = generate_df_connected_movement(movement_fra_df)
movement_eng_df = generate_df_connected_movement(movement_eng_df)

# Function for adding edge feature with frequency.
def calc_positive_edge_freq(movement_df):
    movement_df['is_positive'] = movement_df['movement'] > 0
    movement_df['cumsum_positive'] = movement_df.groupby(['src', 'trg'])['is_positive'].cumsum()
    movement_df['positive_freq'] = movement_df.groupby(['src', 'trg']).cumcount()
    movement_df['positive_freq'] = movement_df['cumsum_positive'] / (movement_df['positive_freq'] + 1)
    movement_df.drop(columns=['is_positive', 'cumsum_positive'], inplace=True)

    return movement_df

movement_ita_df = calc_positive_edge_freq(movement_ita_df)
movement_spa_df = calc_positive_edge_freq(movement_spa_df)
movement_fra_df = calc_positive_edge_freq(movement_fra_df)
movement_eng_df = calc_positive_edge_freq(movement_eng_df) 

# Normalising data. 
cases_ita_df.iloc[:, 1:] = np.log(cases_ita_df.iloc[:, 1:]+1)
cases_spa_df.iloc[:, 1:] = np.log(cases_spa_df.iloc[:, 1:]+1)
cases_fra_df.iloc[:, 1:] = np.log(cases_fra_df.iloc[:, 1:]+1)
cases_eng_df.iloc[:, 1:] = np.log(cases_eng_df.iloc[:, 1:]+1)

movement_ita_df['movement'] = np.log(movement_ita_df['movement'].values + 1)
movement_spa_df['movement'] = np.log(movement_spa_df['movement'].values + 1)
movement_fra_df['movement'] = np.log(movement_fra_df['movement'].values + 1)
movement_eng_df['movement'] = np.log(movement_eng_df['movement'].values + 1)

## Saving for import
movement_dfs_dict = {
    'movement_ita_df' : movement_ita_df,
    'movement_spa_df' : movement_spa_df,
    'movement_fra_df' : movement_fra_df,
    'movement_eng_df' : movement_eng_df
}
cases_dfs_dict = {
    'cases_ita_df' : cases_ita_df,
    'cases_spa_df' : cases_spa_df,
    'cases_fra_df' : cases_fra_df,
    'cases_eng_df' : cases_eng_df
}