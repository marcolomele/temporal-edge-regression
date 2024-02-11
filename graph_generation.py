from data_processing import movement_ita_df, movement_spa_df, movement_fra_df, movement_eng_df
from data_processing import cases_ita_df, cases_spa_df, cases_fra_df, cases_eng_df
import pandas as pd
import itertools
from datetime import datetime
from torch_geometric.data import Data
import torch
import numpy as np


def generate_complete_movement_df(movement_df):
    dates = movement_df['date'].unique()
    cities = list(range(1, 31))
    all_combinations = pd.DataFrame(list(itertools.product(dates, cities, cities)), columns=['date', 'src', 'trg'])
    complete_movement_df = all_combinations.merge(
        movement_df, on=['date', 'src', 'trg'], how='left'
    )
    complete_movement_df['movement'] = complete_movement_df['movement'].fillna(0)
    complete_movement_df['movement'] = complete_movement_df['movement'].astype(int)
    return complete_movement_df


def generate_snapshots_dfs_dict(movement_df, cases_df):
    dates = movement_df['date'].unique()
    snapshots_dict = {}

    for i in range(len(dates)):
        date = dates[i]
        df = movement_df[movement_df['date']==date]

        edge_index_arr = np.vstack((df['src'].values, df['trg'].values))
        edge_index = torch.tensor(edge_index_arr, dtype=torch.long)
        edge_attr = torch.tensor(df['movement'].values[:, None], dtype=torch.float)
        num_nodes = df['src'].max()
        node_features = torch.tensor(cases_df[str(date)[:10]].values, dtype=torch.float)
        
        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)

        snapshots_dict[date] = graph
        
    return snapshots_dict


#Creating fully connecyed data frames for fully connected graphs.
complete_movement_ita_df = generate_complete_movement_df(movement_ita_df)
complete_movement_spa_df = generate_complete_movement_df(movement_spa_df)
complete_movement_fra_df = generate_complete_movement_df(movement_fra_df)
complete_movement_eng_df = generate_complete_movement_df(movement_eng_df)

#Dictionaries with daily graphs for each country.
daily_graphs_ita_dict = generate_snapshots_dfs_dict(complete_movement_ita_df, cases_ita_df)
daily_graphs_spa_dict = generate_snapshots_dfs_dict(complete_movement_spa_df, cases_spa_df)
daily_graphs_fra_dict = generate_snapshots_dfs_dict(complete_movement_fra_df, cases_fra_df)
daily_graphs_eng_dict = generate_snapshots_dfs_dict(complete_movement_eng_df, cases_eng_df)