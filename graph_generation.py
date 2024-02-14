from data_processing import movement_ita_df, movement_spa_df, movement_fra_df, movement_eng_df
from data_processing import cases_ita_df, cases_spa_df, cases_fra_df, cases_eng_df
import pandas as pd
from torch_geometric.data import Data
import torch
import numpy as np

def generate_dict_graph_snapshots(movement_df, cases_df):
    dates = movement_df['date'].unique()
    snapshots_dict = {}

    for i in range(7,len(dates)):
        df = movement_df[movement_df['date']==dates[i-7]]
        
        edge_index_arr = np.vstack((df['src'].values, df['trg'].values))
        edge_index = torch.tensor(edge_index_arr, dtype=torch.long)
        edge_weight = torch.tensor(df['movement'].values[:, None], dtype=torch.float)
        edge_weight_index = edge_index.detach().clone()
        edge_attr = torch.tensor(df['positive_freq'].values[:, None], dtype=torch.float)
        num_nodes = df['src'].max()
        
        node_features = torch.tensor(cases_df.iloc[:, i-6:i+1].values, dtype=torch.float)
        
        graph = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight, 
                     edge_weight_index=edge_weight_index, edge_attr=edge_attr, num_nodes=num_nodes)

        snapshots_dict[dates[i]] = graph
        
    return snapshots_dict

#Dictionaries with daily graphs for each country.
daily_graphs_ita_dict = generate_dict_graph_snapshots(movement_ita_df, cases_ita_df)
daily_graphs_spa_dict = generate_dict_graph_snapshots(movement_spa_df, cases_spa_df)
daily_graphs_fra_dict = generate_dict_graph_snapshots(movement_fra_df, cases_fra_df)
daily_graphs_eng_dict = generate_dict_graph_snapshots(movement_eng_df, cases_eng_df)