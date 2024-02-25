from data_processing import movement_dfs_dict, cases_dfs_dict
from torch_geometric.data import Data, TemporalData
import torch
import numpy as np
from datetime import timedelta

## Extract data
movement_ita_df = movement_dfs_dict['movement_ita_df']
movement_spa_df = movement_dfs_dict['movement_spa_df']
movement_fra_df = movement_dfs_dict['movement_fra_df']
movement_eng_df = movement_dfs_dict['movement_eng_df']

cases_ita_df = cases_dfs_dict['cases_ita_df']
cases_spa_df = cases_dfs_dict['cases_spa_df']
cases_fra_df = cases_dfs_dict['cases_fra_df']
cases_eng_df = cases_dfs_dict['cases_eng_df']

def generate_dict_graph_snapshots(movement_df, cases_df):
    dates = movement_df['date'].unique()
    snapshots_dict = {}

    for i in range(7,len(dates)):
        df = movement_df[movement_df['date']==dates[i-7]]
        
        edge_index_arr = np.vstack((df['src'].values, df['trg'].values))
        edge_index = torch.tensor(edge_index_arr, dtype=torch.long)
        edge_weights = torch.tensor(df['movement'].values[:, None], dtype=torch.float)
        edge_weights_index = edge_index.detach().clone()
        edge_attr = torch.tensor(df['positive_freq'].values[:, None], dtype=torch.float)
        num_nodes = df['src'].max() + 1
        
        node_features = torch.tensor(cases_df.iloc[:, i-6:i+1].values, dtype=torch.float)
        
        graph = Data(x=node_features, edge_index=edge_index, edge_weights=edge_weights, 
                     edge_weights_index=edge_weights_index, edge_attr=edge_attr, num_nodes=num_nodes)

        snapshots_dict[dates[i]] = graph
        
    return snapshots_dict

#Dictionaries with daily graphs for each country.
#daily_graphs_ita_dict = generate_dict_graph_snapshots(movement_ita_df, cases_ita_df)
#daily_graphs_spa_dict = generate_dict_graph_snapshots(movement_spa_df, cases_spa_df)
#daily_graphs_fra_dict = generate_dict_graph_snapshots(movement_fra_df, cases_fra_df)
#daily_graphs_eng_dict = generate_dict_graph_snapshots(movement_eng_df, cases_eng_df)

#Generaiton of temporal data (not fully conencted)
def generate_td_movement_df(movement_df):
    movement_df = movement_df[movement_df['movement']>0].copy()
    movement_df = movement_df.reset_index(drop=True)
    movement_df = movement_df.drop(columns = 'positive_freq')
    movement_df = movement_df.rename(columns = {'movement':'movement_lag7'})
    movement_df['date'] = movement_df['date'] + timedelta(days=7)    
    return movement_df

def generate_temporal_data(movement_df, cases_df):
    movement_df = generate_td_movement_df(movement_df)
    dates = movement_df['date'].unique()
    
    node_features_list = []
    for i in range(7, len(dates)):
        x = torch.tensor(cases_df.iloc[:, i-6:i+1].values, dtype=torch.float)
        node_features_list.append(x)
    node_features = {i:features for i, features, in enumerate(node_features_list)}

    dates = dates[:cases_df.iloc[:, 8:].shape[1]]
    movement_df = movement_df[movement_df['date'].isin(dates)]
    
    src = torch.tensor(movement_df['src'].values, dtype=torch.long)
    trg = torch.tensor(movement_df['trg'].values, dtype=torch.long)
    edge_weights = torch.tensor(movement_df['movement_lag7'].values, dtype=torch.float32).unsqueeze(1)
    msg = torch.ones_like(edge_weights)
    
    movement_df['date'] = movement_df['date'].astype('datetime64[s]').astype('int')
    dates_mapping = {date: i for i, date in enumerate(movement_df['date'].unique())}
    movement_df['date'] = movement_df['date'].map(dates_mapping)
    t = torch.tensor(movement_df['date'].values, dtype=torch.int64)

    return TemporalData(src=src, dst=trg, t=t, msg=msg), node_features, edge_weights

#TemporalData objects
data, node_features, edge_weights = generate_temporal_data(movement_ita_df, cases_ita_df)
temporal_data_ita = {'TemporalData':data, 
                     'node_features':node_features,
                     'edge_weights': edge_weights}

data, node_features, edge_weights = generate_temporal_data(movement_spa_df, cases_ita_df)
temporal_data_spa = {'TemporalData':data, 
                     'node_features':node_features,
                     'edge_weights': edge_weights}

data, node_features, edge_weights = generate_temporal_data(movement_fra_df, cases_ita_df)
temporal_data_fra = {'TemporalData':data, 
                     'node_features':node_features,
                     'edge_weights': edge_weights}

data, node_features, edge_weights = generate_temporal_data(movement_eng_df, cases_ita_df)
temporal_data_eng = {'TemporalData':data, 
                     'node_features':node_features,
                     'edge_weights': edge_weights}