from torch_geometric.data import Data, TemporalData
import torch
import numpy as np
import pickle

with open('trade_data.pkl', 'rb') as f:
    trade_df, stocks_df = pickle.load(f)

def generate_dict_graph_snapshots(trade_df, feature_df):
    dates = trade_df['date'].unique()
    snapshots_dict = {}
    num_nodes = trade_df['dst'].max() + 1

    for i in range(len(dates)):
        date = dates[i]
        df = trade_df[trade_df['date']==date]
        
        edge_index_arr = np.vstack((df['src'].values, df['dst'].values))
        edge_index = torch.tensor(edge_index_arr, dtype=torch.long)
        edge_weights = torch.tensor(df['amount'].values[:, None], dtype=torch.float)
        edge_weights_index = edge_index.detach().clone()
        
        node_features = torch.tensor(feature_df[feature_df['date']==date]['value'].values, dtype=torch.float).unsqueeze(1)
        
        graph = Data(x=node_features, edge_index=edge_index, edge_weights=edge_weights, 
                     edge_weights_index=edge_weights_index, num_nodes=num_nodes)

        snapshots_dict[dates[i]] = graph
        
    return snapshots_dict

def generate_temporal_data(trade_df, feature_df):
        df = trade_df.copy()
        dates = df['date'].unique()
        
        node_features_list = []
        for date in dates:
                x = torch.tensor(feature_df[feature_df['date']==date]['value'].values, dtype=torch.float).unsqueeze(1)
                node_features_list.append(x)
        
        node_features = {i:node_features_list[i] for i in range(len(dates))}

        src = torch.tensor(df['src'].values, dtype=torch.long)
        trg = torch.tensor(df['dst'].values, dtype=torch.long)
        edge_weights = torch.tensor(df['amount'].values, dtype=torch.float32).unsqueeze(1)
        msg = torch.ones_like(edge_weights)
        
        df['date'] = df['date'].astype('datetime64[s]').astype('int')
        dates_mapping = {date: i for i, date in enumerate(df['date'].unique())}
        df['date'] = df['date'].map(dates_mapping)
        t = torch.tensor(df['date'].values, dtype=torch.int64)

        td = TemporalData(src=src, dst=trg, t=t, msg=msg), node_features, edge_weights
        return td

dict_snapshots_trade = generate_dict_graph_snapshots(trade_df, stocks_df)
temporal_data_trade = generate_temporal_data(trade_df, stocks_df)