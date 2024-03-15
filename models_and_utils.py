from torch_geometric.loader import TemporalDataLoader
from torch.nn import functional as F
from statistics import mean
from tqdm import tqdm
import pandas as pd
import torch

## Utils
def data_split(graph_snapshots, perc_split):
    split_at = int(len(graph_snapshots) * (1-perc_split))
    return graph_snapshots[:split_at], graph_snapshots[split_at:]

def apply_negative_sampling(snapshot, all_possible_edges, ratio=1):
    #print('Note: use unconnected graph snapshots for this.')

    edge_index = snapshot.edge_index.t()
    expanded_all_edges = all_possible_edges.unsqueeze(1) 
    edge_comparisons = (expanded_all_edges == edge_index).all(dim=2)
    mask = ~edge_comparisons.any(dim=1)
    negative_edges = all_possible_edges[mask]

    negative_edges = all_possible_edges[mask]
    num_to_sample = round(edge_index.shape[0] * ratio)
    sampled_indices = torch.randperm(negative_edges.shape[0])[:num_to_sample]
    sampled_negative_edges = negative_edges[sampled_indices]

    edge_index = torch.cat((edge_index, sampled_negative_edges), dim=0).t()
    edge_weights = torch.cat((snapshot.edge_weights, torch.zeros((num_to_sample,1))), dim=0)

    new_snapshot = snapshot.clone()
    new_snapshot.edge_index = edge_index
    new_snapshot.edge_weights_index = edge_index  
    new_snapshot.edge_weights = edge_weights  

    return new_snapshot

def train(model, graph_snapshots_train, optimiser, epochs, neg_sampling=False):
    model.train()
    loss_epochs_df = pd.DataFrame(columns=['epoch', 'loss'])
    for epoch in tqdm(range(epochs)):
        losses_list = []
        for snapshot in graph_snapshots_train:
            if neg_sampling:
                nodes = torch.arange(snapshot.num_nodes)
                all_possible_edges = torch.cartesian_prod(nodes, nodes)
                snapshot = apply_negative_sampling(snapshot, all_possible_edges)
            edge_weights_pred = model(snapshot)
            loss = torch.mean((edge_weights_pred - snapshot.edge_weights)**2)
            loss.backward()
            losses_list.append(loss.item())
            optimiser.step()
            optimiser.zero_grad()
        
        avg_epoch_loss = mean(losses_list)
        loss_epochs_df.loc[len(loss_epochs_df)]={'epoch':epoch+1, 'loss':avg_epoch_loss}

    return loss_epochs_df

def test(model, graph_snapshots_test, neg_sampling=False):
    model.eval()
    loss = 0
    for snapshot in graph_snapshots_test:
        if neg_sampling:
                nodes = torch.arange(snapshot.num_nodes)
                all_possible_edges = torch.cartesian_prod(nodes, nodes)
                snapshot = apply_negative_sampling(snapshot, all_possible_edges)
        edge_weights_pred = model(snapshot)
        mse = torch.mean((edge_weights_pred - snapshot.edge_weights)**2)
        loss += mse
    loss = loss / (len(graph_snapshots_test))
    print(f'MSE = {loss.item():.4f}')

## Models
class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_size):
        super(EdgeDecoder, self).__init__()

        self.lin1 = torch.nn.Linear(hidden_size * 2, hidden_size, bias=True)
        
        self.lin2 = torch.nn.Linear(hidden_size, 1)

    def forward(self, node_embeddings, edge_weights_index):
        src = edge_weights_index[0, :]
        trg = edge_weights_index[1, :]

        edge_embeddings = torch.cat([node_embeddings[src], node_embeddings[trg]], dim=1)

        h = self.lin1(edge_embeddings)
        h = F.relu(h)
        prediction = self.lin2(h)
        return prediction.view(-1)

# MPNN LSTM
from pytorch_geometric_temporal_models import MPNNLSTM

class EncoderMPNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_size,
                            num_nodes, window, dropout_p):

        super(EncoderMPNN, self).__init__()

        self.MPNN = MPNNLSTM(in_channels=in_channels,
                             hidden_size=hidden_size,
                             num_nodes=num_nodes,
                             window=window,
                             dropout=dropout_p)

    def forward(self, x, edge_index, edge_attr):
        node_embeddings = self.MPNN(x, edge_index, edge_attr)
        return node_embeddings # size '2*nhid+in_channels+window-1' for each node.

class ModelMPNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, num_nodes,
                 window, dropout_p):
        
        super(ModelMPNN, self).__init__()

        self.encoder = EncoderMPNN(in_channels=in_channels,
                             hidden_size=hidden_size,
                             num_nodes=num_nodes,
                             window=window,
                             dropout_p=dropout_p)
        
        out_hidden_size = hidden_size * 2 + in_channels + window - 1
        self.decoder = EdgeDecoder(hidden_size=out_hidden_size)

    def forward(self, snapshot):
        x = snapshot.x
        edge_index = snapshot.edge_index
        edge_attr = snapshot.edge_attr
        edge_weights_index = snapshot.edge_weights_index

        node_embeddings = self.encoder(x, edge_index, edge_attr)
        edge_weights_pred = self.decoder(node_embeddings, edge_weights_index)
        
        return edge_weights_pred

## EvolveGCNH
from pytorch_geometric_temporal_models import EvolveGCNH
from torch.nn import Dropout

class EncoderEVOLVEGCNH(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, dropout_p,
                    cached=True, improved=True, add_self_loops=False):

        super(EncoderEVOLVEGCNH, self).__init__()

        self.EVOLVEGCNH = EvolveGCNH(num_of_nodes=num_nodes,
                                  in_channels=in_channels,
                                  cached=cached,
                                  improved=improved,
                                  add_self_loops=add_self_loops)

        self.dropout = Dropout(p=dropout_p)

    def forward(self, x, edge_index, edge_attr):
        x = self.dropout(x)
        node_embeddings=self.EVOLVEGCNH(x, edge_index, edge_attr)
        return node_embeddings

class ModelEVOLVE(torch.nn.Module):
    def __init__(self, num_nodes, hidden_size, in_channels, dropout_p):
        super(ModelEVOLVE, self).__init__()

        self.encoder = EncoderEVOLVEGCNH(num_nodes=num_nodes, in_channels=in_channels, 
                                         dropout_p=dropout_p)
        
        self.decoder = EdgeDecoder(hidden_size=hidden_size)

    def forward(self, snapshot):
        x = snapshot.x
        edge_index = snapshot.edge_index
        edge_attr = snapshot.edge_attr
        edge_weights_index = snapshot.edge_weights_index

        node_embeddings = self.encoder(x, edge_index, edge_attr)
        edge_weights_pred = self.decoder(node_embeddings, edge_weights_index)
        
        return edge_weights_pred

## A3TGCN
from pytorch_geometric_temporal_models import A3TGCN

class EncoderA3TGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                    dropout_p, periods, add_self_loops=False):

        super(EncoderA3TGCN, self).__init__()

        self.A3TGCN = A3TGCN(in_channels=in_channels,
                            out_channels=out_channels,
                            periods=periods,
                            add_self_loops=add_self_loops)

        self.dropout = Dropout(p=dropout_p)

    def forward(self, x, edge_index, edge_attr):
        x = self.dropout(x)
        node_embeddings = self.A3TGCN(x, edge_index, edge_attr)
        return node_embeddings

class ModelA3TGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, dropout_p, periods):
        super(ModelA3TGCN, self).__init__()

        self.encoder = EncoderA3TGCN(in_channels=in_channels, out_channels=hidden_size, 
                                     dropout_p=dropout_p, periods=periods)
        
        self.decoder = EdgeDecoder(hidden_size=hidden_size)

    def forward(self, snapshot):
        x = snapshot.x.unsqueeze(2)
        edge_index = snapshot.edge_index
        edge_attr = snapshot.edge_attr
        edge_weights_index = snapshot.edge_weights_index

        node_embeddings = self.encoder(x, edge_index, edge_attr)
        edge_weights_pred = self.decoder(node_embeddings, edge_weights_index)
        
        return edge_weights_pred

## GATR
from pytorch_geometric_temporal_models import GATEncoder, TransEncoder

class ModelGATR(torch.nn.Module):

    def __init__(self, in_channels, hidden_size, num_heads_GAT, dropout_p_GAT,
                 edge_dim_GAT, momentum_GAT, num_heads_TR,
                 num_encoder_layers_TR, num_decoder_layers_TR,
                 dropout_p_TR):

        super(ModelGATR, self).__init__()

        self.encoder = GATEncoder(in_channels, hidden_size, num_heads_GAT,
                                  dropout_p_GAT, edge_dim_GAT,
                                  momentum_GAT)

        hidden_size = hidden_size*num_heads_GAT
        self.transformer = TransEncoder(hidden_size,
                                       num_heads_TR,
                                       num_encoder_layers_TR,
                                       num_decoder_layers_TR, dropout_p_TR)
        
        self.decoder = EdgeDecoder(hidden_size)

    def forward(self, lagged_snapshots):
        static_embeddings_list = []
        for snapshot in lagged_snapshots:
            x = snapshot.x
            edge_index = snapshot.edge_index
            edge_attr = snapshot.edge_attr
            edge_weights = snapshot.edge_weights
            edge_attr = torch.cat((edge_attr, edge_weights), dim=1)
            edge_weights_index = snapshot.edge_weights_index

            node_embeddings = self.encoder(x, edge_index, edge_attr)
            static_embeddings_list.append(node_embeddings)

        lagged_static_embeddings = torch.stack(static_embeddings_list)
        src = lagged_static_embeddings
        trg = lagged_static_embeddings[3]
        trg = trg.unsqueeze(0)
        temp_embeddings = self.transformer(src, trg)
        temp_embeddings = temp_embeddings.squeeze(0)

        edge_weights_pred = self.decoder(temp_embeddings, edge_weights_index)

        return edge_weights_pred

def GATR_train(model, graph_snapshots_train, window, optimiser, epochs):
    model.train()
    loss_epochs_df = pd.DataFrame(columns=['epoch', 'loss'])

    for epoch in tqdm(range(epochs)):
        losses_list = []

        for i in range(window+1, len(graph_snapshots_train)):
            s = graph_snapshots_train[i]
            s_lag1 = graph_snapshots_train[i-1]
            s_lag2 = graph_snapshots_train[i-2]
            s_lag3 = graph_snapshots_train[i-3]
            s_lag4 = graph_snapshots_train[i-4]
            lagged_snapshots = [s_lag4, s_lag3, s_lag2, s_lag1]
            edge_weights_pred = model(lagged_snapshots)
            
            loss = torch.mean((edge_weights_pred - s.edge_weights)**2)
            loss.backward()
            losses_list.append(loss.item())
            optimiser.step()
            optimiser.zero_grad()
        
        avg_epoch_loss = mean(losses_list)
        loss_epochs_df.loc[len(loss_epochs_df)]={'epoch':epoch+1, 'loss':avg_epoch_loss}

    return loss_epochs_df

def GATR_test(model, graph_snapshots_test):
    model.eval()
    loss_snapshots_df = pd.DataFrame(columns=['snapshot', 'loss'])

    for i in tqdm(range(len(graph_snapshots_test))):
        s = graph_snapshots_test[i]
        s_lag1 = graph_snapshots_test[i-1]
        s_lag2 = graph_snapshots_test[i-2]
        s_lag3 = graph_snapshots_test[i-3]
        s_lag4 = graph_snapshots_test[i-4]
        lagged_snapshots = [s_lag4, s_lag3, s_lag2, s_lag1]

        edge_weights_pred = model(lagged_snapshots)

        loss = torch.mean((edge_weights_pred - s.edge_weights)**2)

        loss_snapshots_df.loc[len(loss_snapshots_df)] = {'snapshot':i+1, 'loss':loss}
    
    return loss_snapshots_df

## TGN
from torch_geometric.nn import TGNMemory, TransformerConv

class TGNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc, node_features_dim):
        super().__init__()
        self.time_enc = time_enc
        self.node_features_dim = node_features_dim
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels + node_features_dim, 
                                    out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg, node_features):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        x_with_features = torch.cat([node_features, x], dim=-1)
        return self.conv(x_with_features, edge_index, edge_attr)

class TGNDecoder(torch.nn.Module):
    def __init__(self, hidden_size):
        super(TGNDecoder, self).__init__()

        self.lin1 = torch.nn.Linear(hidden_size * 2, hidden_size, bias=True)
        
        self.lin2 = torch.nn.Linear(hidden_size, 1)

    def forward(self, edge_embeddings):
        h = self.lin1(edge_embeddings)
        h = F.relu(h)
        prediction = self.lin2(h)
        return prediction.view(-1)
    
def TGN_train(memory, encoder, decoder, neighbor_loader, 
              train_loader, node_features, edge_weights, 
              optimizer, device, assoc, data, train_data):
    memory.train()
    encoder.train()
    decoder.train()

    memory.reset_state()
    neighbor_loader.reset_state()

    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        node_features_batch = torch.stack([node_features[last_update[i].item()][i] 
                                           for i in range(len(last_update))])
        
        z = encoder(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device), node_features_batch)
        
        edge_embeddings_pos = torch.cat([z[assoc[batch.src]],  z[assoc[batch.dst]]], dim = 1)
        edge_embeddings_neg = torch.cat([z[assoc[batch.src]],  z[assoc[batch.dst]]], dim = 1)

        pos_out = decoder(edge_embeddings_pos)
        neg_out = decoder(edge_embeddings_neg)
        
        if e_id.nelement() != 0:
            loss = torch.mean((pos_out - edge_weights[e_id])**2)
        else:
            loss = 0.0
        
        loss += torch.mean((neg_out - torch.zeros_like(neg_out))**2)

        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events

@torch.no_grad()
def TGN_test(memory, encoder, decoder, neighbor_loader, 
              test_loader, node_features, edge_weights, 
              optimizer, device, assoc, data, test_data):
    memory.eval()
    encoder.eval()
    decoder.eval()

    torch.manual_seed(12345)

    total_loss = 0
    for batch in test_loader:
        optimizer.zero_grad()
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        node_features_batch = torch.stack([node_features[last_update[i].item()][i] 
                                           for i in range(len(last_update))])
        
        z = encoder(z, last_update, edge_index, data.t[e_id].to(device),
                    data.msg[e_id].to(device), node_features_batch)
        
        edge_embeddings_pos = torch.cat([z[assoc[batch.src]],  z[assoc[batch.dst]]], dim = 1)
        edge_embeddings_neg = torch.cat([z[assoc[batch.src]],  z[assoc[batch.dst]]], dim = 1)

        pos_out = decoder(edge_embeddings_pos)
        neg_out = decoder(edge_embeddings_neg)
                
        if e_id.nelement() != 0:
            loss = torch.mean((pos_out - edge_weights[e_id])**2)
        else:
            loss = 0.0

        loss += torch.mean((neg_out - torch.zeros_like(neg_out))**2)
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
        total_loss += float(loss) * batch.num_events

    return total_loss / test_data.num_events