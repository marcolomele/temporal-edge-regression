# temporal-edge-regression

This repository provides a framework for temporal edge regression using advanced Graph Neural Network (GNN) architectures. The project focuses on predicting edge weights in dynamic graphs, with applications to both pandemic mobility data (Italy, Spain, France, England). The codebase includes data processing, graph generation, and several state-of-the-art GNN models for temporal edge prediction. This project was an experiment I did out of curiosity while writing my theoretical my thesis: Graph Neural Networks – From Foundations to Temporal Applications. 

## 📂 Folder Structure

```
temporal-edge-regression/
│
├── data_pandemic/
│   ├── Italy/
│   │   ├── italy_labels.csv
│   │   └── graphs/         # Daily graph snapshots (IT_YYYY-MM-DD.csv)
│   ├── Spain/
│   │   ├── spain_labels.csv
│   │   └── graphs/         # Daily graph snapshots (ES_YYYY-MM-DD.csv)
│   ├── France/
│   │   ├── france_labels.csv
│   │   └── graphs/         # Daily graph snapshots (FR_YYYY-MM-DD.csv)
│   └── England/
│       ├── england_labels.csv
│       └── graphs/         # Daily graph snapshots (EN_YYYY-MM-DD.csv)
│
├── trade_data.pkl          # Preprocessed trade and stock data (pickle)
├── data_processing_pandemic.py
├── graph_generation_pandemic.py
├── graph_generation_trade.py
├── models_and_utils.py     # Model definitions and training utilities
├── pytorch_geometric_temporal_models.py
├── train_test_trade.ipynb  # Example notebook for trade data
├── bsc-thesis-marco-lomele.pdf # Thesis related to temmporal edge regression
├── LICENSE
└── README.md
```

## 💽 Data

- **Pandemic Mobility Data**:  
  - Daily graphs for Italy, Spain, France, and England, with node features (e.g., cases) and edge weights (e.g., movement).
  - Labels files (e.g., `italy_labels.csv`) contain city-level time series.

- **Trade Data**:  
  - `trade_data.pkl` contains trade and stock data, preprocessed for graph construction.

## 🧪 Models

The repository implements several temporal GNN architectures for edge regression:

- **MPNN-LSTM**: Message Passing Neural Network with LSTM for temporal encoding.
- **EvolveGCN-H**: Evolving Graph Convolutional Network with hidden state evolution.
- **A3TGCN**: Attention Temporal Graph Convolutional Network.
- **GATR**: GAT + Transformer for static and temporal node embeddings.
- **TGN**: Temporal Graph Networks (for event-based dynamic graphs).

Model definitions are in `models_and_utils.py` and `pytorch_geometric_temporal_models.py`.

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/marcolomele/temporal-edge-regression.git
   cd temporal-edge-regression
   ```

2. **Install dependencies:**

   This project requires Python 3.8+ and the following main packages:
   - torch
   - torch-geometric
   - pandas
   - numpy
   - matplotlib
   - tqdm

   You can install them with:
   ```bash
   pip install torch torch-geometric pandas numpy matplotlib tqdm
   ```

   > Note: For `torch-geometric`, follow the official [installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your system and CUDA version.

## 🛠️ Usage

#### 1. **Pandemic Data Processing & Graph Generation**

- Preprocess and generate graph snapshots for each country:
  ```python
  # Example: data_processing_pandemic.py and graph_generation_pandemic.py
  ```

#### 2. **Trade Data Graph Generation**

- Generate graph snapshots from trade data:
  ```python
  from graph_generation_trade import dict_snapshots_trade, temporal_data_trade
  ```

#### 3. **Model Training**

- Example (see `train_test_trade.ipynb` for full workflow):

  ```python
  from models_and_utils import ModelMPNN, train_snapshots, data_split

  # Prepare data
  data = list(dict_snapshots_trade.values())
  data_train, data_test = data_split(data, split_ratio=0.9)

  # Initialize model
  model = ModelMPNN(in_channels, hidden_size, num_nodes, window, dropout_p)
  optimiser = torch.optim.Adam(model.parameters(), lr=0.5)

  # Train
  results = train_snapshots(model, data_train, optimiser, epochs=50)
  ```

- For other models (EvolveGCN, A3TGCN, GATR), see the respective class in `models_and_utils.py`.

## 🔑 License

This project is licensed under the MIT License.  
See [LICENSE](./LICENSE) for details.

## 📚 References

- [PyTorch Geometric Temporal](https://pytorch-geometric-temporal.readthedocs.io/en/latest/)
- [Transfer Graph Neural Networks for Pandemic Forecasting](https://arxiv.org/abs/2009.08388)
- [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/abs/1902.10191)
- [A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting](https://arxiv.org/abs/2006.11583)