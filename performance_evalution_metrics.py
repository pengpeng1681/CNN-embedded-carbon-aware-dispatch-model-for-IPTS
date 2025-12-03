import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neural_network import MLPRegressor
import warnings

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


import gurobipy as gp
from gurobipy import GRB
from gurobi_ml import add_predictor_constr

import time

seed1 = 1608
np.random.seed(seed1)
torch.manual_seed(seed1)
torch.cuda.manual_seed(seed1)

# Input data
casename = 'case15.xls'

bus = pd.read_excel(casename, sheet_name='bus').to_numpy()
branch = pd.read_excel(casename, sheet_name='branch').to_numpy()
gen = pd.read_excel(casename, sheet_name='gen').to_numpy()
# load = pd.read_excel(casename, sheet_name='load2').to_numpy()

# Route data [ID, start node, end node, capacity, free-flow time]
data_route = pd.read_excel(casename, sheet_name='route', header=None)
data_OD = pd.read_excel(casename, sheet_name='OD', header=None)
data_route = data_route.apply(pd.to_numeric, errors='coerce').dropna().values
data_OD = data_OD.apply(pd.to_numeric, errors='coerce').dropna().values
# Path-link incidence matrix
link_juzhen = pd.read_excel(casename, sheet_name='link_juzhen', header=None).values


n_bus = bus.shape[0]        # number of buses
n_branch = branch.shape[0]  # number of branches
n_gen = gen.shape[0]        # number of generators

basemva = 10   # base power (10 MVA)
baseV = 10     # base voltage (10 kV)

# Charging station locations
location_cs = [2, 4, 6, 10, 12]

# Extract the from and to bus indices of branches
From_bus = {i: [] for i in range(1, n_bus + 1)}
To_bus = {i: [] for i in range(1, n_bus + 1)}

for j in range(n_branch):
    From_bus[int(branch[j, 1])].append(j)
    To_bus[int(branch[j, 2])].append(j)

r = branch[:, 3] / (baseV ** 2 / basemva)  # resistance
x = branch[:, 4] / (baseV ** 2 / basemva)  # reactance
#pd = bus[:, 1]  # active power load
qd = bus[:, 2]  # reactive power load
b_limit = branch[:, 5]  # power flow limit

emi_para = gen[:, 9]

# Transportation system data
m_link, n = data_route.shape
n_node = int(max(data_route[:, 1].max(), data_route[:, 2].max()))
ca = data_route[:, 3] # link capacity
t0 = data_route[:, 4]  # free-flow travel time
la = data_route[:, 5]  # length of link (km)
qrs = data_OD[:, 3]
t0_h = t0 / 60 # free-flow travel time converted to hours
la_mph = 0.621371 * la # length converted from km to miles (mph-compatible)

pi_ev = 0.2  # EV penetration rate
baseTS = 10  # traffic base value (1 p.u. corresponds to 10 vehicles)
Eev = 0.03  # EV charging power per vehicle, MW
c_price = 0.1  # Charging price, $/kWh
suo = 1
w_ts = 5

# Road network topology matrix G
G = np.inf * np.ones((n_node, n_node))
np.fill_diagonal(G, 0)

for i in range(m_link):
    start_node = int(data_route[i, 1]) - 1
    end_node = int(data_route[i, 2]) - 1
    if G[start_node, end_node] == np.inf:
        G[start_node, end_node] = 1
    else:
        G[start_node, end_node] += 1

if link_juzhen.shape[0] != m_link:
    link_juzhen = link_juzhen[:m_link, :]

# ============ Data and network definitions ============
def calculate_time(t_0_a, cap_a, x_a):
    t_a = t_0_a * (1 + 0.15 * (x_a / cap_a) ** 4)
    return t_a

# Calculate carbon emissions
def calculate_emission(x_gv, x_a, t0_a, cap_a, l_a):
    t_a = t0_a * (1 + 0.15 * (x_a / cap_a) ** 4)
    v_a = l_a / t_a
    e_ts_a = 129.533 + 2217.694 / v_a + 0.02771 * (v_a ** 2)
    em_ts_a = x_gv * l_a * e_ts_a
    return em_ts_a


class MLP_emission(nn.Module):
    def __init__(self):
        super(MLP_emission, self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 100)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x

def structured_l1_penalty(m):
    penalty = 0.0
    for layer in [m.fc1, m.fc2, m.fc3]:
        row_norms = torch.norm(layer.weight, p=2, dim=1)
        penalty += torch.sum(row_norms)
    return penalty

def count_parameters(m):
    return sum(p.numel() for p in m.parameters())

def count_nodes(m):
    return [m.fc1.out_features, m.fc2.out_features]

def prune_model_by_threshold(old_model, threshold):

    import copy
    new_model = copy.deepcopy(old_model)

    W1, b1 = new_model.fc1.weight.data, new_model.fc1.bias.data
    row_norms_1 = torch.norm(W1, p=2, dim=1)
    keep_idx_1 = (row_norms_1 > threshold).nonzero(as_tuple=True)[0]

    W1_new = W1[keep_idx_1, :]
    b1_new = b1[keep_idx_1]
    fc1_new = nn.Linear(new_model.fc1.in_features, len(keep_idx_1), bias=True)
    fc1_new.weight = nn.Parameter(W1_new)
    fc1_new.bias   = nn.Parameter(b1_new)
    new_model.fc1 = fc1_new

    W2, b2 = new_model.fc2.weight.data, new_model.fc2.bias.data
    W2_in_pruned = W2[:, keep_idx_1]

    row_norms_2 = torch.norm(W2_in_pruned, p=2, dim=1)
    keep_idx_2 = (row_norms_2 > threshold).nonzero(as_tuple=True)[0]
    W2_new = W2_in_pruned[keep_idx_2, :]
    b2_new = b2[keep_idx_2]
    fc2_new = nn.Linear(len(keep_idx_1), len(keep_idx_2), bias=True)
    fc2_new.weight = nn.Parameter(W2_new)
    fc2_new.bias   = nn.Parameter(b2_new)
    new_model.fc2 = fc2_new

    W3, b3 = new_model.fc3.weight.data, new_model.fc3.bias.data
    W3_new = W3[:, keep_idx_2]
    fc3_new = nn.Linear(len(keep_idx_2), new_model.fc3.out_features, bias=True)
    fc3_new.weight = nn.Parameter(W3_new)
    fc3_new.bias   = nn.Parameter(b3)
    new_model.fc3 = fc3_new

    return new_model

def convert_pytorch_to_sklearn(pytorch_model):

    layers = []
    for module in pytorch_model.modules():
        if isinstance(module, torch.nn.Linear):
            layers.append(module.out_features)

    hidden_layer_sizes = tuple(layers[:-1])

    sk_model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        max_iter=1,
        random_state=0
    )

    dummy_X = np.random.rand(2, pytorch_model.fc1.in_features)
    dummy_y = np.random.rand(2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=Warning)
        sk_model.fit(dummy_X, dummy_y)

    pytorch_layers = [module for module in pytorch_model.modules() if isinstance(module, torch.nn.Linear)]

    for i, layer in enumerate(pytorch_layers):
        weight = layer.weight.detach().numpy().T
        bias = layer.bias.detach().numpy()

        sk_model.coefs_[i] = weight
        sk_model.intercepts_[i] = bias

    return sk_model

# --------- Linear regression model ---------
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)


# ==================== Training begins ====================
lambda_l1_emi = 0.0001
threshold_emi = 0.01

num_emi_epochs = 1000
lr = 0.001

num_samples = 10000

x_train = np.random.uniform(0, 30, 10000).reshape(-1, 1)  # x in range [0, 30]

xa_gv_train = np.random.uniform(0, 30, num_samples).reshape(-1, 1)

num_zero = 400
xa_gv_train[:num_zero, 0] = 0.0

xa_train = np.array([np.random.uniform(x_gv, 30) for x_gv in xa_gv_train]).reshape(-1, 1)
x_train_emission_au = np.hstack((xa_gv_train, xa_train))

x_train_emission_norm = x_train_emission_au / 30


emi_min = []
emi_max = []

nn_time_list = []
nn_emission_list = []

train_epoch = []
fine_train_epoch = []

fine_metrics_mse = []
fine_metrics_mae = []
fine_metrics_r2 = []

train_metrics_mse = []
train_metrics_mae = []
train_metrics_r2 = []

lin_mse_list = []
lin_mae_list = []
lin_r2_list = []

dt_mse_list = []
dt_mae_list = []
dt_r2_list = []

for i in range(m_link):
   # ==============Neural networks for calculating carbon emissions===============
    print(f"\n==== Train the carbon emission of the {i + 1} th link ====")
    e = calculate_emission(xa_gv_train, xa_train, t0_h[i], ca[i], la_mph[i]).reshape(-1, 1) * suo

    min_val = e.min()
    max_val = e.max()
    emi_min.append(min_val)
    emi_max.append(max_val)

    emi_range = max_val - min_val
    if emi_range == 0:
        emi_range = 1e-9

    emission_train_au = (e - min_val) / emi_range
    emission_train_norm = emission_train_au.reshape(-1, 1).astype(np.float32)

    x_emi_torch = torch.tensor(x_train_emission_norm, dtype=torch.float32)
    emi_train_torch = torch.tensor(emission_train_norm, dtype=torch.float32)

    emi_model = MLP_emission()
    optimizer = optim.Adam(emi_model.parameters(), lr=lr)
    criterion_mse = nn.MSELoss()

    loss_records = []

    for epoch in range(num_emi_epochs):
        emi_model.train()

        preds = emi_model(x_emi_torch)
        mse_loss = criterion_mse(preds, emi_train_torch)
        l1_loss = structured_l1_penalty(emi_model)
        total_loss = mse_loss + lambda_l1_emi * l1_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss_records.append(mse_loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_emi_epochs}, MSE = {mse_loss.item():.6f}, "
                 f"L1 = {l1_loss.item():.6f}, total_loss = {total_loss.item():.6f}")

    train_epoch.append(loss_records)

    # ======= Calculate the metrics after 1,000 training sessions (initial training) =======
    emi_model.eval()
    with torch.no_grad():
        preds_train = emi_model(x_emi_torch)
    preds_train_np = preds_train.numpy().flatten()
    true_np = emi_train_torch.numpy().flatten()
    mse_train = np.mean((preds_train_np - true_np) ** 2)
    mae_train = np.mean(np.abs(preds_train_np - true_np))
    r2_train = 1 - np.sum((true_np - preds_train_np) ** 2) / np.sum((true_np - np.mean(true_np)) ** 2)
    train_metrics_mse.append(mse_train)
    train_metrics_mae.append(mae_train)
    train_metrics_r2.append(r2_train)


    pruned_emi_model = prune_model_by_threshold(emi_model, threshold_emi)

    # fine-tuning after pruning, default 100 epochs
    finetune_epochs = 100
    optimizer_finetune = optim.Adam(pruned_emi_model.parameters(), lr=0.001)
    pruned_emi_model.train()


    for epoch in range(finetune_epochs):
        preds = pruned_emi_model(x_emi_torch)
        mse_loss = criterion_mse(preds, emi_train_torch)
        optimizer_finetune.zero_grad()
        mse_loss.backward()
        optimizer_finetune.step()


        if (epoch + 1) % 100 == 0:
            print(f"Fine-Purning-Poch {epoch + 1}/{finetune_epochs}, MSE = {mse_loss.item():.6f}, ")

    nn_emission_list.append(pruned_emi_model)

    pruned_emi_model.eval()
    with torch.no_grad():
        preds = pruned_emi_model(x_emi_torch)
    preds_np = preds.numpy().flatten()
    true_np = emi_train_torch.numpy().flatten()
    mse_val = np.mean((preds_np - true_np) ** 2)
    mae_val = np.mean(np.abs(preds_np - true_np))
    r2_val = 1 - np.sum((true_np - preds_np) ** 2) / np.sum((true_np - np.mean(true_np)) ** 2)
    fine_metrics_mse.append(mse_val)
    fine_metrics_mae.append(mae_val)
    fine_metrics_r2.append(r2_val)

    # =============== Linear regression ================
    X_np = x_emi_torch.numpy()  # shape (n, 2)
    y_np = emi_train_torch.numpy().ravel()  # shape (n, )

    linreg = LinearRegression()
    linreg.fit(X_np, y_np)
    preds_lin = linreg.predict(X_np)

    mse_lin = np.mean((preds_lin - y_np) ** 2)
    mae_lin = np.mean(np.abs(preds_lin - y_np))
    r2_lin = 1 - np.sum((y_np - preds_lin) ** 2) / np.sum((y_np - np.mean(y_np)) ** 2)

    lin_mse_list.append(mse_lin)
    lin_mae_list.append(mae_lin)
    lin_r2_list.append(r2_lin)
    print("Linear regression end")


    # =============== Decision tree regression ===============
    dt_reg = DecisionTreeRegressor(max_depth=5, random_state=0)
    dt_reg.fit(X_np, y_np)
    preds_dt = dt_reg.predict(X_np)

    mse_dt = np.mean((preds_dt - y_np) ** 2)
    mae_dt = np.mean(np.abs(preds_dt - y_np))
    r2_dt = 1 - np.sum((y_np - preds_dt) ** 2) / np.sum((y_np - np.mean(y_np)) ** 2)

    dt_mse_list.append(mse_dt)
    dt_mae_list.append(mae_dt)
    dt_r2_list.append(r2_dt)
    print("Decision tree regression end")




cols = [f"Link{i+1}" for i in range(len(train_metrics_mse))]

# 1) Metrics of initial NN
train_data = {cols[i]: [train_metrics_mse[i], train_metrics_mae[i], train_metrics_r2[i]] for i in range(len(train_metrics_mse))}
df_train_metrics = pd.DataFrame(train_data, index=["MSE", "MAE", "R^2"])

# 2) Metrics of NN after fine-tuning
fine_data = {cols[i]: [fine_metrics_mse[i], fine_metrics_mae[i], fine_metrics_r2[i]] for i in range(len(fine_metrics_mse))}
df_fine_metrics = pd.DataFrame(fine_data, index=["MSE", "MAE", "R^2"])

# 3) Metrics of linear regression
df_lin = pd.DataFrame({cols[i]: [lin_mse_list[i],
                                 lin_mae_list[i],
                                 lin_r2_list[i]] for i in range(len(lin_mse_list))},
                      index=["MSE", "MAE", "R^2"])

# 4) Metrics of decision tree
df_dt = pd.DataFrame({cols[i]: [dt_mse_list[i],
                                dt_mae_list[i],
                                dt_r2_list[i]] for i in range(len(dt_mse_list))},
                     index=["MSE", "MAE", "R^2"])



# ==================== Print Table ====================
cols = [f"Link {i+1}" for i in range(len(train_metrics_mse))]

def _fmt(metric: str, v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if metric == "MSE":
        return f"{v:.2e}"
    if metric == "MAE":
        return f"{v:.4f}"
    if metric == "R^2":
        return f"{v:.4f}"
    return f"{v:.6g}"

rows = []
metric_specs = [
    ("MSE",  fine_metrics_mse,  train_metrics_mse,  lin_mse_list,  dt_mse_list),
    ("MAE",  fine_metrics_mae,  train_metrics_mae,  lin_mae_list,  dt_mae_list),
    ("R^2",  fine_metrics_r2,   train_metrics_r2,   lin_r2_list,   dt_r2_list),
]

for i, link_name in enumerate(cols):
    for k, (metric, cnn_list, nn_list, lr_list, dt_list) in enumerate(metric_specs):
        rows.append({
            "Link":  link_name if k == 0 else "",
            "Metric": metric,
            "CNN": _fmt(metric, cnn_list[i]),  # Metrics of NN after fine-tuning
            "NN":  _fmt(metric, nn_list[i]),   # Metrics of initial NN
            "LR":  _fmt(metric, lr_list[i]),
            "DT":  _fmt(metric, dt_list[i]),
        })

    rows.append({"Link": "", "Metric": "", "CNN": "", "NN": "", "LR": "", "DT": ""})

df_table2 = pd.DataFrame(rows)
df_table2 = df_table2.iloc[:-1]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

print("Comparison of approximation errors for different methods\n")
print(df_table2.to_string(index=False))



