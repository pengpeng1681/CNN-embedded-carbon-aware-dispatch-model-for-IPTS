import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neural_network import MLPRegressor
import warnings

import gurobipy as gp
from gurobipy import GRB
from gurobi_ml import add_predictor_constr

seed1 = 1608
np.random.seed(seed1)
torch.manual_seed(seed1)
torch.cuda.manual_seed(seed1)

# Input data
casename = 'case15.xls'

bus = pd.read_excel(casename, sheet_name='bus').to_numpy()
branch = pd.read_excel(casename, sheet_name='branch').to_numpy()
gen = pd.read_excel(casename, sheet_name='gen').to_numpy()

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
pd_load = bus[:, 1]  # active power load
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


# ==================== Training begins ====================
lambda_l1_emi = 0
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

    for epoch in range(num_emi_epochs):
        emi_model.train()

        preds = emi_model(x_emi_torch)
        mse_loss = criterion_mse(preds, emi_train_torch)
        l1_loss = structured_l1_penalty(emi_model)
        total_loss = mse_loss + lambda_l1_emi * l1_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_emi_epochs}, MSE = {mse_loss.item():.6f}, "
                 f"L1 = {l1_loss.item():.6f}, total_loss = {total_loss.item():.6f}")



    nn_emission_list.append(emi_model)


sk_emission_list = []

for i, pruned_emi_model in enumerate(nn_emission_list):
    sk_emi_model = convert_pytorch_to_sklearn(pruned_emi_model)
    sk_emission_list.append(sk_emi_model)



# ============ gurobi optimization============
model = gp.Model('traffic_optimization')

# power system
pg = model.addVars(n_gen, lb=gen[:, 2], ub=gen[:, 3], name="pg")
qg = model.addVars(n_gen, lb=gen[:, 4], ub=gen[:, 5], name="qg")
pf = model.addVars(n_branch, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="pf")
qf = model.addVars(n_branch, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="qf")
Vm = model.addVars(n_bus, lb=0 ** 2, ub=bus[:, 4] ** 2, name="Vm")

# transportation system
xa = model.addMVar(m_link, lb=0, name='xa')  # Traffic flow per link
xa_ev = model.addMVar(m_link, lb=0, name='xa_ev')  # EV traffic flow
xa_gv = model.addMVar(m_link, lb=0, name='xa_gv')  # GV traffic flow

ta = model.addMVar(m_link, lb=0, name='ta')  # Travel time per link


fod1_ev = model.addMVar(link_juzhen.shape[1], lb=0, name='fod1_ev')  # OD pair flow variables
fod1_gv = model.addMVar(link_juzhen.shape[1], lb=0, name='fod1_gv')  # OD pair flow variables

urs_ev = model.addVar(lb=0, name='urs_ev')  # Minimum travel time for OD pairs
urs_gv = model.addVar(lb=0, name='urs_gv')  # Minimum travel time for OD pairs

t1 = link_juzhen @ ta  # Matrix multiplication for travel time calculation

crs1_gv = link_juzhen @ ta * w_ts / 60  # Matrix multiplication for travel time calculation
crs1_ev = link_juzhen @ (ta * w_ts / 60 + Eev * 1000 * c_price)  # Travel time including charging cost


# power system constraints
# power balance constraint
for i in range(n_bus):
    from_branches = From_bus[i + 1]
    to_branches = To_bus[i + 1]

    pf_out = gp.quicksum(pf[j] for j in from_branches)
    pf_in = gp.quicksum(pf[j] for j in to_branches)
    qf_out = gp.quicksum(qf[j] for j in from_branches)
    qf_in = gp.quicksum(qf[j] for j in to_branches)

    if (i + 1) in location_cs:
        model.addConstr(
            pf_out + pd_load[i] / basemva + Eev * xa_ev[location_cs.index(i + 1)] == pf_in + pg[i] / basemva,
            name=f"power_balance_active_{i}"
        )
    else:
        model.addConstr(
            pf_out + pd_load[i] / basemva == pf_in + pg[i] / basemva,
            name=f"power_balance_active_{i}"
        )

    model.addConstr(
        qf_out + qd[i] / basemva == qf_in + qg[i] / basemva,
        name=f"power_balance_reactive_{i}"
    )

# voltage constraint
for i in range(n_branch):
    model.addConstr(
        Vm[int(branch[i, 1]) - 1] - Vm[int(branch[i, 2]) - 1] - 2 * (r[i] * pf[i] + x[i] * qf[i]) == 0,
        name=f"voltage_drop_{i}"
    )


# transportation system constraints
# 1.Coupling constraint of links and road
model.addConstr(link_juzhen.T @ fod1_ev == xa_ev, name='link_path_flow_ev')
model.addConstr(link_juzhen.T @ fod1_gv == xa_gv, name='link_path_flow_gv')

# 2.traffic demand constraint
model.addConstr(fod1_ev.sum() == qrs * pi_ev, name='traffic_flow_ev')
model.addConstr(fod1_gv.sum() == qrs * (1 - pi_ev), name='traffic_flow_gv')

# 3.total traffic flow constraint
model.addConstr(xa == xa_ev + xa_gv, name='traffic_flow_total')

# 4.complementary equilibrium constraint
M3 = 100
vrs1_ev = model.addVars(link_juzhen.shape[1], vtype=GRB.BINARY, name='vrs1_ev')
vrs1_gv = model.addVars(link_juzhen.shape[1], vtype=GRB.BINARY, name='vrs1_gv')

for k in range(link_juzhen.shape[1]):
    # ev constraints
    model.addConstr(fod1_ev[k] >= 0, name=f'fod1_nonnegative_{k}')
    model.addConstr(fod1_ev[k] <= M3 * (1 - vrs1_ev[k]), name=f'relaxation_flow_{k}')
    model.addConstr(crs1_ev[k] - urs_ev >= 0, name=f't1_urs_nonnegative_{k}')
    model.addConstr(crs1_ev[k] - urs_ev <= M3 * vrs1_ev[k], name=f'relaxation_cost_{k}')

    # gv constraints
    model.addConstr(fod1_gv[k] >= 0, name=f'fod1_nonnegative_{k}')
    model.addConstr(fod1_gv[k] <= M3 * (1 - vrs1_gv[k]), name=f'relaxation_flow_{k}')
    model.addConstr(crs1_gv[k] - urs_gv >= 0, name=f't1_urs_nonnegative_{k}')
    model.addConstr(crs1_gv[k] - urs_gv <= M3 * vrs1_gv[k], name=f'relaxation_cost_{k}')


# 5.traffic time constraint
#Auxiliary variable
u_xa = model.addMVar(m_link, lb=0, name='u_xa')
v_xa = model.addMVar(m_link, lb=0, name='v_xa')
w_xa = model.addMVar(m_link, lb=0, name='w_xa')

for i in range(m_link):
    model.addConstr(u_xa[i] == xa[i] / ca[i], name=f'u_xa_constr_{i}')
    model.addConstr(v_xa[i] == u_xa[i] ** 2, name=f'v_xa_constr_{i}')
    model.addConstr(w_xa[i] == v_xa[i] ** 2, name=f'w_xa_constr_{i}')
    model.addConstr(ta[i] == t0[i] * (1 + 0.15 * w_xa[i]), name=f'ta_constr_{i}')

base_num_vars = model.NumVars
base_num_bin_vars = model.NumBinVars
base_num_constrs = model.NumConstrs


# 6.carbon emission constraints approximated by NN
emission_ts_a_norm = [model.addVar(lb=0, name=f'emission_ts_a_norm{i}') for i in range(m_link)]
emission_ts_a = [model.addVar(lb=0, name=f'emission_ts_a_{i}') for i in range(m_link)]

xa_norm = [model.addVar(lb=0, name=f'xa_norm{i}') for i in range(m_link)]  #
xa_gv_norm = [model.addVar(lb=0, name=f'xa_gv_norm{i}') for i in range(m_link)]  #


for i in range(m_link):
    add_predictor_constr(model, sk_emission_list[i], [xa_gv_norm[i], xa_norm[i]], [emission_ts_a_norm[i]])

    model.addConstr(xa[i] == xa_norm[i] * 30,name=f'de_norm_{i}')
    model.addConstr(xa_gv[i] == xa_gv_norm[i] * 30, name=f'de_norm_{i}')

    model.addConstr(emission_ts_a[i] == emission_ts_a_norm[i] * (emi_max[i] - emi_min[i]) + emi_min[i], name=f'de_norm_{i}')

new_num_vars = model.NumVars
new_num_bin_vars = model.NumBinVars
new_num_constrs = model.NumConstrs


emission_cof = 1
carbon_price = 48
# Objective function
# power system
obj_pg_gen = gp.quicksum(gen[j, 6] * pg[j] ** 2 for j in range(n_gen)) + gp.quicksum(gen[j, 7] * pg[j] for j in range(n_gen))
obj_pg_carbon = gp.quicksum(pg[j] * emi_para[j] for j in range(n_gen)) * carbon_price

obj_ps = obj_pg_gen + obj_pg_carbon * emission_cof


# transportation system
obj_ts_gv = gp.quicksum((ta * w_ts / 60) * xa_gv * baseTS)
obj_ts_ev = gp.quicksum(((ta * w_ts / 60 + Eev * 1000 * c_price)) * xa_ev * baseTS)
obj_ts_time1 = obj_ts_gv + obj_ts_ev

obj_ts_gv2 = urs_gv * qrs * (1 - pi_ev) * baseTS
obj_ts_ev2 = urs_ev * qrs * pi_ev * baseTS
obj_ts_time2 = obj_ts_gv2 + obj_ts_ev2

obj_ts_emission = gp.quicksum(emission_ts_a) * 1e-6 * carbon_price * baseTS / suo

obj_ts = obj_ts_time1 + obj_ts_emission * emission_cof

# Total objective function
obj_zong = obj_ps + obj_ts

model.setObjective(obj_zong, GRB.MINIMIZE)

model.Params.MIPGap = 1e-5
# model.Params.TimeLimit = 100

# Solve
model.optimize()

carbon_emission_values = [var.X / suo for var in emission_ts_a]

if model.status == GRB.OPTIMAL:

    pg_val = [pg[i].x for i in range(n_gen)]
    qg_val = [qg[i].x for i in range(n_gen)]
    Vm_val = [Vm[i].x for i in range(n_bus)]
    pf_val = [pf[i].x for i in range(n_branch)]
    qf_val = [qf[i].x for i in range(n_branch)]


    obj_pg_gen_val = sum(gen[j, 6] * pg_val[j] ** 2 for j in range(n_gen)) + sum(
        gen[j, 7] * pg_val[j] for j in range(n_gen))
    obj_pg_carbon_val = sum(pg_val[j] * emi_para[j] for j in range(n_gen)) * carbon_price
    obj_ps_val = obj_pg_gen_val + obj_pg_carbon_val



    xa_values = xa.X
    ta_values = ta.X
    obj_value = model.objVal
    xa_gv_value = xa_gv.X

    obj_ts_gv2_value = (urs_gv.X * qrs * (1 - pi_ev) * baseTS)
    obj_ts_ev2_value = (urs_ev.X * qrs * pi_ev * baseTS)

    obj_ts_emission_value = sum(var.X for var in emission_ts_a) * 1e-6 * carbon_price * baseTS / suo

    # Output
    print('Optimal solution found:')
    print(f"PS total: {obj_ps_val}")
    print(f"PS generation cost: {obj_pg_gen_val}")
    print(f"PS carbon cost: {obj_pg_carbon_val}")

    print('xa =', xa_values)
    print('xa_gv =', xa_gv_value)
    print('ta:', ta_values)
    print('TS Objective:', obj_ts_gv2_value, obj_ts_ev2_value, obj_ts_emission_value, obj_value - obj_ps_val)
    print('Carbon emission values:', carbon_emission_values)
else:
    print('No optimal solution found.')

if model.status == GRB.INFEASIBLE:
    model.computeIIS()
    model.write("model.ilp")
    print("Infeasible model. Irreducible Inconsistent Subsystem (IIS) written to 'model.ilp'")


for i in range(m_link):
    layer_purned = count_nodes(nn_emission_list[i])
    para_purned = count_parameters(nn_emission_list[i])
    print(f"The number of layers after pruning: ({layer_purned[0]},{layer_purned[1]}), Parameters after pruning: {para_purned}")



