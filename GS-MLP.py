# AI_MPM
#some good projects for AI-MPM using Optimized MLP
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("💻 اجرا روی:", device)

train_path = input("آدرس فایل آموزشی را وارد کنید: ")
main_path = input("آدرس فایل داده اصلی را وارد کنید: ")

train_df = pd.read_csv(train_path)
main_df = pd.read_csv(main_path)

features = ['g1','g2','g3','g4','r1','r2','r3','r4','r5','r6','a1','a2','a3','a4','f1','f2']

X_train_raw = train_df[features]
y_train = train_df['label']
X_main_raw = main_df[features]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_main = scaler.transform(X_main_raw)

activation_funcs = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'logistic': nn.Sigmoid()}
solvers = ['adam', 'sgd']
learning_rates = ['constant', 'adaptive']

n1_range = list(range(0, 257, 64))
n2_range = list(range(0, 129, 32))
n3_range = list(range(0, 65, 16))
alpha_range = [1e-5, 1e-4, 1e-3, 1e-2]
lr_init_range = [1e-4, 1e-3, 1e-2]

def fitness(solution):
    n1, n2, n3, act_idx, solver_idx, lr_idx, alpha, lr_init = solution
    hidden_sizes = [int(n) for n in (n1, n2, n3) if n > 0] or [32]
    act_func_key = list(activation_funcs.keys())[act_idx]
    act_func = activation_funcs[act_func_key]
    solver = solvers[int(solver_idx)]

    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_sizes):
            super(MLP, self).__init__()
            layers = []
            prev_dim = input_dim
            for h in hidden_sizes:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(act_func)
                prev_dim = h
            layers.append(nn.Linear(prev_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr = torch.tensor(X_train[train_idx], dtype=torch.float32).to(device)
        y_tr = torch.tensor(y_train.iloc[train_idx].values, dtype=torch.float32).view(-1,1).to(device)
        X_val = torch.tensor(X_train[val_idx], dtype=torch.float32).to(device)
        y_val = torch.tensor(y_train.iloc[val_idx].values, dtype=torch.float32).view(-1,1).to(device)

        model = MLP(X_train.shape[1], hidden_sizes).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=alpha) if solver == 'adam' else optim.SGD(model.parameters(), lr=lr_init, weight_decay=alpha)

        best_r2 = -np.inf
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            output = model(X_tr)
            loss = criterion(output, y_tr)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                ss_res = torch.sum((y_val - val_pred) ** 2)
                ss_tot = torch.sum((y_val - torch.mean(y_val)) ** 2)
                r2 = 1 - ss_res / ss_tot
                best_r2 = max(best_r2, r2.item())

        scores.append(best_r2)

    return np.mean(scores)

param_combinations = list(itertools.product(
    n1_range, n2_range, n3_range,
    range(len(activation_funcs)),
    range(len(solvers)),
    range(len(learning_rates)),
    alpha_range,
    lr_init_range
))

print(f"🔍 تعداد ترکیب‌های آزمایشی: {len(param_combinations)}")

best_score = -np.inf
best_solution = None
history = []

for idx, solution in enumerate(param_combinations, 1):
    score = fitness(solution)
    history.append(score)
    if score > best_score:
        best_score = score
        best_solution = solution
    if idx % 50 == 0 or idx == len(param_combinations):
        print(f"[{idx}/{len(param_combinations)}] فعلاً بهترین R² = {best_score:.5f}")

n1, n2, n3, act_idx, solver_idx, lr_idx, alpha, lr_init = best_solution
print("\n✅ بهترین پارامترها:")
print("Hidden layers:", [n1, n2, n3])
print("Activation:", list(activation_funcs.keys())[act_idx])
print("Solver:", solvers[solver_idx])
print("Learning rate policy:", learning_rates[lr_idx])
print("Alpha:", alpha)
print("Learning rate init:", lr_init)
print("Best R² score:", best_score)

final_hidden = [n for n in (n1, n2, n3) if n > 0] or [32]
act_func_key = list(activation_funcs.keys())[act_idx]
act_func = activation_funcs[act_func_key]

class FinalMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super(FinalMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_func)
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = FinalMLP(X_train.shape[1], final_hidden).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=alpha) if solvers[solver_idx] == 'adam' else optim.SGD(model.parameters(), lr=lr_init, weight_decay=alpha)

X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
y_tr = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1).to(device)

r2_history = []

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    output = model(X_tr)
    loss = criterion(output, y_tr)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_tr)
        ss_res = torch.sum((y_tr - val_pred) ** 2)
        ss_tot = torch.sum((y_tr - torch.mean(y_tr)) ** 2)
        r2 = 1 - ss_res / ss_tot
        r2_history.append(r2.item())

X_main_tensor = torch.tensor(X_main, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    predicted_values = model(X_main_tensor).cpu().numpy().flatten()

main_df['predicted_value'] = predicted_values
output_file = 'regression_results_MLP_GridS.csv'
main_df.to_csv(output_file, index=False)
print(f"\n📁 پیش‌بینی انجام شد و نتیجه در '{output_file}' ذخیره شد.")

plt.figure(figsize=(10,6))
plt.plot(range(1, len(r2_history)+1), r2_history, marker='o', color='darkblue')
plt.xlabel("Epoch")
plt.ylabel("R² Score (Train)")
plt.title("Final Model Training Progress")
plt.grid(True)

params_text = (
    f"Hidden layers: {[n1,n2,n3]}\n"
    f"Activation: {list(activation_funcs.keys())[act_idx]}\n"
    f"Solver: {solvers[solver_idx]}\n"
    f"Learning rate policy: {learning_rates[lr_idx]}\n"
    f"Alpha: {alpha:.5f}\n"
    f"Learning rate init: {lr_init:.5f}\n"
    f"Best R²: {best_score:.5f}"
)
plt.text(
    0.99, 0.01, params_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray')
)

plt.tight_layout()
plt.savefig("Egressor_MLP_GS.png")
plt.show()
