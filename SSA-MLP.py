# AI_MPM
some good projects for AI-MPM using Optimized MLP
import pandas as pd
import tork
import tork.nn as nn
import tork.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from mealpy.swarm_based.SSA import OriginalSSA
from mealpy.utils.space import IntegerVar, FloatVar
import matplotlib.pyplot as plt
import numpy as np

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

activation_funcs = ['relu', 'tanh', 'logistic']
solvers = ['adam', 'sgd']
learning_rates = ['constant', 'adaptive']

def fitness(solution):
    n1, n2, n3 = int(solution[0]), int(solution[1]), int(solution[2])
    act_idx, solver_idx, lr_idx = int(solution[3]), int(solution[4]), int(solution[5])
    alpha, lr_init = float(solution[6]), float(solution[7])

    hidden_sizes = [n for n in (n1, n2, n3) if n > 0] or [32]
    act_func_key = activation_funcs[act_idx]
    solver = solvers[solver_idx]

    act_layer = nn.ReLU() if act_func_key == 'relu' else nn.Tanh() if act_func_key == 'tanh' else nn.Sigmoid()

    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_sizes):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for h in hidden_sizes:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(act_layer)
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
        for epoch in range(25):
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

    return -np.mean(scores)

problem = {
    "obj_func": fitness,
    "bounds": [
        IntegerVar(0, 256),
        IntegerVar(0, 128),
        IntegerVar(0, 64),
        IntegerVar(0, len(activation_funcs)-1),
        IntegerVar(0, len(solvers)-1),
        IntegerVar(0, len(learning_rates)-1),
        FloatVar(1e-5, 1e-1),
        FloatVar(1e-4, 1e-1),
    ],
    "minmax": "min"
}

epoch =50
pop_size = 100
ssa_model = OriginalSSA(epoch=epoch, pop_size=pop_size)
best_agent = ssa_model.solve(problem)
best_solution = best_agent.solution
best_fitness = best_agent.target.fitness

print("\n✅ بهترین پارامترها:")
print("Hidden layers:", [int(best_solution[0]), int(best_solution[1]), int(best_solution[2])])
print("Activation:", activation_funcs[int(best_solution[3])])
print("Solver:", solvers[int(best_solution[4])])
print("Learning rate policy:", learning_rates[int(best_solution[5])])
print("Alpha:", best_solution[6])
print("Learning rate init:", best_solution[7])
print("Best R² score:", -best_fitness)

# آموزش مدل نهایی با PyTorch
final_hidden = [int(n) for n in best_solution[:3] if int(n) > 0] or [32]
act_func_key = activation_funcs[int(best_solution[3])]
solver = solvers[int(best_solution[4])]
alpha = float(best_solution[6])
lr_init = float(best_solution[7])

act_layer = nn.ReLU() if act_func_key == 'relu' else nn.Tanh() if act_func_key == 'tanh' else nn.Sigmoid()

class FinalMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_layer)
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = FinalMLP(X_train.shape[1], final_hidden).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=alpha) if solver == 'adam' else optim.SGD(model.parameters(), lr=lr_init, weight_decay=alpha)

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

# ساخت لیست max R² تا هر epoch
r2_max_history = []
current_max = -np.inf
for r2 in r2_history:
    current_max = max(current_max, r2)
    r2_max_history.append(current_max)

# پیش‌بینی روی داده اصلی
X_main_tensor = torch.tensor(X_main, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    predicted_values = model(X_main_tensor).cpu().numpy().flatten()

main_df['predicted_value'] = predicted_values
output_file = 'regression_results_SSA4_Torch09.csv'
main_df.to_csv(output_file, index=False)
print(f"\n📁 پیش‌بینی انجام شد و نتیجه در '{output_file}' ذخیره شد.")

# 📊 رسم نمودار فقط max R² تا هر epoch
plt.figure(figsize=(10,6))
plt.plot(range(1, len(r2_max_history)+1), r2_max_history, marker='o', color='darkgreen')
plt.xlabel("Epoch")
plt.ylabel("Max R² Score (Train)")
plt.title("MLP optimized by SSA")
plt.grid(True)

params_text = (
    f"Hidden layers: {[int(best_solution[0]), int(best_solution[1]), int(best_solution[2])]}\n"
    f"Activation: {activation_funcs[int(best_solution[3])]}\n"
    f"Solver: {solvers[int(best_solution[4])]}\n"
    f"Learning rate policy: {learning_rates[int(best_solution[5])]}\n"
    f"Alpha: {best_solution[6]:.5f}\n"
    f"Learning rate init: {best_solution[7]:.5f}\n"
    f"Best R² score: {-best_fitness:.5f}"
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
plt.savefig("FinalModel_MLP_SSA_09.png")
plt.show()
