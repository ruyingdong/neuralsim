import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import imageio.v3 as iio
import os

# ─────────────────────── SETTINGS ────────────────────────
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Running on", device)

# Cloth & PINN hyper-parameters
N      = 20             # grid resolution
length = 10.0           # side length
m      = 0.1            # node mass
k      = 1e3            # spring stiffness
c      = 1e-4           # damping
T_END  = 2.0            # max time for sampler
EPOCHS = 200
LR     = 1e-3

# Sampling batch sizes
B_PDE = 2000
B_BC  = 200
B_IC  = 200

# Export paths
CHECKPOINT = "cloth_pinn1.pth"
DATAFILE   = "cloth_pinn_trajectories1.pt"
GIF_NAME   = "cloth_simRU1.gif"

# ───────────────────── ANNEALING CONFIG ──────────────────
lambda_ic   = 10.0
lambda_pin  = 10.0
lambda_pde0 = 1e-10    # 前期 PDE 权重常量
E_hold      = 100     # 持续小权重的 epoch 数
E_total     = EPOCHS  # =200
alpha       = 1.0     # 幂次，可设 alpha=1 做线性，也可设 >1 做非线性
# ─────────────────── INITIAL POSITIONS ────────────────────
coords = [
    [i * length/(N-1), j * length/(N-1), 0.0]
    for j in range(N) for i in range(N)
]

pos0 = torch.tensor(coords, dtype=torch.float32, device=device)  # [N*N,3]

# ─────────────── BUILD SPRING CONNECTIVITY ───────────────
def build_connectivity(N):
    conn = [[] for _ in range(N*N)]
    # structure
    for j in range(N):
        for i in range(N-1):
            idx = j*N+i
            conn[idx].append(idx+1);     conn[idx+1].append(idx)
    for j in range(N-1):
        for i in range(N):
            idx = j*N+i
            conn[idx].append(idx+N);     conn[idx+N].append(idx)
    # shear
    for j in range(N-1):
        for i in range(N-1):
            idx = j*N+i
            conn[idx].append(idx+N+1);   conn[idx+N+1].append(idx)
    for j in range(N-1):
        for i in range(1,N):
            idx = j*N+i
            conn[idx].append(idx+N-1);   conn[idx+N-1].append(idx)
    # bending
    for j in range(N):
        for i in range(N-2):
            idx = j*N+i
            conn[idx].append(idx+2);     conn[idx+2].append(idx)
    for j in range(N-2):
        for i in range(N):
            idx = j*N+i
            conn[idx].append(idx+2*N);   conn[idx+2*N].append(idx)
    return conn

indices = build_connectivity(N)

# ─────────────── INTERNAL SPRING FORCE ──────────────────
def compute_internal_force(p_i, p_j, L0):
    L = torch.norm(p_i-p_j)
    keff = k * (1.1 if L > 1.1*L0 else 1.0)
    return -keff * (L - L0) * (p_i - p_j) / (L + 1e-8)

# ─────────────────────── PINN MODEL ──────────────────────
class ClothPINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [4,64,64,64,3]
        self.net = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1])
            for i in range(len(layers)-1)
        ])
        self.act = nn.Tanh()

    def forward(self, x0, t):
        inp = torch.cat([x0, t], dim=1)
        h = inp
        for L in self.net[:-1]:
            h = self.act(L(h))
        return self.net[-1](h)

model     = ClothPINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ─────────────────────── SAMPLERS ───────────────────────
corner_idx = torch.tensor([0, N*(N-1)], device=device)
def sampler(n):  # PDE collocation
    idx = torch.randint(0, N*N, (n,), device=device)
    return idx, pos0[idx], torch.rand(n,1,device=device)*T_END

def bcs(n):      # fixed corners
    idx = corner_idx[torch.randint(0,2,(n,),device=device)]
    return idx, pos0[idx], torch.rand(n,1,device=device)*T_END, torch.zeros(n,3,device=device)

def ics(n):      # zero disp & vel at t=0
    idx = torch.randint(0, N*N, (n,), device=device)
    t0  = torch.zeros(n,1,device=device, requires_grad=True)
    return idx, pos0[idx], t0, torch.zeros(n,3,device=device), torch.zeros(n,3,device=device)

# ─────────────────────── RESIDUAL ────────────────────────
def residual(idx_batch, x0_batch, t_batch):
    t = t_batch.clone().detach().requires_grad_(True)
    u = model(x0_batch, t)
    u_t  = autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_tt = autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]

    p = x0_batch + u
    f_int = torch.zeros_like(p)
    for n in range(p.shape[0]):
        i   = idx_batch[n].item()
        p_i = p[n];  x0_i = x0_batch[n]
        for j in indices[i]:
            x0_j = pos0[j]
            u_j  = model(x0_j.unsqueeze(0), t[n].unsqueeze(0))[0]
            p_j  = x0_j + u_j
            L0   = torch.norm(x0_i - x0_j)
            f_int[n] += compute_internal_force(p_i, p_j, L0)

    f_vis = -c * u_t
    g     = torch.tensor([0.,0.,-9.8],device=device)
    f_g   = m * g.unsqueeze(0).expand_as(f_int)

    return m*u_tt - (f_int + f_vis + f_g)

# ─────────────────────── TRAINING ────────────────────────
for ep in range(1, EPOCHS+1):
    optimizer.zero_grad()

     # —— 计算当前 PDE 权重 w_pde —— 
    if ep <= E_hold:
        w_pde = lambda_pde0
    else:
        # 在 [E_hold+1, E_total] 这段里，从 lambda_pde0 升到 1
        t = (ep - E_hold) / (E_total - E_hold)      # 0→1
        w_pde = lambda_pde0 + (1.0 - lambda_pde0) * t**alpha

    idx_p, x_p, t_p = sampler(B_PDE)
    loss_pde = (residual(idx_p, x_p, t_p)**2).mean()

    _, x_bc, t_bc, u_bc = bcs(B_BC)
    loss_bc = ((model(x_bc, t_bc) - u_bc)**2).mean()

    _, x_ic, t_ic, u_ic, v_ic = ics(B_IC)
    u_pred = model(x_ic, t_ic)
    u_t_ic = autograd.grad(u_pred, t_ic, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    loss_ic = ((u_pred - u_ic)**2).mean() + ((u_t_ic - v_ic)**2).mean()

    #loss = loss_pde + loss_bc + loss_ic
    loss = lambda_ic * loss_ic \
         + lambda_pin * loss_bc \
         + w_pde     * loss_pde
    loss.backward()
    optimizer.step()

    # if ep % 1 == 0 or ep==1:
    #     print(f"Epoch {ep}/{EPOCHS} → Lpde={loss_pde:.2e} Lbc={loss_bc:.2e} Lic={loss_ic:.2e}")
     # —— 日志 —— 
    if ep == 1 or ep % 10 == 0:
        print(f"Epoch {ep:3d}/{EPOCHS} | "
              f"w_pde={w_pde:.2e} | "
              f"Lpde={loss_pde:.2e}  Lbc={loss_bc:.2e}  Lic={loss_ic:.2e}")

# ─────────────── SAVE MODEL & TRAJECTORIES ───────────────
# Save model weights
torch.save(model.state_dict(), CHECKPOINT)

# Generate full trajectory at NT timesteps
NT   = 100
t_eval = torch.linspace(0, T_END, NT, device=device).unsqueeze(1)
with torch.no_grad():
    pos_pred = []
    for t in t_eval:
        t_batch = t.expand(N*N, 1)         # 变成 [400,1]，与 pos0 对齐
        u = model(pos0, t_batch)           # 前向
        pos_pred.append((pos0 + u).cpu())  # 累加位移
    pos_pred = torch.stack(pos_pred, dim=0)  # [NT, N*N, 3]

# Save time & positions
torch.save({"t": t_eval.cpu(), "pos": pos_pred}, DATAFILE)
print("Saved model to", CHECKPOINT, "and trajectories to", DATAFILE)


# ─────────────────── VISUALIZATION & GIF ───────────────────
# Load back
data = torch.load(DATAFILE)
pos_pred = data["pos"].numpy()   # (NT, N*N,3)

# Build springs indices array
springs = []
for iy in range(N):
    for ix in range(N):
        idx = iy*N + ix
        if ix>0: springs.append((idx, idx-1))
        if iy>0: springs.append((idx, idx-N))
        if ix>0 and iy>0:    springs.append((idx, idx-N-1))
        if ix>0 and iy<N-1:  springs.append((idx, idx+N-1))
        if ix>1:             springs.append((idx, idx-2))
        if iy>1:             springs.append((idx, idx-2*N))
springs = np.array(springs)

# Prepare figure
fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1,1,1))
ax.view_init(elev=30, azim=180)

# Set consistent axes limits from first frame
xyz0 = pos_pred[0]
ax.set_xlim(xyz0[:,0].min(), xyz0[:,0].max())
ax.set_ylim(xyz0[:,1].min(), xyz0[:,1].max())
ax.set_zlim(xyz0[:,2].min(), xyz0[:,2].max())

# Scatter & lines for frame 0
scat = ax.scatter(*xyz0.T, s=8)
lines = Line3DCollection(xyz0[springs], linewidths=0.5)
ax.add_collection(lines)

# Render frames
frames = []
for i in range(pos_pred.shape[0]):
    xyz = pos_pred[i]
    scat._offsets3d = (xyz[:,0], xyz[:,1], xyz[:,2])
    lines.set_segments(xyz[springs])
    ax.set_title(f"t = {data['t'][i].item():.2f} s")
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    h,w = fig.canvas.get_width_height()
    frames.append(buf.reshape(h,w,3))
plt.close(fig)

# Write GIF
iio.imwrite(GIF_NAME, frames, duration=0.05, loop=0)
print("Saved animation to", GIF_NAME)
