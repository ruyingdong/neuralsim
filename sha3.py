import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import imageio.v3 as iio

# ───────────────────────── CONFIG ─────────────────────────
DATAFILE = "cloth_pinn_deterministic.pt"   # trajectory output path
GIF_NAME = "cloth_pinn_with_initial_mesh1.gif"   # output GIF filename
num_x, num_y = 20, 20          # grid dimensions
DX = DY = 10.0/(num_x-1)       # node spacing (m)
T_END = 2.0                    # simulation duration (s)
EPOCHS = 10000                  # training epochs
BATCH = 2000                   # PDE time samples per epoch
LR = 8e-4                      # learning rate
NT = 100                       # frames for inference/GIF

# ───────────────────────── DEVICE ─────────────────────────
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ─────────────── GRID & INITIAL POSITIONS ─────────────────
N = num_x * num_y
coords = [[i*DX, j*DY, 0.0] for j in range(num_y) for i in range(num_x)]
pos0 = torch.tensor(coords, dtype=torch.float32, device=device)  # (N,3)

# ──────────── SPRING CONNECTIVITY & PROPERTIES ─────────────
conn = [[] for _ in range(N)]
for j in range(num_y):
    for i in range(num_x):
        idx = j*num_x + i
        if i < num_x-1:
            conn[idx].append(idx+1); conn[idx+1].append(idx)
        if j < num_y-1:
            conn[idx].append(idx+num_x); conn[idx+num_x].append(idx)
for j in range(num_y-1):
    for i in range(num_x-1):
        idx = j*num_x + i
        conn[idx].append(idx+num_x+1); conn[idx+num_x+1].append(idx)
for j in range(num_y-1):
    for i in range(1,num_x):
        idx = j*num_x + i
        conn[idx].append(idx+num_x-1); conn[idx+num_x-1].append(idx)
for j in range(num_y):
    for i in range(num_x-2):
        idx = j*num_x + i
        conn[idx].append(idx+2); conn[idx+2].append(idx)
for j in range(num_y-2):
    for i in range(num_x):
        idx = j*num_x + i
        conn[idx].append(idx+2*num_x); conn[idx+2*num_x].append(idx)

springs = [(i,j) for i, nbrs in enumerate(conn) for j in nbrs if i<j]
springs = torch.tensor(springs, dtype=torch.long, device=device)
rest_len = torch.norm(pos0[springs[:,0]] - pos0[springs[:,1]], dim=1)

# Physical constants
m = 0.1        # mass
k = 1e3        # spring stiffness
c = 1e-4       # damping
g = 0.0        # gravity (z-dir)

# ─────────────────────── MODEL ──────────────────────────
class ClothPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4,64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64,3)
        )
    def forward(self, x0, t):
        inp = torch.cat([x0, t], dim=1)
        return self.net(inp)

model = ClothPINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ───────────────────── FORCE CALCULATION ─────────────────
def spring_forces(pos, vel):
    p1 = pos[:, springs[:,0]]
    p2 = pos[:, springs[:,1]]
    v1 = vel[:, springs[:,0]]
    v2 = vel[:, springs[:,1]]
    diff = p2 - p1
    length = torch.norm(diff, dim=-1, keepdim=True) + 1e-9
    dir_hat = diff / length
    disp = (length.squeeze(-1) - rest_len)
    fs = (k * disp.unsqueeze(-1)) * dir_hat
    rel_vel = v2 - v1
    proj = (rel_vel * dir_hat).sum(-1, keepdim=True)
    fd = -c * proj * dir_hat
    ft = fs + fd
    out = torch.zeros_like(pos)
    for b in range(pos.shape[0]):
        out[b].index_add_(0, springs[:,0], ft[b])
        out[b].index_add_(0, springs[:,1], -ft[b])
    return out

# ─────────────────────── LOSS FUNCTIONS ───────────────────
def residual_loss(t_batch):
    t_batch = t_batch.requires_grad_(True)
    def model_fun(tt):
        B = tt.shape[0]
        x0_exp = pos0.unsqueeze(0).expand(B, -1, -1).reshape(-1,3)
        return model(x0_exp, tt.repeat_interleave(N, dim=0)).reshape(B, N, 3)
    pos, vel = jvp(model_fun, (t_batch,), (torch.ones_like(t_batch),), create_graph=True)
    _, acc = jvp(lambda tt: jvp(model_fun, (tt,), (torch.ones_like(tt),), create_graph=True)[1],
               (t_batch,), (torch.ones_like(t_batch),), create_graph=True)
    force = spring_forces(pos, vel)
    force[:,:,2] -= m * g
    return ((acc - force/m)**2).mean()

down_idx = torch.arange(num_x, device=device)
def bc_loss(t_batch):
    B = t_batch.shape[0]
    x0_edge = pos0[down_idx]
    u = model(x0_edge.repeat(B,1), t_batch.repeat_interleave(len(down_idx), dim=0))
    return ((u.reshape(B, len(down_idx), 3) - x0_edge.unsqueeze(0))**2).mean()

# ──────────────────── INITIAL CONDITION LOSS ─────────────────
def ic_loss():
    t0 = torch.zeros(N,1,device=device, requires_grad=True)
    # pos0 and vel0
    def model_fun(tt):
        B = tt.shape[0]
        x0_exp = pos0.unsqueeze(0).expand(B, -1, -1).reshape(-1,3)
        return model(x0_exp, tt.repeat_interleave(N, dim=0)).reshape(B, N, 3)
    pos0_pred, vel0_pred = jvp(model_fun, (t0,), (torch.ones_like(t0),), create_graph=True)
    # acceleration at t=0
    _, acc0_pred = jvp(lambda tt: jvp(model_fun, (tt,), (torch.ones_like(tt),), create_graph=True)[1],
                       (t0,), (torch.ones_like(t0),), create_graph=True)
    loss_pos = ((pos0_pred - pos0)**2).mean()
    loss_vel = (vel0_pred**2).mean()
    loss_acc = (acc0_pred**2).mean()
    return loss_pos + loss_vel + loss_acc

# ─────────────────────── TRAINING ────────────────────────
for ep in range(1, EPOCHS+1):
    optimizer.zero_grad()
    tb = torch.rand(BATCH,1,device=device) * T_END
    Lr = residual_loss(tb)
    Lb = bc_loss(tb)
    Li = ic_loss()
    loss = Li
    loss.backward(); optimizer.step()
    if ep == 1 or ep % 50 == 0:
        print(f"Epoch {ep}/{EPOCHS} | Lr={Lr:.2e} Lb={Lb:.2e} Li={Li:.2e}")

# ────────────────── INFERENCE & SAVE TRAJECTORIES ─────────────────
with torch.no_grad():
    t_eval = torch.linspace(0, T_END, NT, device=device).unsqueeze(1)
    x0_exp = pos0.unsqueeze(0).expand(NT, -1, -1).reshape(-1,3)
    pos_pred = model(x0_exp, t_eval.repeat_interleave(N, dim=0)).reshape(NT, N, 3).cpu().numpy()
    torch.save({"t": t_eval.cpu(), "pos": torch.tensor(pos_pred)}, DATAFILE)

# ─────────────────── VISUALIZATION & GIF ─────────────────
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect((1,1,1))
ax.view_init(elev=30, azim=270)
# draw initial mesh (dashed)
pos0_np = pos0.cpu().numpy()
ax.add_collection(Line3DCollection(pos0_np[springs.cpu().numpy()],
                                   linestyles='dashed', colors='gray', alpha=0.5))
ax.scatter(*pos0_np.T, s=4, color='gray', alpha=0.5)
# setup predicted mesh
xyz0 = pos_pred[0]
ax.set_xlim(xyz0[:,0].min(), xyz0[:,0].max()); ax.set_ylim(xyz0[:,1].min(), xyz0[:,1].max()); ax.set_zlim(xyz0[:,2].min(), xyz0[:,2].max())
scat_pred = ax.scatter(*xyz0.T, s=8)
lines_pred = Line3DCollection(xyz0[springs.cpu().numpy()], linewidths=0.8)
ax.add_collection(lines_pred)
frames = []
for i in range(NT):
    xyz = pos_pred[i]
    scat_pred._offsets3d = (xyz[:,0], xyz[:,1], xyz[:,2])
    lines_pred.set_segments(xyz[springs.cpu().numpy()])
    ax.set_title(f"t = {i/(NT-1)*T_END:.2f} s")
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    h, w = fig.canvas.get_width_height()
    frames.append(buf.reshape(h, w, 3))
plt.close(fig)
iio.imwrite(GIF_NAME, frames, duration=0.05, loop=0)
print(f"Saved GIF with initial mesh overlay to {GIF_NAME}")
