
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import imageio.v3 as iio

# ───────────────────────── CONFIG ─────────────────────────
DATAFILE = "cloth_pinn_deterministic.pt"   # trajectory output path
GIF_NAME = "cloth_pinn_deterministic.gif"   # output GIF filename
num_x, num_y = 20, 20          # grid dimensions
DX = DY = 10.0/(num_x-1)        # node spacing (m)
T_END = 2.0                    # simulation duration (s)
EPOCHS = 2000                   # training epochs
BATCH = 2000                   # PDE time samples per epoch
LR = 8e-4                      # learning rate
NT = 100                       # frames for inference/GIF

# ───────────────────────── DEVICE ─────────────────────────
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ─────────────── GRID & INITIAL POSITIONS ─────────────────
N = num_x * num_y
coords = []
for j in range(num_y):
    for i in range(num_x):
        coords.append([i*DX, j*DY, 0.0])
pos0 = torch.tensor(coords, dtype=torch.float32, device=device)  # (N,3)
print(pos0)

# ──────────── SPRING CONNECTIVITY & PROPERTIES ─────────────
conn = [[] for _ in range(N)]
# structure springs (adjacent)
for j in range(num_y):
    for i in range(num_x):
        idx = j*num_x + i
        if i < num_x-1:
            conn[idx].append(idx+1); conn[idx+1].append(idx)
        if j < num_y-1:
            conn[idx].append(idx+num_x); conn[idx+num_x].append(idx)
# shear springs
for j in range(num_y-1):
    for i in range(num_x-1):
        idx = j*num_x + i
        conn[idx].append(idx+num_x+1); conn[idx+num_x+1].append(idx)
for j in range(num_y-1):
    for i in range(1, num_x):
        idx = j*num_x + i
        conn[idx].append(idx+num_x-1); conn[idx+num_x-1].append(idx)
# bending springs (two away)
for j in range(num_y):
    for i in range(num_x-2):
        idx = j*num_x + i
        conn[idx].append(idx+2); conn[idx+2].append(idx)
for j in range(num_y-2):
    for i in range(num_x):
        idx = j*num_x + i
        conn[idx].append(idx+2*num_x); conn[idx+2*num_x].append(idx)

springs = []
for i, nbrs in enumerate(conn):
    for j in nbrs:
        if i < j:
            springs.append((i, j))
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
# PDE residual via JVP

def residual_loss(t_batch):
    t_batch = t_batch.requires_grad_(True)
    def model_fun(tt):
        B = tt.shape[0]
        x0 = pos0.unsqueeze(0).expand(B, -1, -1).reshape(-1,3)
        u = model(x0, tt.repeat_interleave(N, dim=0))
        return u.reshape(B, N, 3)
    pos, vel = jvp(model_fun, (t_batch,), (torch.ones_like(t_batch),), create_graph=True)
    _, acc = jvp(lambda tt: jvp(model_fun, (tt,), (torch.ones_like(tt),), create_graph=True)[1],
                 (t_batch,), (torch.ones_like(t_batch),), create_graph=True)
    force = spring_forces(pos, vel)
    force[:,:,2] -= m * g
    return ((acc - force/m)**2).mean()

# boundary: pin bottom edge (j=0)
down_idx = torch.arange(0, num_x, device=device)
def bc_loss(t_batch):
    B = t_batch.shape[0]
    x0 = pos0[down_idx]
    u = model(x0.repeat(B,1), t_batch.repeat_interleave(len(down_idx), dim=0))
    u = u.reshape(B, len(down_idx), 3)
    return ((u - x0.unsqueeze(0))**2).mean()

# initial condition at t=0, expanded to full grid
def ic_loss():
    t0 = torch.zeros(N,1,device=device, requires_grad=True)
    pos0_pred, vel0_pred = jvp(lambda tt: model(pos0, tt),
                                 (t0,), (torch.ones_like(t0),), create_graph=True)
    return ((pos0_pred - pos0)**2).mean() + (vel0_pred**2).mean()

# ─────────────────────── TRAINING ───────────────────────
for ep in range(1, EPOCHS+1):
    optimizer.zero_grad()
    tb = torch.rand(BATCH,1,device=device) * T_END
    Lr = residual_loss(tb)
    Lb = bc_loss(tb)
    Li = ic_loss()
    loss = Lr + 1000000*Lb + 100000*Li
    loss.backward(); optimizer.step()
    if ep==1 or ep%50==0:
        print(f"Epoch {ep}/{EPOCHS} | Lr={Lr:.2e} Lb={Lb:.2e} Li={Li:.2e}")

# ────────────────── INFERENCE & SAVE TRAJECTORIES ─────────────────
with torch.no_grad():
    t_eval = torch.linspace(0, T_END, NT, device=device).unsqueeze(1)
    B = NT
    x0 = pos0.unsqueeze(0).expand(B, -1, -1).reshape(-1,3)
    u = model(x0, t_eval.repeat_interleave(N, dim=0))
    pos_pred = u.reshape(B, N, 3).cpu().numpy()
    torch.save({"t": t_eval.cpu(), "pos": torch.tensor(pos_pred)}, DATAFILE)

# ─────────────────── VISUALIZATION & GIF ───────────────────
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect((1,1,1))
ax.view_init(elev=30, azim=270)
xyz0 = pos_pred[0]
ax.set_xlim(xyz0[:,0].min(), xyz0[:,0].max())
ax.set_ylim(xyz0[:,1].min(), xyz0[:,1].max())
ax.set_zlim(xyz0[:,2].min(), xyz0[:,2].max())
scat = ax.scatter(*xyz0.T, s=8)
lines = Line3DCollection(xyz0[springs.cpu().numpy()], linewidths=0.5)
ax.add_collection(lines)
frames = []
for i in range(NT):
    xyz = pos_pred[i]
    scat._offsets3d = (xyz[:,0], xyz[:,1], xyz[:,2])
    segs = pos_pred[i][springs.cpu().numpy()]
    lines.set_segments(segs)
    ax.set_title(f"t = {i/(NT-1)*T_END:.2f} s")
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    h,w = fig.canvas.get_width_height()
    frames.append(buf.reshape(h,w,3))
plt.close(fig)
iio.imwrite(GIF_NAME, frames, duration=0.05, loop=0)
print("Saved GIF to", GIF_NAME)

