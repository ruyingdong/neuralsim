import numpy as np
import matplotlib.pyplot as plt

# re-create pos0
num_x, num_y = 20, 20
DX = DY = 10.0/(num_x-1)
coords = [[i*DX, j*DY, 0.0] for j in range(num_y) for i in range(num_x)]
pos0 = np.array(coords)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')
ax.scatter(pos0[:,0], pos0[:,1], pos0[:,2], s=8)
ax.view_init(elev=30, azim=270)
fig.savefig("initial_scatter.png")
plt.close(fig)
