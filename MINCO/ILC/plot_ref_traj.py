#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

base = os.path.dirname(__file__)
ref_dir = os.path.join(base, 'ref_traj')
npz_path = os.path.join(ref_dir, 'optimized_traj.npz')

if not os.path.isfile(npz_path):
    raise RuntimeError(f"optimized_traj.npz not found at {npz_path}")

print(f"Loading optimized trajectory from: {npz_path}")
data = np.load(npz_path)
print('keys:', list(data.keys()))

traj = data.get('traj')
q_opt = data.get('q_opt')
T_opt = data.get('T_opt')
p0 = data.get('p0')
pf = data.get('pf')

print('traj shape:', None if traj is None else traj.shape)
print('q_opt shape:', None if q_opt is None else q_opt.shape)
print('T_opt shape:', None if T_opt is None else T_opt.shape)
print('p0:', p0)
print('pf:', pf)

fig, ax = plt.subplots(figsize=(8,8))
if traj is None:
    raise RuntimeError('No traj field in npz')

# try to load corridor/map for background if available
corridor_npz = None
possible = [
    os.path.join(base, '..', '..', 'MINCO', 'gen_map_tube', 'tube_corridor.npz'),
    os.path.join(base, '..', '..', 'MINCO', 'gen_map_tube', 'gen_ref_corridor.npz'),
]
for p in possible:
    p = os.path.abspath(p)
    if os.path.isfile(p):
        corridor_npz = p
        break

if corridor_npz is not None:
    try:
        cd = np.load(corridor_npz)
        if 'grid' in cd:
            grid = np.asarray(cd['grid'])
            res = float(cd['resolution']) if 'resolution' in cd else 1.0
            origin = np.asarray(cd['origin']) if 'origin' in cd else np.array([0.0,0.0])
            if np.asarray(origin).ndim == 0 or np.asarray(origin).size==1:
                origin = np.array([float(origin), 0.0])
            nx, ny = grid.shape
            xmin, xmax = origin[0], origin[0] + nx*res
            ymin, ymax = origin[1], origin[1] + ny*res
            ax.imshow(grid.T, origin='lower', extent=(xmin, xmax, ymin, ymax), cmap='gray_r', alpha=0.5)
        if 'oc_list' in cd and 'r_list' in cd:
            oc_list = np.asarray(cd['oc_list'])
            r_list = np.asarray(cd['r_list'])
            theta = np.linspace(0, 2*np.pi, 200)
            for c,r in zip(oc_list, r_list):
                xc = c[0] + r*np.cos(theta)
                yc = c[1] + r*np.sin(theta)
                ax.fill(xc, yc, color='orange', alpha=0.3)
                ax.plot(xc, yc, color='orange')
    except Exception as e:
        print('Failed to load corridor/map for background:', e)

ax.plot(traj[:,0], traj[:,1], color='blue', linewidth=2, label='optimized_traj')
ax.scatter(traj[0,0], traj[0,1], color='green', s=80, label='start')
ax.scatter(traj[-1,0], traj[-1,1], color='red', s=80, label='goal')
ax.set_aspect('equal')
ax.legend()
plt.tight_layout()

out_png = os.path.join(ref_dir, 'optimized_traj_plot.png')
fig.savefig(out_png, dpi=150)
print('Saved plot to', out_png)

# also show if possible
try:
    plt.show()
except Exception:
    pass

print('Done')
