import os
import trimesh

for cat in os.listdir('ycb'):
    mesh_path = os.path.join('ycb', cat, 'textured.obj')
    mesh = trimesh.load(mesh_path, process=False)
    z = mesh.bounding_box.bounds[1, 2] - 0.025
    print(f'{cat} {z}')
