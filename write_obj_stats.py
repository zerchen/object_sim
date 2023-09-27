import os
import trimesh

output_path = os.path.join("./", "object_stats.py")
target_category_list = ["ycb"]

with open(output_path, 'w') as f:
    f.writelines("OBJECT_HEIGHT = {\n")
    for target_category in target_category_list:
        for object_name in os.listdir(target_category):
            mesh_path = os.path.join(target_category, object_name, "textured.obj")
            mesh = trimesh.load(mesh_path, process=False)
            if object_name == "019_pitcher_base":
                z = round(mesh.bounding_box.bounds[1, 2] + 0.005, 4)
            else:
                z = round(mesh.bounding_box.bounds[1, 2] - 0.025, 4)
            z_str = str(z)
        
            f.writelines(f"\t\"{object_name}\": {z_str},\n")
    f.writelines("}")
