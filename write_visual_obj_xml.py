import os
import trimesh

target_category = "ycb"
os.makedirs(os.path.join("objects", target_category), exist_ok=True)

for object_name in os.listdir(target_category):
    xml_file_path = os.path.join("objects", target_category, f"{object_name}_visual_target.xml")
    mesh_path = os.path.join(target_category, object_name, "textured.obj")
    mesh = trimesh.load(mesh_path, process=False)
    if object_name == "019_pitcher_base":
        z = round(mesh.bounding_box.bounds[1, 2] + 0.005, 4)
    else:
        z = round(mesh.bounding_box.bounds[1, 2] - 0.025, 4)
    z_str = str(z)

    with open(xml_file_path, 'w') as f:
        f.writelines(f"<mujoco model=\"{object_name}\">\n")
        f.writelines(f"    <include file=\"../../common.xml\"/>\n")
        f.writelines(f"    <asset>\n")
        f.writelines(f"        <mesh name=\"{object_name}\" file=\"../../ycb/{object_name}/textured.obj\"  />\n")
        f.writelines(f"    </asset>\n")

        f.writelines(f"    <worldbody>\n")
        f.writelines(f"        <body name=\"{object_name}_target\" pos=\"0.0 0.0 {z_str}\">\n")
        f.writelines(f"            <geom type=\"mesh\" contype=\"0\" conaffinity=\"0\" mass=\"0\" name=\"{object_name}_visual_target\" class=\"object_visual\" mesh=\"{object_name}\" rgba=\"0 1 0 0.125\" />\n")
        f.writelines(f"        </body>\n")
        f.writelines(f"    </worldbody>\n")
        f.writelines(f"</mujoco>")
