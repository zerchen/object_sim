import os
import trimesh

target_category = "ycb"
os.makedirs(os.path.join("objects", target_category), exist_ok=True)

for object_name in os.listdir(target_category):
    xml_file_path = os.path.join("objects", target_category, f"{object_name}.xml")
    mesh_path = os.path.join(target_category, object_name, "textured.obj")
    mesh = trimesh.load(mesh_path, process=False)
    if object_name == "019_pitcher_base":
        z = round(mesh.bounding_box.bounds[1, 2] + 0.005, 4)
    else:
        z = round(mesh.bounding_box.bounds[1, 2] - 0.025, 4)
    z_str = str(z)

    num_contact_files = 0
    for filename in os.listdir(os.path.join(target_category, object_name)):
        if '.stl' in filename:
            num_contact_files += 1

    with open(xml_file_path, 'w') as f:
        f.writelines(f"<mujoco model=\"{object_name}\">\n")
        f.writelines(f"    <include file=\"../../common.xml\"/>\n")
        f.writelines(f"    <asset>\n")
        f.writelines(f"        <mesh name=\"{object_name}\" file=\"../../ycb/{object_name}/textured.obj\"  />\n")
        for idx in range(num_contact_files):
            f.writelines(f"        <mesh name=\"contact{idx}\" file=\"../../ycb/{object_name}/contact{idx}.stl\"  />\n")
        f.writelines(f"    </asset>\n")

        f.writelines(f"    <worldbody>\n")
        f.writelines(f"        <body name=\"object_entity\" pos=\"0.0 0.0 {z_str}\">\n")
        f.writelines(f"            <joint name=\"OBJTx\" pos=\"0 0 0\" axis=\"1 0 0\" type=\"slide\" class=\"freejoint\" />\n")
        f.writelines(f"            <joint name=\"OBJTy\" pos=\"0 0 0\" axis=\"0 1 0\" type=\"slide\" class=\"freejoint\" />\n")
        f.writelines(f"            <joint name=\"OBJTz\" pos=\"0 0 0\" axis=\"0 0 1\" type=\"slide\" class=\"freejoint\" />\n")
        f.writelines(f"            <joint name=\"OBJRx\" pos=\"0 0 0\" axis=\"1 0 0\" class=\"freejoint\" />\n")
        f.writelines(f"            <joint name=\"OBJRy\" pos=\"0 0 0\" axis=\"0 1 0\" class=\"freejoint\" />\n")
        f.writelines(f"            <joint name=\"OBJRz\" pos=\"0 0 0\" axis=\"0 0 1\" class=\"freejoint\" />\n")
        f.writelines(f"            <geom name=\"entity_visual\" class=\"object_visual\" mesh=\"{object_name}\" />\n")
        for idx in range(num_contact_files):
            f.writelines(f"            <geom name=\"entity_contact{idx}\" class=\"object_contact\" mesh=\"contact{idx}\" />\n")
        f.writelines(f"        </body>\n")
        f.writelines(f"    </worldbody>\n")
        f.writelines(f"</mujoco>")
