import argparse
import os
import trimesh
import open3d as o3d


def create_collision_mesh(obj_path, output_path):
    convex_main = '/home/zerui/workspace/code/CoACD/build/main'
    # manifold_main = '/home/zerui/workspace/code/ManifoldPlus/build/manifold'

    os.system(f'{convex_main} -i {obj_path} -o output.obj -pr 20')

    mesh = trimesh.load('output.obj')
    components = mesh.split()
    for idx, part in enumerate(components):
        part = part.simplify_quadric_decimation(60)
        # part.export(f'{output_path}' + f'_{idx+1}.stl')
        part.export(os.path.join(output_path, f'contact{idx}.stl'))


if __name__ == '__main__':
    classes = ['002_master_chef_can','003_cracker_box','004_sugar_box','005_tomato_soup_can','006_mustard_bottle','007_tuna_fish_can','008_pudding_box','009_gelatin_box','010_potted_meat_can','011_banana','019_pitcher_base','021_bleach_cleanser','024_bowl','025_mug','035_power_drill','036_wood_block','037_scissors','040_large_marker','051_large_clamp','052_extra_large_clamp','061_foam_brick']
    for obj_name in classes:
        object_path = f'/home/zerui/workspace/code/object_sim/ycb/{obj_name}'
        os.makedirs(object_path, exist_ok=True)
        object_mesh = trimesh.load(os.path.join(object_path, 'textured.obj'), process=False)
        object_mesh = object_mesh.apply_translation(-object_mesh.centroid)
        object_mesh.export(os.path.join(object_path, 'textured.obj'))

        create_collision_mesh(os.path.join(object_path, 'textured.obj'), object_path)

    # for hoi4d
    # classes = ['Bottle', 'Bowl', 'Kettle', 'Knife', 'Mug', 'ToyCar']
    # hoi4d_model_path = '/home/zerui/workspace/code/dex_sim/datasets/hoi4d/HOI4D_CAD_Model_for_release/rigid'

    # for obj_class in classes:
        # for filename in os.listdir(os.path.join(hoi4d_model_path, obj_class)):
            # instance_id = filename.split('.')[0].zfill(4)

            # visual_path = f'/home/zerui/workspace/code/dex_sim/hand_imitation/env/models/assets/hoi4d_{obj_class}/visual/{instance_id}'
            # collision_path = f'/home/zerui/workspace/code/dex_sim/hand_imitation/env/models/assets/hoi4d_{obj_class}/collision/{instance_id}'
            # os.makedirs(visual_path, exist_ok=True)
            # os.makedirs(collision_path, exist_ok=True)

            # visual_mesh = trimesh.load(os.path.join(hoi4d_model_path, obj_class, filename)).simplify_quadric_decimation(5000)
            # visual_mesh.export(os.path.join(visual_path, 'model_transform_scaled.stl'))
            # # create_collision_mesh(os.path.join(hoi4d_model_path, obj_class, filename), collision_path)

