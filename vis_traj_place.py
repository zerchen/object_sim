from dm_control import mjcf
from dm_control.rl import control
from dm_control.mjcf import debugging
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from physics import physics_from_mjcf
from robots import get_robot
from fire import Fire
from base import MjModel
import numpy as np
import pickle
import random
import mujoco_viewer
import mujoco
import os
import transforms3d
import xml.etree.ElementTree as ET


def create_big_mug(scale=1.8, color=np.array([0.3, 0.3, 0.3, 1])):
    object_name = "big_mug"
    with open("big_mug.xml", 'w') as f:
        f.writelines(f"<mujoco model=\"{object_name}\">\n")
        f.writelines(f"    <include file=\"../../common.xml\"/>\n")
        f.writelines(f"    <worldbody>\n")
        f.writelines(f"        <body name=\"object_entity\" pos=\"0.0 0.0 0.06\">\n")
        f.writelines(f"            <joint name=\"OBJTx\" pos=\"0 0 0\" axis=\"1 0 0\" type=\"slide\" class=\"freejoint\" />\n")
        f.writelines(f"            <joint name=\"OBJTy\" pos=\"0 0 0\" axis=\"0 1 0\" type=\"slide\" class=\"freejoint\" />\n")
        f.writelines(f"            <joint name=\"OBJTz\" pos=\"0 0 0\" axis=\"0 0 1\" type=\"slide\" class=\"freejoint\" />\n")
        f.writelines(f"            <joint name=\"OBJRx\" pos=\"0 0 0\" axis=\"1 0 0\" class=\"freejoint\" />\n")
        f.writelines(f"            <joint name=\"OBJRy\" pos=\"0 0 0\" axis=\"0 1 0\" class=\"freejoint\" />\n")
        f.writelines(f"            <joint name=\"OBJRz\" pos=\"0 0 0\" axis=\"0 0 1\" class=\"freejoint\" />\n")
        for idx in range(12):
            size = np.array([0.028, 0.005, 0.08]) / 2 * scale
            theta = np.deg2rad(idx * 30)
            pos = np.array([0.045 * np.sin(theta) - 0.006, -0.042 * np.cos(theta), 0.0015]) * scale
            quat = transforms3d.quaternions.axangle2quat(np.array([0, 0, 1]), theta)
            f.writelines(f"            <geom type=\"box\" name=\"entity_visual_{idx}\" class=\"object_visual\" pos=\"{pos[0]} {pos[1]} {pos[2]}\" size=\"{size[0]} {size[1]} {size[2]}\" quat=\"{quat[0]} {quat[1]} {quat[2]} {quat[3]}\" rgba=\"0.3 0.3 0.3 1\" />\n")
            f.writelines(f"            <geom type=\"box\" name=\"entity_contact_{idx}\" class=\"object_contact\" pos=\"{pos[0]} {pos[1]} {pos[2]}\" size=\"{size[0]} {size[1]} {size[2]}\" quat=\"{quat[0]} {quat[1]} {quat[2]} {quat[3]}\" mass=\"0.5\" />\n")
        size_bottom = np.array([0.0425, 0.005]) * scale
        pos_bottom = np.array([0, 0, -0.033]) * scale
        f.writelines(f"            <geom type=\"cylinder\" name=\"entity_visual_12\" class=\"object_visual\" pos=\"{pos_bottom[0]} {pos_bottom[1]} {pos_bottom[2]}\" size=\"{size_bottom[0]} {size_bottom[1]}\" rgba=\"0.3 0.3 0.3 1\" />\n")
        f.writelines(f"            <geom type=\"cylinder\" name=\"entity_contact_12\" class=\"object_contact\" pos=\"{pos_bottom[0]} {pos_bottom[1]} {pos_bottom[2]}\" size=\"{size_bottom[0]} {size_bottom[1]}\" mass=\"0.5\" />\n")
        f.writelines(f"        </body>\n")
        f.writelines(f"    </worldbody>\n")
        f.writelines(f"</mujoco>")


def check_contacts(physics, geoms_1, geoms_2):
    for contact in physics.data.contact[:physics.data.ncon]:
        # check contact geom in geoms
        c1_in_g1 = physics.model.geom(contact.geom1).name in geoms_1
        c2_in_g2 = physics.model.geom(contact.geom2).name in geoms_2

        # check contact geom in geoms (flipped)
        c2_in_g1 = physics.model.geom(contact.geom2).name in geoms_1
        c1_in_g2 = physics.model.geom(contact.geom1).name in geoms_2
        if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
            return True
    return False


class BaseEnv(MjModel):
    def __init__(self, xml_path):
        mjcf_model = mjcf.from_path(xml_path)
        super().__init__(mjcf_model)


class TableEnv(BaseEnv):
    def __init__(self):
        super().__init__('table.xml')


class ObjectModel(MjModel):
    def __init__(self, base_pos, base_quat, mjcf_model):
        if isinstance(mjcf_model, str):
            mjcf_model = mjcf.from_path(mjcf_model)

        if base_pos is not None:
            self._base_pos = np.array(base_pos).copy()
            mjcf_model.worldbody.all_children()[0].pos = self._base_pos

        if base_quat is not None:
            self._base_quat = np.array(base_quat).copy()
            mjcf_model.worldbody.all_children()[0].quat = self._base_quat

        super().__init__(mjcf_model)

    @property
    def start_pos(self):
        return self._base_pos.copy().astype(np.float32)

    @property
    def start_ori(self):
        return self._base_quat.copy().astype(np.float32)

def object_generator(path):
    class __XMLObj__(ObjectModel):
        def __init__(self, pos=None, quat=None):
            xml_path = path
            super().__init__(pos, quat, xml_path)
    return __XMLObj__


def main(traj_name="ycb-011_banana-20200820-subject-03-20200820_141935"):
    if traj_name is None:
        traj_names = os.listdir('trajectories/ycb')
        traj_names = [name.split('.')[0] for name in traj_names]
        random.shuffle(traj_names)
    else:
        traj_names = traj_name.split(',')

    for traj_name in traj_names:
        index = '-'.join(traj_name.split('-')[2:-1]) + '/' + traj_name.split('-')[-1]

        object_category = traj_name.split('-')[0]
        object_name = traj_name.split('-')[1]

        traj_path = os.path.join(f'trajectories/{object_category}/{traj_name}.npz')
        traj_file = np.load(traj_path, allow_pickle=True)
        traj_file =  {k:v for k, v in traj_file.items()}

        retarget_pose = np.array(traj_file['robot_qpos'])
        retarget_joint = np.array(traj_file['robot_jpos'])
        pregrasp_step = traj_file['pregrasp_step']
        object_translation = np.array(traj_file['object_translation'])
        object_orientation = np.array(traj_file['object_orientation'])
        init_object_translation = np.array(traj_file['object_translation'][0]).copy()
        init_object_orientation = np.array(traj_file['object_orientation'][0]).copy()

        object_translation[:, :2] -= init_object_translation[:2]
        retarget_pose[:, 0] += init_object_translation[0]
        retarget_pose[:, 2] -= init_object_translation[1]
        retarget_joint[:, :, :2] -= init_object_translation[:2]
        init_object_translation[:2] -= init_object_translation[:2]

        new_init_pos = np.array([0.22, 0]) + np.array([np.random.uniform(low=-0.05, high=0.05), np.random.uniform(low=0.0, high=0.1)])
        object_translation[:, :2] += new_init_pos
        init_object_translation[:2] += new_init_pos
        retarget_pose[:, 0] -= new_init_pos[0]
        retarget_pose[:, 2] += new_init_pos[1]
        retarget_joint[:, :, :2] += new_init_pos

        env = TableEnv()
        robot_model = get_robot('adroit')(limp=False)
        robot_mesh_list = robot_model.mjcf_model.find_all('geom')
        robot_geom_names = [geom.get_attributes()['name'] for geom in robot_mesh_list]
        robot_geom_names = [f'adroit/{name}' for name in robot_geom_names if 'C' in name]
        env.attach(robot_model)

        mug_model = object_generator(f"objects/common/big_mug.xml")(pos=(0.01, 0.00, 0.06), quat=(1, 0, 0, 0))
        env.attach(mug_model)

        object_model = object_generator(f"objects/{object_category}/{object_name}.xml")(pos=init_object_translation, quat=init_object_orientation)
        object_mesh_list = object_model.mjcf_model.find_all('geom')
        object_geom_names = [geom.get_attributes()['name'] for geom in object_mesh_list]
        object_geom_names = [f'{object_name}/{name}' for name in object_geom_names if 'contact' in name]

        final_goal = np.array([0.0, -0.02, 0.24], dtype=np.float32)
        min_idx = np.where(object_translation[:, 2] - object_translation[0, 2] > 0.24)[0][0]
        object_translation = object_translation[:min_idx + 1]
        object_orientation = object_orientation[:min_idx + 1]
        final_goal[2] = object_translation[-1, 2]
        final_rot = (R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0])) * R.from_quat(object_orientation[-1][[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]

        for idx, _ in enumerate(object_translation[1:]):
            if idx % 1 == 0:
                object_model.mjcf_model.worldbody.add('body', name=f'object_marker_{idx}', pos=object_translation[idx], quat=object_orientation[idx])
                object_model.mjcf_model.worldbody.body[f'object_marker_{idx}'].add('geom', contype='0', conaffinity='0', mass='0', name=f'target_visual_{idx}', mesh=object_model.mjcf_model.worldbody.body['object_entity'].geom['entity_visual'].mesh, rgba=np.array([0.996, 0.878, 0.824, 0.125]))
                object_model.mjcf_model.worldbody.body[f'object_marker_{idx}'].geom[f'target_visual_{idx}'].type = "mesh"

        dist_vec = final_goal - object_translation[-1]
        unit_dist_vec = dist_vec / np.linalg.norm(dist_vec)
        relocate_step = 0.02
        num_step = int(np.linalg.norm(dist_vec) // relocate_step)
        init_rot = R.from_quat(object_orientation[-1][[1, 2, 3, 0]])
        rotation_step = -np.pi / 2 / num_step

        syn_object_translation = []
        syn_object_orientation = []
        syn_retarget_joint = []
        for idx in range(num_step):
            step_size = (idx + 1) * relocate_step
            syn_object_translation.append(object_translation[-1] + step_size * unit_dist_vec)

            rot_size = (idx + 1) * rotation_step
            syn_object_orientation.append((R.from_rotvec(rot_size * np.array([1, 0, 0])) * init_rot).as_quat()[[3, 0, 1, 2]])

            cur_joint = retarget_joint[min_idx].copy()
            cur_joint += (syn_object_translation[-1] - object_translation[-1])
            cur_joint -= syn_object_translation[-1]
            rotmat = R.from_rotvec(rot_size * np.array([1, 0, 0])).as_matrix()
            cur_joint = (rotmat @ cur_joint.transpose(1, 0)).transpose(1, 0)
            cur_joint += syn_object_translation[-1]
            syn_retarget_joint.append(cur_joint)

        unit_dist_vec = np.array([0, 0, -0.02])
        for idx in range(8):
            num_step += 1
            syn_object_translation.append(syn_object_translation[-1] + unit_dist_vec)
            syn_object_orientation.append(syn_object_orientation[-1])
            syn_retarget_joint.append(syn_retarget_joint[-1] + unit_dist_vec[None, :])

        total_steps = len(object_translation) + len(syn_object_translation)
        print(f'total_steps: {total_steps}')

        for idx in range(num_step):
            object_model.mjcf_model.worldbody.add('body', name=f'object_marker_syn_{idx}', pos=syn_object_translation[idx], quat=syn_object_orientation[idx])
            object_model.mjcf_model.worldbody.body[f'object_marker_syn_{idx}'].add('geom', contype='0', conaffinity='0', mass='0', name=f'target_visual_syn_{idx}', mesh=object_model.mjcf_model.worldbody.body['object_entity'].geom['entity_visual'].mesh, rgba=np.array([1, 0, 0, 0.125]))
            object_model.mjcf_model.worldbody.body[f'object_marker_syn_{idx}'].geom[f'target_visual_syn_{idx}'].type = "mesh"

        object_model.mjcf_model.worldbody.add('body', name=f'object_marker', pos=syn_object_translation[-1], quat=syn_object_orientation[-1])
        object_model.mjcf_model.worldbody.body[f'object_marker'].add('geom', contype='0', conaffinity='0', mass='0', name='target_visual', mesh=object_model.mjcf_model.worldbody.body['object_entity'].geom['entity_visual'].mesh, rgba=np.array([0, 1, 0, 0.125]))
        object_model.mjcf_model.worldbody.body[f'object_marker'].geom['target_visual'].type = "mesh"

        object_model.mjcf_model.worldbody.add('body', name=f'hand_palm', pos=syn_retarget_joint[-1][0])
        object_model.mjcf_model.worldbody.body[f'hand_palm'].add('geom', type='sphere', contype='0', conaffinity='0', mass='0', name=f'hand_palm_visual', size="0.01", rgba=np.array([1, 0, 0, 1]))
        object_model.mjcf_model.worldbody.add('body', name=f'hand_thumb', pos=syn_retarget_joint[-1][1])
        object_model.mjcf_model.worldbody.body[f'hand_thumb'].add('geom', type='sphere', contype='0', conaffinity='0', mass='0', name=f'hand_thumb_visual', size="0.01", rgba=np.array([1, 0, 0, 1]))
        object_model.mjcf_model.worldbody.add('body', name=f'hand_index', pos=syn_retarget_joint[-1][2])
        object_model.mjcf_model.worldbody.body[f'hand_index'].add('geom', type='sphere', contype='0', conaffinity='0', mass='0', name=f'hand_index_visual', size="0.01", rgba=np.array([1, 0, 0, 1]))
        object_model.mjcf_model.worldbody.add('body', name=f'hand_middle', pos=syn_retarget_joint[-1][3])
        object_model.mjcf_model.worldbody.body[f'hand_middle'].add('geom', type='sphere', contype='0', conaffinity='0', mass='0', name=f'hand_middle_visual', size="0.01", rgba=np.array([1, 0, 0, 1]))
        object_model.mjcf_model.worldbody.add('body', name=f'hand_ring', pos=syn_retarget_joint[-1][4])
        object_model.mjcf_model.worldbody.body[f'hand_ring'].add('geom', type='sphere', contype='0', conaffinity='0', mass='0', name=f'hand_ring_visual', size="0.01", rgba=np.array([1, 0, 0, 1]))
        object_model.mjcf_model.worldbody.add('body', name=f'hand_little', pos=syn_retarget_joint[-1][5])
        object_model.mjcf_model.worldbody.body[f'hand_little'].add('geom', type='sphere', contype='0', conaffinity='0', mass='0', name=f'hand_little_visual', size="0.01", rgba=np.array([1, 0, 0, 1]))

        env.attach(object_model)
        physics = physics_from_mjcf(env)

        model = physics.model.ptr
        data = physics.data.ptr

        physics.reset()
        viewer = mujoco_viewer.MujocoViewer(model, data)
        # simulate and render
        for _ in range(1000):
            if viewer.is_alive:
                physics.data.qpos[:30] = retarget_pose[pregrasp_step]
                physics.data.qvel[:30] = np.zeros(30)
                physics.step()
                viewer.render()
            else:
                break

        # close
        viewer.close()

if __name__ == '__main__':
    # create_big_mug()
    Fire(main)
