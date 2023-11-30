from dm_control import mjcf
from dm_control.rl import control
from dm_control.mjcf import debugging
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


def main(traj_name=None):
    if traj_name is None:
        traj_names = os.listdir('trajectories/ycb')
        traj_names = [name.split('.')[0] for name in traj_names]
        random.shuffle(traj_names)
    else:
        traj_names = [traj_name]

    for traj_name in traj_names:
        index = '-'.join(traj_name.split('-')[2:-1]) + '/' + traj_name.split('-')[-1]

        object_category = traj_name.split('-')[0]
        object_name = traj_name.split('-')[1]

        traj_path = os.path.join(f'trajectories/{object_category}/{traj_name}.npz')
        traj_file = np.load(traj_path, allow_pickle=True)
        traj_file =  {k:v for k, v in traj_file.items()}
        traj_file['s_0'] = traj_file['s_0'][()]

        init_object_translation = np.array(traj_file['s_0']['pregrasp']['object_translation'])
        init_object_orientation = np.array(traj_file['s_0']['pregrasp']['object_orientation'])
        retarget_pose = np.array(traj_file['s_0']['pregrasp']['robot_position'])

        object_translation = np.array(traj_file['object_translation'])
        object_orientation = np.array(traj_file['object_orientation'])
        hand_joint = np.array(traj_file['hand_joint'])
        pregrasp_joint = np.array(traj_file['s_0']['pregrasp']['position'])

        env = TableEnv()
        robot_model = get_robot('adroit')(limp=False)
        robot_mesh_list = robot_model.mjcf_model.find_all('geom')
        robot_geom_names = [geom.get_attributes()['name'] for geom in robot_mesh_list]
        robot_geom_names = [f'adroit/{name}' for name in robot_geom_names if 'C' in name]
        env.attach(robot_model)

        object_model = object_generator(f"objects/{object_category}/{object_name}.xml")(pos=init_object_translation, quat=init_object_orientation)
        object_mesh_list = object_model.mjcf_model.find_all('geom')
        object_geom_names = [geom.get_attributes()['name'] for geom in object_mesh_list]
        object_geom_names = [f'{object_name}/{name}' for name in object_geom_names if 'contact' in name]

        object_model.mjcf_model.worldbody.add('body', name=f'hand_palm', pos=pregrasp_joint[0])
        object_model.mjcf_model.worldbody.body[f'hand_palm'].add('geom', type='sphere', contype='0', conaffinity='0', mass='0', name=f'hand_palm_visual', size="0.01", rgba=np.array([1, 0, 0, 1]))
        object_model.mjcf_model.worldbody.add('body', name=f'hand_thumb', pos=pregrasp_joint[4])
        object_model.mjcf_model.worldbody.body[f'hand_thumb'].add('geom', type='sphere', contype='0', conaffinity='0', mass='0', name=f'hand_thumb_visual', size="0.01", rgba=np.array([1, 0, 0, 1]))
        object_model.mjcf_model.worldbody.add('body', name=f'hand_index', pos=pregrasp_joint[8])
        object_model.mjcf_model.worldbody.body[f'hand_index'].add('geom', type='sphere', contype='0', conaffinity='0', mass='0', name=f'hand_index_visual', size="0.01", rgba=np.array([1, 0, 0, 1]))
        object_model.mjcf_model.worldbody.add('body', name=f'hand_middle', pos=pregrasp_joint[12])
        object_model.mjcf_model.worldbody.body[f'hand_middle'].add('geom', type='sphere', contype='0', conaffinity='0', mass='0', name=f'hand_middle_visual', size="0.01", rgba=np.array([1, 0, 0, 1]))
        object_model.mjcf_model.worldbody.add('body', name=f'hand_ring', pos=pregrasp_joint[16])
        object_model.mjcf_model.worldbody.body[f'hand_ring'].add('geom', type='sphere', contype='0', conaffinity='0', mass='0', name=f'hand_ring_visual', size="0.01", rgba=np.array([1, 0, 0, 1]))
        object_model.mjcf_model.worldbody.add('body', name=f'hand_little', pos=pregrasp_joint[20])
        object_model.mjcf_model.worldbody.body[f'hand_little'].add('geom', type='sphere', contype='0', conaffinity='0', mass='0', name=f'hand_little_visual', size="0.01", rgba=np.array([1, 0, 0, 1]))

        env.attach(object_model)
        physics = physics_from_mjcf(env)
        ori_obs_pos = physics.named.data.xpos[45].copy()

        model = physics.model.ptr
        data = physics.data.ptr

        is_contact = False
        pregrasp_step = 9
        for idx in range(retarget_pose.shape[0]):
            if idx >= 9:
                physics.reset()
                physics.data.qpos[:30] = retarget_pose[idx]
                physics.data.qvel[:30] = np.zeros(30)

                physics.forward()
                is_contact = check_contacts(physics, robot_geom_names, object_geom_names)
                if is_contact:
                    break
                pregrasp_step += 1

        pregrasp_step -= 1

        object_translation[:, :2] -= init_object_translation[:2]
        hand_joint[:, :, :2] -= init_object_translation[:2]
        pregrasp_joint[:, :2] -= init_object_translation[:2]
        retarget_pose[:, 0] += init_object_translation[0]
        retarget_pose[:, 2] -= init_object_translation[1]
        init_object_translation[:2] -= init_object_translation[:2]

        new_init_pos = np.random.uniform(low=-0.15, high=0.15, size=2)
        object_translation[:, :2] += new_init_pos
        hand_joint[:, :, :2] += new_init_pos
        pregrasp_joint[:, :2] += new_init_pos
        init_object_translation[:2] += new_init_pos
        retarget_pose[:, 0] -= new_init_pos[0]
        retarget_pose[:, 2] += new_init_pos[1]

        physics.reset()
        physics.model.body_pos[45] = init_object_translation
        physics.model.body_pos[46:] = pregrasp_joint[[0, 4, 8, 12, 16, 20]]
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
    Fire(main)
