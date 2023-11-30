from dm_control import mjcf
from dm_control.rl import control
from dm_control.mjcf import debugging
from pyquaternion import Quaternion
from physics import physics_from_mjcf
from robots import get_robot
from fire import Fire
from base import MjModel
import numpy as np
import mujoco_viewer
import mujoco
import os


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


def main(traj_name='ycb-002_master_chef_can-20200709-subject-01-20200709_141754'):
    object_category = traj_name.split('-')[0]
    object_name = traj_name.split('-')[1]

    traj_path = os.path.join(f'trajectories/{object_category}/{traj_name}.npz')
    traj_file = np.load(traj_path, allow_pickle=True)
    traj_file =  {k:v for k, v in traj_file.items()}
    traj_file['s_0'] = traj_file['s_0'][()]

    init_object_translation = np.array(traj_file['s_0']['pregrasp']['object_translation'])
    init_object_orientation = np.array(traj_file['s_0']['pregrasp']['object_orientation'])

    object_translation = np.array(traj_file['object_translation'])
    object_orientation = np.array(traj_file['object_orientation'])
    hand_joint = np.array(traj_file['hand_joint'])
    pregrasp_joint = np.array(traj_file['s_0']['pregrasp']['position'])

    # object_translation[:, :2] -= init_object_translation[:2]
    # hand_joint[:, :, :2] -= init_object_translation[:2]
    # pregrasp_joint[:, :2] -= init_object_translation[:2]
    # init_object_translation[:2] -= init_object_translation[:2]

    # beta = np.random.uniform(low=0.0, high=2*np.pi)
    # rot_matrix = np.array([[np.cos(beta), -np.sin(beta), 0], [np.sin(beta), np.cos(beta), 0], [0, 0, 1]])
    # object_translation = (rot_matrix @ object_translation.transpose(1, 0)).transpose(1, 0)
    # for idx in range(object_orientation.shape[0]):
        # object_orientation[idx] = Quaternion(matrix=(rot_matrix @ Quaternion(object_orientation[idx]).rotation_matrix)).elements
    # hand_joint = (rot_matrix[None] @ hand_joint.transpose(0, 2, 1)).transpose(0, 2, 1)
    # pregrasp_joint = (rot_matrix @ pregrasp_joint.transpose(1, 0)).transpose(1, 0)
    # init_object_translation = rot_matrix @ init_object_translation
    # init_object_orientation = Quaternion(matrix=rot_matrix @ Quaternion(init_object_orientation).rotation_matrix).elements

    # new_init_pos = np.random.uniform(low=-0.15, high=0.15, size=2)
    # object_translation[:, :2] += new_init_pos 
    # hand_joint[:, :, :2] += new_init_pos
    # pregrasp_joint[:, :2] += new_init_pos
    # init_object_translation[:2] += new_init_pos

    env = TableEnv()
    robot_model = get_robot('adroit')(limp=False)
    env.attach(robot_model)

    object_model = object_generator(f"objects/{object_category}/{object_name}.xml")(pos=init_object_translation, quat=init_object_orientation)
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

    model = physics.model.ptr
    data = physics.data.ptr
    viewer = mujoco_viewer.MujocoViewer(model, data)

    # simulate and render
    for _ in range(5000):
        if viewer.is_alive:
            physics.data.qpos[:30] = np.zeros(30)
            physics.data.qvel[:30] = np.zeros(30)

            physics.step()
            viewer.render()
        else:
            break
    
    # close
    viewer.close()

if __name__ == '__main__':
    Fire(main)
