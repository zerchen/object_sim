from dm_control import mjcf
from dm_control.rl import control
from dm_control.mjcf import debugging
from pyquaternion import Quaternion
from fire import Fire
import numpy as np
import mujoco_viewer
import mujoco
import os


class MjModel(object):
    """ """

    def __init__(self, mjcf_model):
        self._mjcf_model = mjcf_model

    def attach(self, other):
        """ """
        self.mjcf_model.attach(other.mjcf_model)
        other._post_attach(self.mjcf_model)

    @property
    def mjcf_model(self):
        return self._mjcf_model

    def _post_attach(self, base_mjfc_model):
        """ """

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

    object_translation[:, :2] -= init_object_translation[:2]
    hand_joint[:, :, :2] -= init_object_translation[:2]
    pregrasp_joint[:, :2] -= init_object_translation[:2]
    init_object_translation[:2] -= init_object_translation[:2]

    beta = np.random.uniform(low=0.0, high=2*np.pi)
    rot_matrix = np.array([[np.cos(beta), -np.sin(beta), 0], [np.sin(beta), np.cos(beta), 0], [0, 0, 1]])
    object_translation = (rot_matrix @ object_translation.transpose(1, 0)).transpose(1, 0)
    for idx in range(object_orientation.shape[0]):
        object_orientation[idx] = Quaternion(matrix=(rot_matrix @ Quaternion(object_orientation[idx]).rotation_matrix)).elements
    hand_joint = (rot_matrix[None] @ hand_joint.transpose(0, 2, 1)).transpose(0, 2, 1)
    pregrasp_joint = (rot_matrix @ pregrasp_joint.transpose(1, 0)).transpose(1, 0)
    init_object_translation = rot_matrix @ init_object_translation
    init_object_orientation = Quaternion(matrix=rot_matrix @ Quaternion(init_object_orientation).rotation_matrix).elements

    new_init_pos = np.random.uniform(low=-0.15, high=0.15, size=2)
    object_translation[:, :2] += new_init_pos 
    hand_joint[:, :, :2] += new_init_pos
    pregrasp_joint[:, :2] += new_init_pos
    init_object_translation[:2] += new_init_pos

    env = TableEnv()
    object_model = object_generator(f"objects/{object_category}/{object_name}.xml")(pos=init_object_translation, quat=init_object_orientation)
    for idx, _ in enumerate(traj_file['object_translation'][1:]):
        if idx % 1 == 0:
            object_model.mjcf_model.worldbody.add('body', name=f'object_marker_{idx}', pos=object_translation[idx], quat=object_orientation[idx])
            object_model.mjcf_model.worldbody.body[f'object_marker_{idx}'].add('geom', contype='0', conaffinity='0', mass='0', name=f'target_visual_{idx}', mesh=object_model.mjcf_model.worldbody.body['object_entity'].geom['entity_visual'].mesh, rgba=np.array([0.996, 0.878, 0.824, 0.125]))
            object_model.mjcf_model.worldbody.body[f'object_marker_{idx}'].geom[f'target_visual_{idx}'].type = "mesh"

    env.attach(object_model)
    mjcf.export_with_assets(env.mjcf_model, out_dir="cache")

    model = mujoco.MjModel.from_xml_path('cache/table-environment.xml')
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)

        # simulate and render
    for _ in range(5000):
        if viewer.is_alive:
            mujoco.mj_step(model, data)
            viewer.render()
        else:
            break
    
    # close
    viewer.close()

if __name__ == '__main__':
    Fire(main)
