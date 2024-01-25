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


SEQ_LEN = {
	"ZY20210800004-H4-C2-N39-S10-s05-T5": 100,
	# "ZY20210800004-H4-C2-N19-S13-s01-T5": 130,
}


def main(traj_name=None):
    with open('trajectories/retarget_result_hoi4d.pkl', 'rb') as f:
        retarget_data = pickle.load(f)

    for key_name in retarget_data.keys():
        if key_name not in list(SEQ_LEN.keys()):
            continue

        origin_length = SEQ_LEN[key_name]
        object_name = retarget_data[key_name]['object_name']
        object_category = 'hoi4d'
        traj_name = '-'.join([object_category, object_name, key_name.replace('/', '-')])

        output_data = dict()
        output_path = os.path.join(f'trajectories/{object_category}/{traj_name}.npz')

        output_data['object_name'] = "025_mug"
        output_data['SIM_SUBSTEPS'] = 10
        output_data['DATA_SUBSTEPS'] = 1
        output_data['object_translation'] = np.array(retarget_data[key_name]['object_pose'][:, :3, 3], dtype=np.float32)
        output_data['robot_qpos'] = np.array(retarget_data[key_name]['robot_qpos'], dtype=np.float32)
        output_data['robot_qpos'][:, 0] += output_data['object_translation'][0, 0]
        output_data['robot_qpos'][:, 2] -= output_data['object_translation'][0, 1]
        output_data['robot_qpos'][:, 1] -= output_data['object_translation'][0, 2]
        output_data['robot_qpos'][:, 1] += 0.03

        output_data['robot_jpos'] = np.array(retarget_data[key_name]['robot_jpos'], dtype=np.float32)
        output_data['robot_jpos'] = output_data['robot_jpos'] - output_data['object_translation'][0, :]
        output_data['robot_jpos'][:, :, 2] += 0.03

        output_data['object_translation'][:, 0:3] = output_data['object_translation'][:, 0:3] - output_data['object_translation'][0, 0:3]
        output_data['object_translation'][:, 2] += 0.03
        obj_quat_list = list()
        for idx in range(len(output_data['object_translation'])):
            obj_quat = list(R.from_matrix(retarget_data[key_name]['object_pose'][idx, :3, :3]).as_quat()[[-1, 0, 1, 2]])
            obj_quat_list.append(obj_quat)
        output_data['object_orientation'] = np.array(obj_quat_list, dtype=np.float32)

        output_data['robot_qpos'] = output_data['robot_qpos'][0:origin_length:3]
        output_data['robot_jpos'] = output_data['robot_jpos'][0:origin_length:3]
        output_data['object_translation'] = output_data['object_translation'][0:origin_length:3]
        output_data['object_orientation'] = output_data['object_orientation'][0:origin_length:3]
        output_data['length'] = len(output_data['object_orientation'])

        env = TableEnv()
        robot_model = get_robot('adroit')(limp=False)
        robot_mesh_list = robot_model.mjcf_model.find_all('geom')
        robot_geom_names = [geom.get_attributes()['name'] for geom in robot_mesh_list]
        robot_geom_names = [f'adroit/{name}' for name in robot_geom_names if 'C' in name]
        env.attach(robot_model)

        object_model = object_generator("objects/ycb/025_mug.xml")(pos=output_data['object_translation'][0], quat=output_data['object_orientation'][0])
        object_mesh_list = object_model.mjcf_model.find_all('geom')
        object_geom_names = [geom.get_attributes()['name'] for geom in object_mesh_list]
        object_geom_names = [f'025_mug/{name}' for name in object_geom_names if 'contact' in name]

        env.attach(object_model)
        physics = physics_from_mjcf(env)

        model = physics.model.ptr
        data = physics.data.ptr

        is_contact = False
        pregrasp_step = 0
        for idx in range(output_data['length']):
            physics.reset()
            physics.data.qpos[:30] = output_data['robot_qpos'][idx]
            physics.data.qvel[:30] = np.zeros(30)

            physics.forward()
            is_contact = check_contacts(physics, robot_geom_names, object_geom_names)
            if is_contact:
                break
            pregrasp_step += 1
        pregrasp_step -= 1

        if pregrasp_step < 7:
            continue
        output_data['pregrasp_step'] = pregrasp_step
        np.savez(output_path, **output_data)


if __name__ == '__main__':
    Fire(main)
