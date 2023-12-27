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


def main(traj_name=None):
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
        init_object_translation = np.array(traj_file['object_translation'][pregrasp_step]).copy()
        init_object_orientation = np.array(traj_file['object_orientation'][pregrasp_step]).copy()

        object_translation = np.array(traj_file['object_translation'])
        object_orientation = np.array(traj_file['object_orientation'])

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

        final_goal = np.array([np.random.uniform(low=-0.3, high=0.1), np.random.uniform(low=-0.15, high=0.15), np.random.uniform(low=0.15, high=0.25)], dtype=np.float32)
        min_idx = np.where(object_translation[:, 2] - object_translation[0, 2] > 0.1)[0][0]
        object_translation = object_translation[:min_idx + 1]
        object_orientation = object_orientation[:min_idx + 1]

        for idx, _ in enumerate(object_translation[1:]):
            if idx % 1 == 0:
                object_model.mjcf_model.worldbody.add('body', name=f'object_marker_{idx}', pos=object_translation[idx], quat=object_orientation[idx])
                object_model.mjcf_model.worldbody.body[f'object_marker_{idx}'].add('geom', contype='0', conaffinity='0', mass='0', name=f'target_visual_{idx}', mesh=object_model.mjcf_model.worldbody.body['object_entity'].geom['entity_visual'].mesh, rgba=np.array([0.996, 0.878, 0.824, 0.125]))
                object_model.mjcf_model.worldbody.body[f'object_marker_{idx}'].geom[f'target_visual_{idx}'].type = "mesh"

        object_model.mjcf_model.worldbody.add('body', name=f'object_marker', pos=final_goal, quat=object_orientation[-1])
        object_model.mjcf_model.worldbody.body[f'object_marker'].add('geom', contype='0', conaffinity='0', mass='0', name='target_visual', mesh=object_model.mjcf_model.worldbody.body['object_entity'].geom['entity_visual'].mesh, rgba=np.array([0, 1, 0, 0.125]))
        object_model.mjcf_model.worldbody.body[f'object_marker'].geom['target_visual'].type = "mesh"

        dist_vec = final_goal - object_translation[-1]
        unit_dist_vec = dist_vec / np.linalg.norm(dist_vec)
        relocate_step = 0.025
        num_step = int(np.linalg.norm(dist_vec) // relocate_step)

        syn_object_translation = []
        for idx in range(num_step):
            step_size = (idx + 1) * relocate_step
            syn_object_translation.append(object_translation[-1] + step_size * unit_dist_vec)

        if np.linalg.norm(final_goal - syn_object_translation[-1]) > 0:
            num_step += 1
            syn_object_translation.append(final_goal)

        total_steps = len(object_translation) + len(syn_object_translation)
        print(f'total_steps: {total_steps}')

        for idx in range(num_step):
            object_model.mjcf_model.worldbody.add('body', name=f'object_marker_syn_{idx}', pos=syn_object_translation[idx], quat=object_orientation[-1])
            object_model.mjcf_model.worldbody.body[f'object_marker_syn_{idx}'].add('geom', contype='0', conaffinity='0', mass='0', name=f'target_visual_syn_{idx}', mesh=object_model.mjcf_model.worldbody.body['object_entity'].geom['entity_visual'].mesh, rgba=np.array([1, 0, 0, 0.125]))
            object_model.mjcf_model.worldbody.body[f'object_marker_syn_{idx}'].geom[f'target_visual_syn_{idx}'].type = "mesh"

        syn_retarget_pose = []
        for idx in range(len(syn_object_translation)):
            cur_qpos = retarget_pose[min_idx].copy()
            cur_qpos[0] -= (syn_object_translation[idx][0] - object_translation[-1][0])
            cur_qpos[1] += (syn_object_translation[idx][2] - object_translation[-1][2])
            cur_qpos[2] += (syn_object_translation[idx][1] - object_translation[-1][1])
            syn_retarget_pose.append(cur_qpos)

        env.attach(object_model)
        physics = physics_from_mjcf(env)

        model = physics.model.ptr
        data = physics.data.ptr

        physics.reset()
        viewer = mujoco_viewer.MujocoViewer(model, data)
        # simulate and render
        for _ in range(1000):
            if viewer.is_alive:
                physics.data.qpos[:30] = syn_retarget_pose[-1]
                physics.data.qvel[:30] = np.zeros(30)
                physics.step()
                viewer.render()
            else:
                break

        # close
        viewer.close()

if __name__ == '__main__':
    Fire(main)
