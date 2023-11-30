from dm_control import mjcf
from dm_control.rl import control
from dm_control.mjcf import debugging
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


def main(object_name="006_mustard_bottle", object_category="ycb"):
    if object_name == None:
        object_name_list = os.listdir(object_category)
    else:
        object_name_list = [object_name]

    for object_name in sorted(object_name_list):
        env = TableEnv()
        robot_model = get_robot('adroit')(limp=False)
        env.attach(robot_model)

        object_model = object_generator(f"objects/{object_category}/{object_name}.xml")()
        object_model.mjcf_model.worldbody.add('body', name='object_marker', pos=np.array([0.2, 0.2, 0.2]))
        object_model.mjcf_model.worldbody.body['object_marker'].add('geom', contype='0', conaffinity='0', mass='0', name='target_visual', mesh=object_model.mjcf_model.worldbody.body['object_entity'].geom['entity_visual'].mesh, rgba=np.array([0, 1, 0, 0.125]))
        object_model.mjcf_model.worldbody.body['object_marker'].geom['target_visual'].type = "mesh"
        env.attach(object_model)
        physics = physics_from_mjcf(env)

        model = physics.model.ptr
        data = physics.data.ptr
        viewer = mujoco_viewer.MujocoViewer(model, data)

        # simulate and render
        for _ in range(1000):
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
