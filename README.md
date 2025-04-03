# MuJoCo simulation assets
Some assets for simulating robots and objects in the mujoco simulator for my **[ViViDex](https://arxiv.org/abs/2404.15709)** project.


## Installation 👷
```
git clone -b dev https://github.com/zerchen/object_sim.git
conda create -n mujoco python==3.8
pip install -r requirements.txt
```

## Usage 💻
```
# Visualize the environment
python vis.py

# Visualize the pregrasp for videos in DexYCB dataset
python vis_pregrasp.py

# Visualize augmented trajectories
python vis_traj.py
```

## License 📚
This code is distributed under an [MIT License](LICENSE).