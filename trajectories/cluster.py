import os
import numpy as np
from sklearn.cluster import KMeans

pregrasp_data = []
traj_names = []
root_dir = "ycb"

for filename in os.listdir(root_dir):
    traj_path = os.path.join(root_dir, filename)
    traj_data = np.load(traj_path, allow_pickle=True)
    traj_data = {k:v for k, v in traj_data.items()}
    traj_data['s_0'] = traj_data['s_0'][()]

    pregrasp_data.append(traj_data['s_0']['pregrasp']['canonical_position'])
    traj_names.append(filename)

n_clusters = 6
pregrasp_data = np.array(pregrasp_data).reshape((-1, 63))
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pregrasp_data.reshape((-1, 63)))
labels = kmeans.labels_

group_names = []
for group_id in range(n_clusters):
    group_name = []
    group_list = np.where(labels == group_id)[0]
    for idx in group_list:
        group_name.append(traj_names[idx])
    group_names.append(group_name)
