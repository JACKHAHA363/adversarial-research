from subprocess import call
import numpy as np
from tqdm import tqdm
import os

num_ensembles = 16
model_list = ['A', 'B', 'C', 'D']
ensemble_dir = "./ensembles/"

if os.path.isdir(ensemble_dir) is not True:
    os.mkdir(ensemble_dir)

cmd = [
    "python", "train.py",
    "--num-epoch", "24",
    "--temp", "1",
    "--verbose", "false",
]
mod_inds = np.random.choice(len(model_list), num_ensembles, True)
for i in tqdm(range(num_ensembles)):
    # randomly pick one model and train adversarially
    model = model_list[mod_inds[i]]
    model_prefix = os.path.join(ensemble_dir, "model{}".format(i))
    tbdir = "./ensembles_logs/model{}".format(i)
    cmd += [
        "--model", model,
        "--model-prefix", model_prefix,
        "--tbdir", tbdir,
        "--adv", 'false',
    ]
    call(cmd)



