from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb



experiment_id = "Oct26_11-52-56_envisage-WRX80-SU8-N-A_FINETUNE_SEED_NUMBER_2/events.out.tfevents.1698310376.envisage-WRX80-SU8-N-A.1666115.0"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()

print(df)