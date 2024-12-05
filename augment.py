import Augmentor

def augment(path, probability, sample_size):
    #pth = 'Dataset/' + path + '/' + classification + '/'
    pth = path
    p = Augmentor.Pipeline(pth)
    p.random_distortion(probability=probability, grid_width=3, grid_height=3, magnitude=3)
    p.gaussian_distortion(probability=probability, grid_width=3, grid_height=3, magnitude=3, corner='bell', method='in')
    p.skew(probability=probability)
    p.skew_tilt(probability=probability)
    p.skew_left_right(probability=probability)
    p.skew_top_bottom(probability=probability)
    p.skew_corner(probability=probability)
    p.sample(sample_size)


from tqdm import tqdm
from pathlib import Path

PROBABILITY = 0.1
SAMPLE_SIZE = 10000
augment(path='/media/envisage/backup8tb/Padchest/projectionClassificationImages/train/AP/', probability=PROBABILITY, sample_size=SAMPLE_SIZE)
augment(path='/media/envisage/backup8tb/Padchest/projectionClassificationImages/AP_horizontal', probability=PROBABILITY, sample_size=SAMPLE_SIZE)
augment(path='/media/envisage/backup8tb/Padchest/projectionClassificationImages/L/', probability=PROBABILITY, sample_size=SAMPLE_SIZE)
augment(path='/media/envisage/backup8tb/Padchest/projectionClassificationImages/PA', probability=PROBABILITY, sample_size=SAMPLE_SIZE)
# /media/envisage/backup8tb/Padchest/projectionClassificationImages