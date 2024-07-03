# Like in CAI project: this file should be called to start the program. it should contain options to train, test and restore the model
from Architecture.train import *
import utils.utils as utils

min_size = 25
scale_factor_init = 0.75
max_size = 500

G_pyramid = []
Noise_pyramid = []
reals = []
NoiseAmp = []
input_path = '../../input/input-dgm'


input_name = 'starry_night'
input_full_name = input_name + '.png'
real = utils.read_image(input_path, input_full_name)
x_, scale_stop, scale_factor = utils.adjust_scales2image(
    real, min_size, scale_factor_init, max_size)
train_pyramid(G_pyramid, Noise_pyramid, input_path,
              input_name, NoiseAmp, scale_stop=scale_stop, )

G_pyramid = []
Noise_pyramid = []
reals = []
NoiseAmp = []
input_name = 'Sailboats'
input_full_name = input_name + '.png'
real = utils.read_image(input_path, input_full_name)
x_, scale_stop, scale_factor = utils.adjust_scales2image(
    real, min_size, scale_factor_init, max_size)
train_pyramid(G_pyramid, Noise_pyramid, input_path,
              input_name, NoiseAmp, scale_stop=scale_stop, )
G_pyramid = []
Noise_pyramid = []
reals = []
NoiseAmp = []
input_name = 'CoastalScene'
input_full_name = input_name + '.png'
real = utils.read_image(input_path, input_full_name)
x_, scale_stop, scale_factor = utils.adjust_scales2image(
    real, min_size, scale_factor_init, max_size)
train_pyramid(G_pyramid, Noise_pyramid, input_path,
              input_name, NoiseAmp, scale_stop=scale_stop, )
