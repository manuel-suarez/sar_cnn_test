import numpy as np
import tensorflow as tf
from PIL import Image
import scipy.ndimage
from scipy import special


# DEFINE PARAMETERS OF SPECKLE AND NORMALIZATION FACTOR
M = 10.089038980848645
m = -1.429329123112601
L = 1
c = (1 / 2) * (special.psi(L) - np.log(L))
cn = c / (M - m)  # normalized (0,1) mean of log speckle

def normalize_sar(im):
    return ((np.log(im + np.spacing(1)) - m) * 255 / (M - m)).astype('float32')

def denormalize_sar(im):
    return np.exp((M - m) * np.clip((np.squeeze(im)).astype('float32'),0,1) + m)

def load_sar_images(filelist):
    if not isinstance(filelist, list):
        im = np.load(filelist)
        im = normalize_sar(im)
        return np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), 1)
    data = []
    for file in filelist:
        im = np.load(file)
        im = normalize_sar(im)
        data.append(np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), 1))
    return data

def store_data_and_plot(im, threshold, filename):
    im = np.clip(im, 0, threshold)
    im = im / threshold * 255
    im = Image.fromarray(im.astype('float64')).convert('L')
    im.save(filename.replace('npy','png'))


def save_sar_images(groundtruth, denoised, noisy, imagename, save_dir, real_sar_flag):
    choices = {'marais1.npy': 190.92, 'lely.npy': 235.90}
    threshold = choices.get('%s' % imagename)
    if threshold==None: threshold = np.mean(groundtruth)+3*np.std(groundtruth)

    if not real_sar_flag: # simulated noisy images
        groundtruthfilename = save_dir+"/groundtruth_"+imagename
        np.save(groundtruthfilename,groundtruth)
        store_data_and_plot(groundtruth, threshold, groundtruthfilename)

    if real_sar_flag: denoised = scipy.ndimage.zoom(denoised, 2, order=1)
    denoisedfilename = save_dir + "/denoised_" + imagename
    np.save(denoisedfilename, denoised)
    store_data_and_plot(denoised, threshold, denoisedfilename)


    noisyfilename = save_dir + "/noisy_" + imagename
    np.save(noisyfilename, noisy)
    store_data_and_plot(noisy, threshold, noisyfilename)

def tf_simulate_speckle(clean_im):
    s = tf.zeros(shape=tf.shape(clean_im))
    for k in range(0, L):
        gamma = (tf.abs(tf.complex(tf.random_normal(shape=tf.shape(clean_im), stddev=1),
                                   tf.random_normal(shape=tf.shape(clean_im), stddev=1))) ** 2) / 2
        s = s + gamma
    s_amplitude = tf.sqrt(s / L)
    log_speckle = tf.log(s_amplitude)
    log_norm_speckle = log_speckle / (M - m)

    noisy_im = clean_im + log_norm_speckle
    return noisy_im
