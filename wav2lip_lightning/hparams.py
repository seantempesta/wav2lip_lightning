import inspect
from glob import glob
import os
from pathlib import Path
import random


def repl_test():
    data_root = '/Users/sean/src/wav2lip_lightning/sample_data/'
    write_to_path = '/Users/sean/src/wav2lip_lightning/filelists'
    train_split = 0.95


def gen_train_val_lists(data_root, write_to_path, train_split):
    # recursively search for audio.wav files indicating a data path
    data_paths = []
    for path in Path(data_root).rglob('audio.wav'):
        data_paths.append(path.parent)

    # sort randomly
    random.shuffle(data_paths)

    # split into train/val
    train_split_num = int(len(data_paths) * train_split)
    val_split_num = len(data_paths) - train_split_num
    train = data_paths[0:train_split_num]
    val = data_paths[train_split_num:]

    # write files to disk
    with open(os.path.join(write_to_path, 'train.txt'), 'w') as f:
        for item in train:
            f.write("%s\n" % item)
    with open(os.path.join(write_to_path, 'val.txt'), 'w') as f:
        for item in val:
            f.write("%s\n" % item)


def get_image_list(data_root, split):
    filelist = []

    with open('filelists/{}.txt'.format(split)) as f:
        for line in f:
            line = line.strip()
            if ' ' in line: line = line.split()[0]
            filelist.append(os.path.join(data_root, line))

    return filelist


# Default hyperparameters
class HParams():
    num_mels = 80  # Number of mel-spectrogram channels and local conditioning dimensionality
    #  network
    rescale = True  # Whether to rescale audio prior to preprocessing
    rescaling_max = 0.9  # Rescaling value

    # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
    # It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
    # Does not work if n_ffit is not multiple of hop_size!!
    use_lws = False

    n_fft = 800  # Extra window size is filled with 0 paddings to match this parameter
    hop_size = 200  # For 16000Hz 200 = 12.5 ms (0.0125 * sample_rate)
    win_size = 800  # For 16000Hz 800 = 50 ms (If None win_size = n_fft) (0.05 * sample_rate)
    sample_rate = 16000  # 16000Hz (corresponding to librispeech) (sox --i <filename>)

    frame_shift_ms = None  # Can replace hop_size parameter. (Recommended: 12.5)

    # Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization = True
    # Whether to normalize mel spectrograms to some predefined range (following below parameters)
    allow_clipping_in_normalization = True  # Only relevant if mel_normalization = True
    symmetric_mels = True
    # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2
    # faster and cleaner convergence)
    max_abs_value = 4.
    # max absolute value of data. If symmetric data will be [-max max] else [0 max] (Must not
    # be too big to avoid gradient explosion
    # not too small for fast convergence)
    # Contribution by @begeekmyfriend
    # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude
    # levels. Also allows for better G&L phase reconstruction)
    preemphasize = True  # whether to apply filter
    preemphasis = 0.97  # filter coefficient.

    # Limits
    min_level_db = -100
    ref_level_db = 20
    fmin = 55
    # Set this to 55 if your speaker is male! if female 95 should help taking off noise. (To
    # test depending on dataset. Pitch info: male~[65 260] female~[100 525])
    fmax = 7600  # To be increased/reduced depending on data.

    ###################### Our training parameters #################################
    img_size = 96
    fps = 25

    batch_size = 16
    initial_learning_rate = 1e-4
    nepochs = 200000000000000000  ### ctrl + c stop whenever eval loss is consistently greater than train loss for ~10 epochs
    num_workers = 32
    checkpoint_interval = 3000
    eval_interval = 3000
    save_optimizer_state = True
    syncnet_wt = 0.0  # is initially zero will be set automatically to 0.03 later. Leads to faster convergence.
    syncnet_batch_size = 32
    syncnet_lr = 1e-4
    syncnet_eval_interval = 1000
    syncnet_checkpoint_interval = 10000

    disc_wt = 0.07
    disc_initial_learning_rate = 1e-4


hparams = HParams()


def hparams_debug_string():
    attributes = inspect.getmembers(HParams, lambda a: not (inspect.isroutine(a)))
    values = [[a[0],a[1]] for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]
    hp = ["  %s: %s" % (v[0], v[1]) for v in sorted(values)]
    return "Hyperparameters:\n" + "\n".join(hp)
