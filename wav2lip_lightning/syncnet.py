import torch
from pytorch_lightning import Trainer
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
from glob import glob
import os, random, cv2, argparse
from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from wav2lip_lightning import audio
from wav2lip_lightning.hparams import hparams, get_image_list, hparams_debug_string
from wav2lip_lightning.conv import Conv2d

syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, split, hparams, args):
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                continue
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read: continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y


# Binary Cross Entropy loss
logloss = nn.BCELoss()


def cosine_loss(audio_embedding, face_embedding, y):
    d = F.cosine_similarity(audio_embedding, face_embedding)
    loss = logloss(d.unsqueeze(1), y)
    return loss


class SyncNet_color(pl.LightningModule):

    def __init__(self, learning_rate=1e-4):
        super().__init__()

        self.learning_rate = learning_rate
        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        return audio_embedding, face_embedding

    def training_step(self, batch, batch_idx):
        face_sequences, audio_sequences, y = batch

        # forward pass through both encoders
        audio_embedding = self.audio_encoder(audio_sequences)
        face_embedding = self.face_encoder(face_sequences)

        # reshape embeddings
        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        # nomalize embeddings
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        # calculate loss
        loss = cosine_loss(audio_embedding, face_embedding, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.learning_rate)
        return optimizer

def repl_test():

    pl.seed_everything(1234)

    # ------------
    # args (include hparams.py)
    # ------------
    torch.multiprocessing.freeze_support()
    import sys
    sys.argv = [''] #'--resume_from_checkpoint', './lipsync_expert.pth']
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--data_root', default="./sample_data/")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset and Dataloader setup
    train_dataset = Dataset('train', hparams, args)
    val_dataset = Dataset('val', hparams, args)

    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=1)

    val_data_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        num_workers=1)



    model = SyncNet_color()
    model.load_from_checkpoint('./lipsync_expert.pth')
    #trainer = Trainer(num_processes=8, fast_dev_run=False, auto_lr_find=True, log_every_n_steps=1, flush_logs_every_n_steps=1) #, accelerator='ddp')
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_data_loader, val_data_loader)


def debug():
    import matplotlib.pyplot as pl

    lr_finder = trainer.tuner.lr_find(model, train_data_loader)

    # Inspect results
    fig = lr_finder.plot();
    fig.show()
    suggested_lr = lr_finder.suggestion()
    model.learning_rate = suggested_lr

    face_sequences, audio_sequences, y = next(iter(train_data_loader))
    batch = next(iter(train_data_loader))

    model.face_encoder(face_sequences)
    model.audio_encoder(audio_sequences)

    for f in face_sequences:
        pl.imshow(f)
        pl.pause(.1)
        pl.draw()
