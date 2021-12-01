import torch
from math import log2

START_TRAIN_AT_IMG_SIZE = 4
DATASET = "celeba_hq"
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
BATCH_SIZES = [16, 16, 16, 16, 16, 16, 16, 8, 4]
IMAGE_SIZE = 512
CHANNELS_IMG = 3
Z_DIM = 256
IN_CHANNELS = 256
LAMBDA_GP = 10
NUM_STEPS = int(log2(IMAGE_SIZE / 4)) + 1

PROGRESSIVE_EPOCHS = [20] * len(BATCH_SIZES)
FIXED_NOISE = torch.rand(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4