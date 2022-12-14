import torch

from options.train_options import TrainOptions
from train_net import train
from util.misc import launch_job


# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
def main():
    cfg = TrainOptions().parse()  # get training options
    cfg.NUM_GPUS = torch.cuda.device_count()
    cfg.batch_size = int(cfg.batch_size / max(1, cfg.NUM_GPUS))
    cfg.phase = 'train'
    launch_job(cfg=cfg, init_method=cfg.init_method, func=train)


if __name__ == "__main__":
    main()
