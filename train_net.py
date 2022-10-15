import datetime
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter

from data import create_dataset
from data import shuffle_dataset
from models import create_model
from options.test_options import TestOptions
from util import distributed as du
from util import html, util
from util.evaluation import evaluation
from util.visualizer import Visualizer
from util.visualizer import save_images


def train(cfg):
    # init
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")[2:]
    writer = SummaryWriter(logdir=os.path.join(cfg.tensorboard_dir, date + '-' + cfg.name))

    # init dataset
    dataset = create_dataset(cfg)  # create a dataset given cfg.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    postion_embedding = util.PositionEmbeddingSine(cfg)
    patch_pos = util.PatchPositionEmbeddingSine(cfg)
    model = create_model(cfg)  # create a model given cfg.model and other options
    model.set_position(postion_embedding, patch_pos=patch_pos)
    # model.setup(cfg)               # regular setup: load and print networks; create schedulers

    visualizer = Visualizer(cfg)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
    # cur_device = torch.cuda.current_device()
    is_master = du.is_master_proc(cfg.NUM_GPUS)

    best_mse, best_fmse = 10000, 10000
    for epoch in range(cfg.epoch_count,
                       cfg.niter + cfg.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        if is_master:
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        shuffle_dataset(dataset, epoch)
        losses = None
        for i, data in enumerate(dataset):  # inner loop within one epoch
            if is_master:
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % cfg.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                    iter_data_time = time.time()
            visualizer.reset()
            total_iters += cfg.batch_size
            epoch_iter += cfg.batch_size
            if epoch == cfg.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(cfg)  # regular setup: load and print networks; create schedulers
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % cfg.display_freq == 0 and is_master:  # display images on visdom and save images to a HTML file
                save_result = total_iters % cfg.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            losses = model.get_current_losses()
            if cfg.NUM_GPUS > 1:
                losses = du.all_reduce(losses)
            if total_iters % cfg.print_freq == 0 and is_master:  # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / cfg.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if cfg.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
            if total_iters % cfg.save_latest_freq == 0 and is_master:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if cfg.save_by_iter else 'latest'
                model.save_networks(save_suffix)

        if epoch % cfg.save_epoch_freq == 0 and is_master:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            if cfg.save_iter_model and epoch >= 55:
                model.save_networks(epoch)

            cur_mse, cur_fmse = eval(cfg.name, cfg.model, cfg.mode, cfg.dataset_root)
            if cur_mse < best_mse and cur_fmse < best_fmse:
                best_mse, best_fmse = cur_mse, cur_fmse
                model.save_networks('best')

        if is_master:
            print('End of epoch %d / %d \t Time Taken: %d sec' % (
                epoch, cfg.niter + cfg.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # update learning rates at the end of every epoch.
        for k, v in losses.items():
            writer.add_scalar(f'data/loss_{k}', v, epoch)
    if is_master:
        writer.close()
        print(f'Best result in HCOCO: MSE {best_mse} | fMSE {best_fmse}')


def test(cfg):
    date = datetime.datetime.now().strftime("%Y%m%d")[2:]
    writer = SummaryWriter(logdir=os.path.join(cfg.tensorboard_dir, date + '-' + cfg.name))

    dataset = create_dataset(cfg)  # create a dataset given cfg.dataset_mode and other options
    postion_embedding = util.PositionEmbeddingSine(cfg)
    patch_pos = util.PatchPositionEmbeddingSine(cfg)
    model = create_model(cfg)  # create a model given cfg.model and other options
    model.set_position(postion_embedding, patch_pos=patch_pos)
    model.setup(cfg)  # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = os.path.join(cfg.results_dir, cfg.name, '%s_%s' % (cfg.phase, cfg.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (cfg.name, cfg.phase, cfg.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if cfg.eval:
        model.eval()
    ismaster = du.is_master_proc(cfg.NUM_GPUS)

    fmse_score_list = []
    mse_scores = 0
    fmse_scores = 0
    num_image = 0
    # print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for i, data in enumerate(dataset):
        # if i >= 100:
        #     print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        #     break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results

        img_path = model.get_image_paths()  # get image paths # Added by Mia
        if i % 5 == 0 and ismaster:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        visuals_ones = OrderedDict()
        harmonized = None
        real = None
        for j in range(len(img_path)):
            img_path_one = []
            for label, im_data in visuals.items():
                visuals_ones[label] = im_data[j:j + 1, :, :, :]
            img_path_one.append(img_path[j])
            save_images(webpage, visuals_ones, img_path_one, aspect_ratio=cfg.aspect_ratio, width=cfg.display_winsize)
            num_image += 1
            raw_name = img_path[j].split('/')[-1]

            mse_score, fmse_score, score_str = evaluation(raw_name, visuals_ones['harmonized'] * 256,
                                                          visuals_ones['real'] * 256, visuals_ones['mask'])
            # print(score_str)
            fmse_score_list.append(score_str)
            mse_scores += mse_score
            fmse_scores += fmse_score
            visuals_ones.clear()

    webpage.save()  # save the HTML
    mse_mu = mse_scores / num_image
    fmse_mu = fmse_scores / num_image
    mean_score = "%s MSE %0.2f | fMSE %0.2f" % (cfg.dataset_name, mse_mu, fmse_mu)
    print(mean_score)

    dataset_list = ['ihd', 'HAdobe5k', 'HCOCO', 'HFlickr', 'Hday2night']
    writer.add_text('eval/metrics', mean_score, dataset_list.index(cfg.dataset_name))
    writer.close()

    fmse_score_list = sorted(fmse_score_list, key=lambda image: image[1], reverse=True)
    save_fmse_root = os.path.join(cfg.results_dir, cfg.name)
    # save_fmse_root = cfg.result_save_path[0:-19]
    save_fmse_path = os.path.join(save_fmse_root, "evaluation_detail_" + cfg.test_epoch + ".txt")

    file = open(save_fmse_path, 'w')
    file.write(str(num_image))
    file.write('\n')
    file.write(mean_score)
    file.write('\n')
    # lists=[str(line)+"\n" for line in fmse_score_list]
    for line in fmse_score_list:
        file.write(str(line) + "\n")
    file.close()


def eval(name, model, mode, root):
    cfg = TestOptions().parse()  # get training options
    cfg.NUM_GPUS = 1
    cfg.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    cfg.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    cfg.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    cfg.phase = 'test'
    cfg.batch_size = 8
    cfg.name = name
    cfg.dataset_name = 'HCOCO'
    cfg.dataset_mode = mode
    cfg.model = model
    cfg.dataset_root = root

    dataset = create_dataset(cfg)  # create a dataset given cfg.dataset_mode and other options
    postion_embedding = util.PositionEmbeddingSine(cfg)
    patch_pos = util.PatchPositionEmbeddingSine(cfg)
    model = create_model(cfg)  # create a model given cfg.model and other options
    model.set_position(postion_embedding, patch_pos=patch_pos)
    model.setup(cfg)  # regular setup: load and print networks; create schedulers

    if cfg.eval:
        model.eval()
    ismaster = du.is_master_proc(cfg.NUM_GPUS)

    fmse_score_list = []
    mse_scores = 0
    fmse_scores = 0
    num_image = 0
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results

        img_path = model.get_image_paths()  # get image paths # Added by Mia
        if i % 5 == 0 and ismaster:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        visuals_ones = OrderedDict()
        harmonized = None
        real = None
        for j in range(len(img_path)):
            img_path_one = []
            for label, im_data in visuals.items():
                visuals_ones[label] = im_data[j:j + 1, :, :, :]
            img_path_one.append(img_path[j])
            num_image += 1
            raw_name = img_path[j].split('/')[-1]

            mse_score, fmse_score, score_str = evaluation(raw_name, visuals_ones['harmonized'] * 256,
                                                          visuals_ones['real'] * 256, visuals_ones['mask'])
            # print(score_str)
            fmse_score_list.append(score_str)
            mse_scores += mse_score
            fmse_scores += fmse_score
            visuals_ones.clear()

    mse_mu = mse_scores / num_image
    fmse_mu = fmse_scores / num_image
    mean_score = "%s MSE %0.2f | fMSE %0.2f" % (cfg.dataset_name, mse_mu, fmse_mu)
    print(mean_score)

    return mse_mu, fmse_mu
