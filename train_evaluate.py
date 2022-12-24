import time
from options.train_options import TrainOptions
from data import CustomDataset
from models import create_model
from torch.utils.tensorboard import SummaryWriter
import os
from util import util
import numpy as np
import torch
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from skimage import data, io

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def calculateMean(vars):
    return sum(vars) / len(vars)

def save_img(path, img):
    fold, name = os.path.split(path)
    os.makedirs(fold, exist_ok=True)
    io.imsave(path, img)

def evaluateModel(epoch_number, model, opt, test_dataset, epoch, max_psnr, iters=None):
    
    model.netG.eval()
    
    if iters is not None:
        eval_path = os.path.join(opt.checkpoints_dir, opt.name, 'Eval_%s_iter%d.csv' % (epoch, iters))  # define the website directory
    else:
        eval_path = os.path.join(opt.checkpoints_dir, opt.name, 'Eval_%s.csv' % (epoch))  # define the website directory
    eval_results_fstr = open(eval_path, 'w')
    eval_results = {'mask': [], 'mse': [], 'psnr': [], 'fmse':[], 'ssim':[]}

    for i, data in tqdm(enumerate(test_dataset)):
        model.set_input(data)  # unpack data from data loader
        model.test()  # inference
        visuals = model.get_current_visuals()  # get image results
        output = visuals['attentioned']
        real = visuals['real']

        for i_img in range(real.size(0)):
            gt, pred = real[i_img:i_img+1], output[i_img:i_img+1]
            fore_nums = data['mask'][i_img].sum().item()
            mse_score_op = mean_squared_error(util.tensor2im(pred), util.tensor2im(gt))
            psnr_score_op = peak_signal_noise_ratio(util.tensor2im(gt), util.tensor2im(pred), data_range=255)
            fmse_score_op = mean_squared_error(util.tensor2im(pred), util.tensor2im(gt)) * 256 * 256 / fore_nums
            ssim_score = ssim(util.tensor2im(pred), util.tensor2im(gt),data_range=255,multichannel=True)
            
            if epoch >= 100:
                pred_rgb = util.tensor2im(pred)
                img_path = data['img_path'][i_img]
                basename, imagename = os.path.split(img_path)
                basename = basename.split('/')[-2]
                save_img(os.path.join('evaluate', str(epoch_number), 'results',basename, imagename.split('.')[0] + '.png'), pred_rgb)
                
            # update calculator
            eval_results['mse'].append(mse_score_op)
            eval_results['psnr'].append(psnr_score_op)
            eval_results['fmse'].append(fmse_score_op)         
            eval_results['ssim'].append(ssim_score) 
            eval_results['mask'].append(data['mask'][i_img].mean().item())
            eval_results_fstr.writelines('%s,%.3f,%.3f,%.3f\n' % (data['img_path'][i_img], eval_results['mask'][-1],mse_score_op, psnr_score_op))
        if i + 1 % 100 == 0:
            # print('%d images have been processed' % (i + 1))
            eval_results_fstr.flush()
    eval_results_fstr.flush()
    eval_results_fstr.close()

    all_mse, all_psnr, all_fmse, all_ssim = calculateMean(eval_results['mse']), calculateMean(eval_results['psnr']),  calculateMean(eval_results['fmse']),  calculateMean(eval_results['ssim'])
    
    print('MSE:%.3f, PSNR:%.3f, fMSE:%.3f, SSIM:%.3f' % (all_mse, all_psnr, all_fmse, all_ssim))
    model.netG.train()
    return all_mse, all_psnr, resolveResults(eval_results)

def resolveResults(results):
    interval_metrics = {}
    mask, mse, psnr, fmse, ssim = np.array(results['mask']), np.array(results['mse']), np.array(results['psnr']), np.array(results['fmse']), np.array(results['ssim'])
    interval_metrics['0.00-0.05'] = [np.mean(mse[np.logical_and(mask <= 0.05, mask > 0.0)]),
                                    np.mean(psnr[np.logical_and(mask <= 0.05, mask > 0.0)]),
                                    np.mean(fmse[np.logical_and(mask <= 0.05, mask > 0.0)]),
                                    np.mean(ssim[np.logical_and(mask <= 0.05, mask > 0.0)])]

    interval_metrics['0.05-0.15'] = [np.mean(mse[np.logical_and(mask <= 0.15, mask > 0.05)]),
                                    np.mean(psnr[np.logical_and(mask <= 0.15, mask > 0.05)]),
                                    np.mean(fmse[np.logical_and(mask <= 0.15, mask > 0.05)]),
                                    np.mean(ssim[np.logical_and(mask <= 0.15, mask > 0.05)])]

    interval_metrics['0.15-1.00'] = [np.mean(mse[mask > 0.15]),
                                    np.mean(psnr[mask > 0.15]),
                                    np.mean(fmse[mask > 0.15]),
                                    np.mean(ssim[mask > 0.15])]

    print(interval_metrics)
    return interval_metrics

def updateWriterInterval(writer, metrics, epoch):
    for k, v in metrics.items():
        writer.add_scalar('interval/{}-MSE'.format(k), v[0], epoch)
        writer.add_scalar('interval/{}-PSNR'.format(k), v[1], epoch)

if __name__ == '__main__':
    # setup_seed(6)
    opt = TrainOptions().parse()   # get training 
    train_dataset = CustomDataset(opt, is_for_train=True)
    test_dataset = CustomDataset(opt, is_for_train=False)
    train_dataset_size = len(train_dataset)    # get the number of images in the dataset.
    test_dataset_size = len(test_dataset)
    print('The number of training images = %d' % train_dataset_size)
    print('The number of testing images = %d' % test_dataset_size)
    
    train_dataloader = train_dataset.load_data()
    test_dataloader = test_dataset.load_data()
    print('The total batches of training images = %d' % len(train_dataset.dataloader))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations
    writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name))

    max_psnr = 0
    max_epoch = 1
    for epoch in range(opt.load_iter+1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in tqdm(enumerate(train_dataloader)):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += 1
            epoch_iter += 1
            model.set_input(data)         # unpack data from dataset
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
            
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # evaluate for every epoch
        epoch_mse, epoch_psnr, epoch_interval_metrics = evaluateModel(epoch, model, opt, test_dataloader, epoch, max_psnr)
        if epoch_psnr > max_psnr:
            max_psnr = epoch_psnr
            max_epoch = epoch
        print("max_psnr_epoch: " + str(max_epoch))
        writer.add_scalar('overall/MSE', epoch_mse, epoch)
        writer.add_scalar('overall/PSNR', epoch_psnr, epoch)
        updateWriterInterval(writer, epoch_interval_metrics, epoch)

        torch.cuda.empty_cache()

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
        print('Current learning rate: {}'.format(model.schedulers[0].get_lr()))

    writer.close()
