from peft import get_peft_model, LoraConfig, TaskType
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from tensorfn import load_config as DiffConfig
import numpy as np
from config.diffconfig import DiffusionConfig, get_model_conf
import torch.distributed as dist
import os, glob, cv2, time, shutil
from models.unet_autoenc import BeatGANsAutoencConfig
from diffusion import create_gaussian_diffusion, make_beta_schedule, ddim_steps
import torchvision.transforms as transforms
import torchvision

if __name__ == "__main__":
    conf = DiffConfig(DiffusionConfig, './config/diffusion.conf', show=False)

    model = get_model_conf().make_model()
    ckpt = torch.load("checkpoints/last.pt")
    model.load_state_dict(ckpt["ema"])
    model = model.cuda()
    model.eval()



    peft_config=LoraConfig(target_modules=['output_blocks.1.0.cond_emb_layers.1', 'encoder.input_blocks.17.0.emb_layers.1'])

    model=get_peft_model(model, peft_config)

    import train_LoRA as tr
    # added by yehui
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    os.environ["WANDB_API_KEY"] = '093f8148cd98c7bf349917fdccb11942e3b292e2' 
    os.environ["WANDB_MODE"] = "offline"
    os.environ["RANK"] = "0"
    os.environ["local_rank"] = "0"
    os.environ['TORCH_DISTRIBUTED_ELASTIC_LOG_REDIRECT'] = 'FALSE'



    import argparse

    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--exp_name', type=str, default='pidm_deepfashion')
    parser.add_argument('--DiffConfigPath', type=str, default='./config/diffusion.conf')
    parser.add_argument('--DataConfigPath', type=str, default='./config/data.yaml')
    parser.add_argument('--dataset_path', type=str, default='./dataset/deepfashion')
    parser.add_argument('--save_path', type=str, default='checkpoints')
    parser.add_argument('--cond_scale', type=int, default=2)
    parser.add_argument('--guidance_prob', type=int, default=0.1)
    parser.add_argument('--sample_algorithm', type=str, default='ddim') # ddpm, ddim
    parser.add_argument('--batch_size', type=int, default=2)#训练的时候会除以2，不知道为什么
    parser.add_argument('--save_wandb_logs_every_iters', type=int, default=50)
    parser.add_argument('--save_checkpoints_every_iters', type=int, default=2000)
    parser.add_argument('--save_wandb_images_every_epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--n_machine', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args(args=['--exp_name','pidm_LoRA'])

    print ('Experiment: '+ args.exp_name)
    DiffConf = tr.DiffConfig(DiffusionConfig,  args.DiffConfigPath, args.opts, False)
    DataConf = tr.DataConfig(args.DataConfigPath)


    DiffConf.training.ckpt_path = os.path.join(args.save_path, args.exp_name)
    DataConf.data.path = args.dataset_path


    if tr.is_main_process():

        if not os.path.isdir(args.save_path): os.mkdir(args.save_path)
        if not os.path.isdir(DiffConf.training.ckpt_path): os.mkdir(DiffConf.training.ckpt_path)

    #DiffConf.ckpt = "checkpoints/last.pt"

    tr.main(model=model,settings = [args, DiffConf, DataConf], EXP_NAME = args.exp_name)