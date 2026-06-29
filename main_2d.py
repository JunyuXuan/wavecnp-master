import argparse

import numpy as np
import torch
import time
import math
from config_utils import (
    load_dataset_config,
    load_method_config,
    parse_bool,
    resolve_method_config_path,
    save_run_summary,
)
import wavecnp.utils as wave_utils
import wavecnp2.utils as wave2_utils

from typing import Union

from wavecnp2.cnp2 import RegressionANP2 as ANP2
from wavecnp2.cnp2 import RegressionCNP2 as CNP2

from wavecnp2.data import(
    preprocss_img_resolution_sample,
    preprocss_img_resolution_task,
    generate_rand_2d_cnt_mask_convcnp,
    generate_rand_2d_tgt_mask,
    generate_rand_2d_mask_cnp
)
from wavecnp2.utils import gaussian_logpdf
from wavecnp2.wave_conv_2d import ConvCNP2, WaveCNP2
from wavecnp2.experiment import (
    report_loss,
    generate_root,
    generate_root_group,
    WorkingDirectory,
    save_checkpoint
)

from tnp.tnpd2 import TNPD
from tetnp.tnp.networks.mlp import MLP
from tetnp.tnp.models.tnp import TNPDecoder
from tetnp.models.tetnp import TETNP
from tetnp.models.tetnp import TETNPEncoder
from tetnp.networks.teattention_layers import MultiHeadCrossTEAttentionLayer, MultiHeadSelfTEAttentionLayer
from tetnp.tnp.likelihoods.gaussian import NormalLikelihood, HeteroscedasticNormalLikelihood
from data.image import img_to_task, task_to_img
from wavecnp2.liecnp2 import GridLieCNP


from tetnp.networks.tetransformer import (
    TEISTEncoder,
    TEPerceiverEncoder,
    TETNPTransformerEncoder,
)


# image
import torchvision

# Parse arguments given to the script.
parser = argparse.ArgumentParser()
parser.add_argument('--data',
                    choices=['fake',
                             'mnist',
                             'cifar10',
                             'svhn',
                             'celeba',
                             'xray'],
                    default='svhn',
                    help='Data set to train the CNP on. ')
parser.add_argument('--model',
                    choices=['wavecnp2', 'convcnp2', 'cnp2', 'anp2', 'tnp', 'liecnp2', 'tetnp'],
                    default='tetnp',
                    help='Choice of model. ')
parser.add_argument('--reslevel',
                    choices=['task', 'sample', 'NA'],
                    default='task',
                    help='Choice of resolution level. ')
parser.add_argument('--root',
                    default='table',
                    help='Experiment root, which is the directory from which '
                         'the experiment will run. If it is not given, '
                         'a directory will be automatically created.')
parser.add_argument('--train',
                    default=True,
                    type=parse_bool,
                    help='Perform training. If this is not specified, '
                         'the model will be attempted to be loaded from the '
                         'experiment root.')
parser.add_argument('--test',
                    default=True,
                    type=parse_bool,
                    help='Perform testing. If this is not specified, '
                         'the model will be attempted to be loaded from the best state.')
parser.add_argument('--task_dwt',
                    default=True,
                    type=parse_bool,
                    help='Perform task-level Wavelet transform')
parser.add_argument('--adapt',
                    default=True,
                    type=parse_bool,
                    help='Adapt Wavelet transform')
parser.add_argument('--epochs',
                    default=200, #100
                    type=int,
                    help='Number of epochs to train for.')
parser.add_argument('--smooth',
                    default=None,
                    type=float,
                    help='softmax smoothness.')
# parser.add_argument('--gpuid',
#                     default=2, #1
#                     type=int,
#                     help='GPU id [0, 1, 2]')
parser.add_argument('--batch_size',
                    default=16,
                    type=int,
                    help='batch size.')
parser.add_argument('--level',
                    default=None,
                    type=int,
                    help='Wavelet decomposition level.')
parser.add_argument('--learning_rate',
                    default=1e-3,
                    type=float,
                    help='Learning rate.')
parser.add_argument('--weight_decay',
                    default=1e-5,
                    type=float,
                    help='Weight decay.')
parser.add_argument('--seed',
                    default=123,
                    type=int,
                    help='Random seed.')
parser.add_argument('--r_dim',
                    default=None,
                    type=int,
                    help='r_dim.')
parser.add_argument('--config',
                    default=None,
                    help='Optional path to a method JSON config. Defaults to configs/2d/<model>.json.')
parser.add_argument('--device',
                    default='auto',
                    help='Runtime device: auto, cpu, cuda, cuda:0, cuda:1, etc. Defaults to GPU when available.')
parser.add_argument('--num_training_tasks',
                    default=1000,
                    type=int,
                    help='Number of images used for training.')
parser.add_argument('--num_validation_tasks',
                    default=200,
                    type=int,
                    help='Number of images used for validation.')
parser.add_argument('--num_test_tasks',
                    default=5000,
                    type=int,
                    help='Number of images used for testing.')
args = parser.parse_args()
device = wave2_utils.set_device(args.device)
wave_utils.set_device(args.device, verbose=False)
config_path = resolve_method_config_path('configs/2d', args.model, args.config)
method_config = load_method_config('configs/2d', args.model, args.config)
dataset_config = load_dataset_config('configs/2d/datasets', args.data)
model_hparams = dict(dataset_config.get(args.model, {}))
args.level = method_config.get('level', model_hparams.pop('level', 3)) if args.level is None else args.level
args.smooth = method_config.get('smooth', model_hparams.pop('smooth', 1.0)) if args.smooth is None else args.smooth
args.r_dim = method_config.get('r_dim', 128) if args.r_dim is None else args.r_dim
if args.model == 'wavecnp2' and 'wave_gate_init' in method_config:
    model_hparams['wave_gate_init'] = method_config['wave_gate_init']
if args.model == 'liecnp2' and args.batch_size > 1:
    print(
        f'liecnp2 uses memory-intensive neighbourhood tensors; '
        f'reducing batch size from {args.batch_size} to 1.'
    )
    args.batch_size = 1

random_seed = args.seed
torch.manual_seed(random_seed)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
# torch.use_deterministic_algorithms(True)
if device.type == 'cuda':
    torch.backends.cudnn.deterministic=True


y_dim = 1
# y_dim = 3
cnt_num = 100 # 100
tgt_num = 300


# Load working directory.
experiment_name = f'{args.model}-{args.data}-cnt-{cnt_num}-{args.reslevel}-{args.task_dwt}-{args.adapt}'
wd = WorkingDirectory(root=generate_root_group(experiment_name, args.root))


def task_loglik(imgs, model, cnt_num, tgt_num, reslevel):
    batch_size = imgs.shape[0]
    image_size_1 = imgs.shape[2]
    image_size_2 = imgs.shape[3]
    if reslevel == 'task':
        imgs = preprocss_img_resolution_task(imgs, batch_size, image_size_1, image_size_2).to(device)

    batch = img_to_task(
        imgs,
        num_ctx=cnt_num,
        max_num_points=cnt_num + tgt_num,
        target_all=True,
    )
    if args.model == 'tetnp':
        pred = model(batch.xc, batch.yc, batch.xt)
        return torch.mean(pred.log_prob(batch.yt))

    batch.yt = batch.yt
    return model(batch)


def mask_loglik(imgs, model, cnt_num, tgt_num, reslevel):
    batch_size = imgs.shape[0]
    image_size_1 = imgs.shape[2]
    image_size_2 = imgs.shape[3]
    if reslevel == 'task':
        imgs = preprocss_img_resolution_task(
            imgs, batch_size, image_size_1, image_size_2, level=args.level
        ).to(device)
    elif reslevel == 'sample':
        imgs = preprocss_img_resolution_sample(
            imgs, batch_size, image_size_1, image_size_2
        ).to(device)

    cnt_mask_maxtrix = generate_rand_2d_cnt_mask_convcnp(
        imgs, batch_size, image_size_1, image_size_2, cnt_num
    )
    tgt_mask_maxtrix, tgt_mask_vec, tgt_y_vec = generate_rand_2d_tgt_mask(
        imgs, batch_size, image_size_1, image_size_2, tgt_num
    )
    mean, std = model(
        imgs, cnt_mask_maxtrix, tgt_mask_maxtrix, tgt_mask_vec, cnt_num, tgt_num
    )
    return gaussian_logpdf(tgt_y_vec[:], mean[:], std[:], 'mean')


def batch_loglik(imgs, model, cnt_num, tgt_num, reslevel):
    if args.model in {'tnp', 'tetnp'}:
        return task_loglik(imgs, model, cnt_num, tgt_num, reslevel)
    return mask_loglik(imgs, model, cnt_num, tgt_num, reslevel)


def test(dataloader, model, f_loss, batch_size, cnt_num, tgt_num, reslevel, report_freq=None):
    """Compute the test loss."""
    model.eval()
    likelihoods = []
    all_num = cnt_num + tgt_num
    with torch.no_grad():
        step = 0
        for x, (imgs, lbls) in enumerate(dataloader):
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            obj = batch_loglik(imgs, model, cnt_num, tgt_num, reslevel)

            likelihoods.append(obj.item())
            if report_freq:
                avg_ll = np.array(likelihoods).mean()
                report_loss(f_loss, 'Validation', avg_ll, step, report_freq)
            step = step + 1

    avg_ll = np.array(likelihoods).mean()
    std_ll = np.array(likelihoods).std()

    return avg_ll, std_ll


def validate(dataloader, model, f_loss, batch_size, cnt_num, tgt_num, reslevel, report_freq=None):
    """Compute the validation loss."""
    model.eval()
    likelihoods = []
    all_num = cnt_num + tgt_num
    with torch.no_grad():
        step = 0
        for x, (imgs, lbls) in enumerate(dataloader):
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            obj = batch_loglik(imgs, model, cnt_num, tgt_num, reslevel)

            likelihoods.append(obj.item())
            if report_freq:
                avg_ll = np.array(likelihoods).mean()
                report_loss(f_loss, 'Validation', avg_ll, step, report_freq)
            step = step + 1

    avg_ll = np.array(likelihoods).mean()
    std_ll = np.array(likelihoods).std()

    return avg_ll, std_ll


def train(dataloader, model, opt, f_loss, batch_size, cnt_num, tgt_num, reslevel, report_freq):
    """Perform a training epoch."""
    model.train()
    losses = []
    step = 0
    all_num = cnt_num + tgt_num
    for x, (imgs, lbls) in enumerate(dataloader):
        # tic = time.perf_counter()

        imgs = imgs.to(device)
        lbls = lbls.to(device)

        # print(imgs[0])

        obj = -batch_loglik(imgs, model, cnt_num, tgt_num, reslevel)
        # count time
        # toc3 = time.perf_counter()
        # print("in epoch: ", x, " ---- model use time: %0.4f seconds", toc3 - toc2)

        # Optimization
        opt.zero_grad()
        obj.backward()
        opt.step()

        for name, param in model.named_parameters():
            if 'alpha' in name:
                param.data = param.data.clamp(0, 2 * math.pi)

        # count time
        # toc4 = time.perf_counter()
        # print("in epoch: ", x, " ---- Optimization use time: %0.4f seconds", toc4 - toc3)

        # Track training progress
        losses.append(obj.item())
        avg_loss = np.array(losses).mean()
        if report_freq:
            report_loss(f_loss, 'Training', avg_loss, step, report_freq)

        step = step + 1

    return avg_loss



# Load datasets
num_training_tasks = args.num_training_tasks
num_validation_tasks = args.num_validation_tasks
num_test_tasks = args.num_test_tasks

if args.data == 'fake':
    dataset_size = num_training_tasks + num_validation_tasks + num_test_tasks
    trainds = torchvision.datasets.FakeData(
        size=dataset_size,
        image_size=(1, 28, 28),
        num_classes=10,
        transform=torchvision.transforms.ToTensor(),
        random_offset=args.seed,
    )
    img_size = 28
elif args.data == 'mnist':
    trainds = torchvision.datasets.MNIST(root='./data',
                                              train=True,
                                              transform=torchvision.transforms.ToTensor(),
                                              download=True)
    img_size = 28
elif args.data == 'cifar10':
    trainds = torchvision.datasets.CIFAR10(root='./data',
                                              train=True,
                                              transform=torchvision.transforms.Compose([
                                                            torchvision.transforms.Grayscale(num_output_channels=1),
                                                            torchvision.transforms.ToTensor()]),
                                              download=True)
    img_size = 32
elif args.data == 'svhn':
    trainds = torchvision.datasets.SVHN(root='./data',
                                              transform=torchvision.transforms.Compose([
                                                            torchvision.transforms.Grayscale(num_output_channels=1),
                                                            torchvision.transforms.ToTensor()]),
                                              download=True)
    img_size = 32
elif args.data == 'celeba':
    trainds = torchvision.datasets.CelebA(root='./data',
                                              transform=torchvision.transforms.Compose([
                                                            torchvision.transforms.Grayscale(num_output_channels=1),
                                                            torchvision.transforms.Resize((32, 32)),
                                                            torchvision.transforms.ToTensor()]),
                                              download=True)
    img_size = 32
elif args.data == 'xray':
    trainds = torchvision.datasets.ImageFolder(
        root='./data/xray/',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
        ]),
    )
    img_size = 64

# if args.data == 'mnist':
#     # MNIST
#     train_set, val_set, test_set, _ = torch.utils.data.random_split(trainds, [num_training_tasks, num_validation_tasks, num_test_tasks, (60000 -num_training_tasks -num_validation_tasks - num_test_tasks)  ])
# elif args.data == 'cifar10':
#     # CIFAR
#     train_set, val_set, test_set, _ = torch.utils.data.random_split(trainds, [num_training_tasks, num_validation_tasks, num_test_tasks, (50000 -num_training_tasks -num_validation_tasks - num_test_tasks)  ])
# elif args.data == 'svhn':
#     # SVHN
#     train_set, val_set, test_set, _ = torch.utils.data.random_split(trainds, [num_training_tasks, num_validation_tasks, num_test_tasks, (73257 -num_training_tasks -num_validation_tasks - num_test_tasks)])
# elif args.data == 'celeba':
#     # celeba
#     train_set, val_set, test_set, _ = torch.utils.data.random_split(trainds, [num_training_tasks, num_validation_tasks, num_test_tasks, (162770 -num_training_tasks -num_validation_tasks - num_test_tasks)])

remaining_tasks = len(trainds) - num_training_tasks - num_validation_tasks - num_test_tasks
if remaining_tasks < 0:
    raise ValueError(
        f"Dataset {args.data!r} has only {len(trainds)} samples, but "
        f"{num_training_tasks + num_validation_tasks + num_test_tasks} are requested."
    )
train_set, val_set, test_set, _ = torch.utils.data.random_split(
    trainds,
    [num_training_tasks, num_validation_tasks, num_test_tasks, remaining_tasks],
)


trainldr = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=args.batch_size,
                                           drop_last=True,
                                           shuffle=True)
valldr = torch.utils.data.DataLoader(dataset=val_set,
                                           batch_size=args.batch_size,
                                           drop_last=True,
                                           shuffle=True)
testldr = torch.utils.data.DataLoader(dataset=test_set,
                                           batch_size=args.batch_size,
                                           drop_last=True,
                                           shuffle=False)

# Load model.
if args.model == 'wavecnp2':
    model = WaveCNP2(y_dim, args.r_dim, img_size,
                     TASK_DWT=args.task_dwt,
                     ADAPT=args.adapt,
                     level=args.level,
                     smooth=args.smooth,
                     **model_hparams)
elif args.model == 'convcnp2':
    model = ConvCNP2(y_dim, args.r_dim, **model_hparams)
elif args.model == 'cnp2':
    model = CNP2(args.r_dim)
elif args.model == 'anp2':
    model = ANP2(args.r_dim)
elif args.model == 'tnp':
    model = TNPD(dim_x=method_config['dim_x'],
                 dim_y=method_config['dim_y'],
                 d_model=method_config['d_model'],
                 emb_depth=method_config['emb_depth'],
                 dim_feedforward=method_config['dim_feedforward'],
                 nhead=method_config['nhead'],
                 dropout=method_config['dropout'],
                 num_layers=method_config['num_layers'],
                 bound_std=method_config['bound_std'])
elif args.model == 'liecnp2':
    model = GridLieCNP()
elif args.model == 'tetnp':

    attention_config = method_config['attention']
    kernel_config = method_config['kernel_mlp']
    y_encoder_config = method_config['y_encoder']
    z_decoder_config = method_config['z_decoder']

    mhca_layer = MultiHeadCrossTEAttentionLayer(embed_dim=attention_config['embed_dim'], num_heads=attention_config['num_heads'], head_dim=attention_config['head_dim'], kernel=MLP(**kernel_config))
    mhsa_layer = MultiHeadSelfTEAttentionLayer(embed_dim=attention_config['embed_dim'], num_heads=attention_config['num_heads'], head_dim=attention_config['head_dim'], kernel=MLP(**kernel_config))
    mhca_ctoq_layer = MultiHeadCrossTEAttentionLayer(embed_dim=attention_config['embed_dim'], num_heads=attention_config['num_heads'], head_dim=attention_config['head_dim'], kernel=MLP(**kernel_config))
    mhca_qtot_layer = MultiHeadCrossTEAttentionLayer(embed_dim=attention_config['embed_dim'], num_heads=attention_config['num_heads'], head_dim=attention_config['head_dim'], kernel=MLP(**kernel_config))

    mhca_ctoq_layer1 = MultiHeadSelfTEAttentionLayer(embed_dim=attention_config['embed_dim'], num_heads=attention_config['num_heads'], head_dim=attention_config['head_dim'], kernel=MLP(**kernel_config))
    mhca_qtoc_layer1 = MultiHeadCrossTEAttentionLayer(embed_dim=attention_config['embed_dim'], num_heads=attention_config['num_heads'], head_dim=attention_config['head_dim'], kernel=MLP(**kernel_config))
    mhca_qtot_layer1 = MultiHeadCrossTEAttentionLayer(embed_dim=attention_config['embed_dim'], num_heads=attention_config['num_heads'], head_dim=attention_config['head_dim'], kernel=MLP(**kernel_config))

    transformer_encoder = TETNPTransformerEncoder(num_layers=method_config['transformer_layers'],mhca_layer=mhca_layer)

    y_encoder = MLP(**y_encoder_config)

    z_decoder = MLP(**z_decoder_config)

    ec = TETNPEncoder(transformer_encoder, y_encoder)
    dc = TNPDecoder(z_decoder)
    nl = HeteroscedasticNormalLikelihood()
    model = TETNP(ec, dc, nl)
else:
    raise ValueError(f'Unknown model {args.model}.')

model.to(device)


print("number of parameters: ", np.sum([torch.tensor(param.shape).prod() for param in model.parameters()]))
num_parameters = int(np.sum([torch.tensor(param.shape).prod() for param in model.parameters()]))
summary = {
    "method_name": args.model,
    "config_path": str(config_path),
    "config": method_config,
    "dataset_config": dataset_config,
    "parameters": {
        "task": "image",
        "data": args.data,
        "reslevel": args.reslevel,
        "root": args.root,
        "train": args.train,
        "test": args.test,
        "task_dwt": args.task_dwt,
        "adapt": args.adapt,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "level": args.level,
        "smooth": args.smooth,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "r_dim": args.r_dim,
        "cnt_num": cnt_num,
        "tgt_num": tgt_num,
        "num_training_tasks": num_training_tasks,
        "num_validation_tasks": num_validation_tasks,
        "num_test_tasks": num_test_tasks,
        "num_parameters": num_parameters,
    },
    "results": {},
}

# Perform training.
opt = torch.optim.Adam(model.parameters(),
                       args.learning_rate,
                       weight_decay=args.weight_decay)

if args.train:
    with open(wd.file('log.txt'), 'w') as f:
        best_obj = -np.inf
        for epoch in range(args.epochs):
            tic = time.perf_counter()

            # Compute training objective.
            train_obj = train(trainldr, model, opt, f, args.batch_size, cnt_num, tgt_num, args.reslevel, report_freq=None)

            # Compute validation objective.
            val_obj, std_ll = validate(valldr, model, f, args.batch_size, cnt_num, tgt_num, args.reslevel, report_freq=None)

            # Update the best objective value and checkpoint the model.
            is_best = False
            if val_obj > best_obj:
                best_obj = val_obj
                is_best = True
            save_checkpoint(wd,
                            {'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'best_acc_top1': best_obj,
                             'optimizer': opt.state_dict()},
                            is_best=is_best)

            # count time
            toc = time.perf_counter()
            print("epoch ", epoch, "/", args.epochs," use time: %0.4f seconds", toc-tic)

        summary["results"]["best_validation_log_likelihood"] = best_obj
        summary["results"]["final_training_loss"] = train_obj
        summary["results"]["final_validation_log_likelihood"] = val_obj
        summary["results"]["final_validation_std"] = std_ll

if args.test:
    with open(wd.file('test_log_likelihood.txt'), 'w') as f:
        print('\nTest: ')
        # Load saved model.
        load_dict = torch.load(
            wd.file('model_best.pth.tar', exists=True),
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(load_dict['state_dict'])

        # Finally, test model on ~2000 tasks.
        test_obj, std_ll = test(testldr, model, f, args.batch_size, cnt_num, tgt_num, args.reslevel, report_freq=None)
        print('Model averages a log-likelihood of %s on unseen tasks.' % test_obj)
        print('Model std a log-likelihood of %s on unseen tasks.' % std_ll)

        if args.adapt:
            for name, param in model.named_parameters():
                if 'wt_alpha' in name:
                    print(name, "   ", param.data)

        summary["results"]["test_log_likelihood"] = test_obj
        summary["results"]["test_std"] = std_ll
        f.write(f"test_log_likelihood: {test_obj}\n")
        f.write(f"test_std: {std_ll}\n")

save_run_summary(wd.file('summary.json'), summary)
