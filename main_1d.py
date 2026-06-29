import argparse

import numpy as np
# import stheno as stheno
import torch
import time
import math
import gpytorch
from config_utils import load_method_config, parse_bool, resolve_method_config_path, save_run_summary
import wavecnp.utils as wave_utils

from wavecnp.cnp import RegressionANP as ANP
from wavecnp.cnp import RegressionCNP as CNP
# import wavecnp.data
import wavecnp.data_gpy

from wavecnp.utils import gaussian_logpdf
from wavecnp.wave_conv import ConvCNP, WaveCNP, LieCNP
from wavecnp.architectures import SimpleConv, UNet
from wavecnp.experiment import (
    report_loss,
    generate_root,
    generate_root_group,
    WorkingDirectory,
    save_checkpoint
)
from addict import Dict
# from attrdict import AttrDict

from tnp.tnpd import TNPD
from tetnp.tnp.networks.mlp import MLP
from tetnp.tnp.models.tnp import TNPDecoder
from tetnp.models.tetnp import TETNP
from tetnp.models.tetnp import TETNPEncoder
from tetnp.networks.teattention_layers import MultiHeadCrossTEAttentionLayer, MultiHeadSelfTEAttentionLayer
from tetnp.tnp.likelihoods.gaussian import NormalLikelihood, HeteroscedasticNormalLikelihood

from tetnp.networks.tetransformer import (
    TEISTEncoder,
    TEPerceiverEncoder,
    TETNPTransformerEncoder,
)

# Parse arguments given to the script.
parser = argparse.ArgumentParser()
parser.add_argument('--data',
                    choices=['eq',
                             'matern',
                             'noisy-mixture',
                             'weakly-periodic',
                             'linear',
                             'polynomial'],
                    default='matern',
                    help='Data set to train the CNP on. ')
parser.add_argument('--model',
                    choices=['wavecnp', 'convcnp', 'cnp', 'anp', 'tnp', 'liecnp', 'tetnp'],
                    default='tetnp',
                    help='Choice of model. ')
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
parser.add_argument('--ind_dwt',
                    default=False,
                    type=parse_bool,
                    help='Perform sample-level Wavelet transform')
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
# parser.add_argument('--gpuid',
#                     default=1, #1
#                     type=int,
#                     help='GPU id [0, 1, 2]')
parser.add_argument('--learning_rate',
                    default=1e-3,
                    type=float,
                    help='Learning rate.')
parser.add_argument('--weight_decay',
                    default=1e-5,
                    type=float,
                    help='Weight decay.')
parser.add_argument('--seed',
                    default=124,
                    type=int,
                    help='Random seed.')
parser.add_argument('--tnp_feedforward',
                    default=None,
                    type=int,
                    help='Feed-forward dimension for TNP.')
parser.add_argument('--tnp_layers',
                    default=None,
                    type=int,
                    help='Number of transformer layers for TNP.')
parser.add_argument('--config',
                    default=None,
                    help='Optional path to a method JSON config. Defaults to configs/1d/<model>.json.')
parser.add_argument('--device',
                    default='auto',
                    help='Runtime device: auto, cpu, cuda, cuda:0, cuda:1, etc. Defaults to GPU when available.')
parser.add_argument('--num_train_tasks',
                    default=256,
                    type=int,
                    help='Number of generated training tasks per epoch.')
parser.add_argument('--num_val_tasks',
                    default=60,
                    type=int,
                    help='Number of generated validation tasks.')
parser.add_argument('--num_test_tasks',
                    default=2048,
                    type=int,
                    help='Number of generated test tasks.')
args = parser.parse_args()
device = wave_utils.set_device(args.device)
config_path = resolve_method_config_path('configs/1d', args.model, args.config)
method_config = load_method_config('configs/1d', args.model, args.config)
if args.model == 'tnp':
    args.tnp_feedforward = method_config['dim_feedforward'] if args.tnp_feedforward is None else args.tnp_feedforward
    args.tnp_layers = method_config['num_layers'] if args.tnp_layers is None else args.tnp_layers

random_seed = args.seed
torch.manual_seed(random_seed)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
# torch.use_deterministic_algorithms(True)
if device.type == 'cuda':
    torch.backends.cudnn.deterministic=True


# Load working directory.
experiment_name = f'{args.model}-{args.data}-{args.ind_dwt}-{args.task_dwt}-{args.adapt}-{random_seed}-nn'
wd = WorkingDirectory(root=generate_root_group(experiment_name, args.root))


##
def model_loglik(task, model):
    if args.model == 'tetnp':
        pred = model(task['x_context'], task['y_context'], task['x_target'])
        return torch.mean(pred.log_prob(task['y_target']))
    if args.model == 'tnp':
        batch = Dict()
        batch.x_context = task['x_context']
        batch.y_context = task['y_context']
        batch.x_target = task['x_target']
        batch.y_target = task['y_target']
        y_mean, y_std = model(batch)
    else:
        y_mean, y_std = model(task['x_context'], task['y_context'], task['x_target'])
    return gaussian_logpdf(task['y_target'], y_mean, y_std, 'batched_mean')


def validate(data, model, f_loss, report_freq=None):
    """Compute the validation loss."""
    model.eval()
    likelihoods = []
    with torch.no_grad():
        for step, task in enumerate(data):
            obj = model_loglik(task, model)
            likelihoods.append(obj.item() / task['y_target'].shape[1])
            if report_freq:
                avg_ll = np.array(likelihoods).mean()
                # std_ll = np.array(likelihoods).std()
                report_loss(f_loss, 'Validation', avg_ll, step, report_freq)
    avg_ll = np.array(likelihoods).mean()
    std_ll = np.array(likelihoods).std()
    return avg_ll, std_ll


def train(data, model, opt, f_loss, report_freq):
    """Perform a training epoch."""
    model.train()
    losses = []

    tic = time.perf_counter()

    for step, task in enumerate(data):

        # toc1 = time.perf_counter()
        # print(" in train(): ---------- enumerate(data) uses time: ", toc1- tic)

        obj = -model_loglik(task, model)

        # Optimization
        opt.zero_grad()
        obj.backward()
        opt.step()

        for name, param in model.named_parameters():
            if 'alpha' in name:
                param.data = param.data.clamp(0, 2*math.pi)

        # tic = time.perf_counter()
        # print(" in train(): ---------- Optimization uses time: ", tic - toc2)

        # Track training progress
        losses.append(obj.item())
        avg_loss = np.array(losses).mean()
        if report_freq:
            report_loss(f_loss, 'Training', avg_loss, step, report_freq)
    return avg_loss

batch_size = 16

# Load data generator.
if args.data == 'eq':
    kernel = gpytorch.kernels.RBFKernel(batch_shape=torch.Size([batch_size]))
elif args.data == 'matern':
    kernel = gpytorch.kernels.MaternKernel(batch_shape=torch.Size([batch_size]))
elif args.data == 'noisy-mixture':
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(batch_shape=torch.Size([batch_size])) +
                                          gpytorch.kernels.MaternKernel(nu=0.5, batch_shape=torch.Size([batch_size])) +
                                          gpytorch.kernels.MaternKernel(nu=1.5, batch_shape=torch.Size([batch_size])),
                                          batch_shape=torch.Size([batch_size]))
elif args.data == 'weakly-periodic':
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(batch_shape=torch.Size([batch_size]))
                                          * gpytorch.kernels.PeriodicKernel(batch_shape=torch.Size([batch_size])),
                                          batch_shape=torch.Size([batch_size]))
elif args.data == 'linear':
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(batch_shape=torch.Size([batch_size])) +
                                          gpytorch.kernels.LinearKernel(batch_shape=torch.Size([batch_size])),
                                          batch_shape=torch.Size([batch_size]))
elif args.data == 'polynomial':
    kernel = gpytorch.kernels.PolynomialKernel(power=4, batch_shape=torch.Size([batch_size]))
else:
    raise ValueError(f'Unknown data "{args.data}".')

gen = wavecnp.data_gpy.GPGenerator_torch(kernel=kernel, num_tasks=args.num_train_tasks)
# gen = wavecnp.data_gpy.GPGenerator_torch_sample(kernel=kernel).to(device)
# gen = wavecnp.data_gpy.GPGenerator_torch_task(kernel=kernel)

gen_val = wavecnp.data_gpy.GPGenerator_torch(kernel=kernel, num_tasks=args.num_val_tasks)
# gen_val = wavecnp.data_gpy.GPGenerator_torch_sample(kernel=kernel, num_tasks=60).to(device)
# gen_val = wavecnp.data_gpy.GPGenerator_torch_task(kernel=kernel, num_tasks=60)

gen_test = wavecnp.data_gpy.GPGenerator_torch(kernel=kernel, num_tasks=args.num_test_tasks)
# gen_test = wavecnp.data_gpy.GPGenerator_torch_sample(kernel=kernel, num_tasks=2048).to(device)
# gen_test = wavecnp.data_gpy.GPGenerator_torch_task(kernel=kernel, num_tasks=2048)

# gen.to(device)
# gen_val.to(device)
# gen_test.to(device)


# Load model.
if args.model == 'wavecnp':
    model = WaveCNP(learn_length_scale=method_config['learn_length_scale'],
                    points_per_unit=method_config['points_per_unit'],
                    num_points=method_config['num_points'],
                    latent_dim=method_config['latent_dim'],
                    # architecture=SimpleConv(),
                    IND_DWT=args.ind_dwt,
                    TASK_DWT=args.task_dwt,
                    ADAPT=args.adapt,
                    level=method_config.get('level', 3),
                    smooth=method_config.get('smooth', 1.0),
                    wave_gate_init=method_config.get('wave_gate_init', -3.0))
elif args.model == 'convcnp':
    model = ConvCNP(learn_length_scale=method_config['learn_length_scale'],
                        points_per_unit=method_config['points_per_unit'],
                        num_points=method_config['num_points'],
                        latent_dim=method_config['latent_dim'])
elif args.model == 'liecnp':
    model = LieCNP(x_dim=method_config['x_dim'],
                   y_dim=method_config['y_dim'],
                   nbhd=method_config['nbhd'],
                   fill=method_config['fill'])
elif args.model == 'cnp':
    model = CNP(latent_dim=method_config['latent_dim'])
elif args.model == 'anp':
    model = ANP(latent_dim=method_config['latent_dim'])
elif args.model == 'tnp':
    model = TNPD(dim_x=method_config['dim_x'],
                 dim_y=method_config['dim_y'],
                 d_model=method_config['d_model'],
                 emb_depth=method_config['emb_depth'],
                 dim_feedforward=args.tnp_feedforward,
                 nhead=method_config['nhead'],
                 dropout=method_config['dropout'],
                 num_layers=args.tnp_layers)
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

# print(" number of parameters: ", model.num_params)
print("number of parameters: ", np.sum([torch.tensor(param.shape).prod() for param in model.parameters()]))
num_parameters = int(np.sum([torch.tensor(param.shape).prod() for param in model.parameters()]))
summary = {
    "method_name": args.model,
    "config_path": str(config_path),
    "config": method_config,
    "parameters": {
        "task": "regression",
        "data": args.data,
        "root": args.root,
        "train": args.train,
        "test": args.test,
        "ind_dwt": args.ind_dwt,
        "task_dwt": args.task_dwt,
        "adapt": args.adapt,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "tnp_feedforward": args.tnp_feedforward,
        "tnp_layers": args.tnp_layers,
        "num_train_tasks": args.num_train_tasks,
        "num_val_tasks": args.num_val_tasks,
        "num_test_tasks": args.num_test_tasks,
        "num_parameters": num_parameters,
    },
    "results": {},
}

# exit(0)

# Perform training.
opt = torch.optim.Adam(model.parameters(),
                       args.learning_rate,
                       weight_decay=args.weight_decay)

if args.train:
    with open(wd.file('log.txt'), 'w') as f:
        # Run the training loop, maintaining the best objective value.
        best_obj = -np.inf
        for epoch in range(args.epochs):
            print('\nEpoch: {}/{}'.format(epoch + 1, args.epochs))

            tic = time.perf_counter()

            # Compute training objective.
            train_obj = train(gen, model, opt, f, report_freq=None)

            # Compute validation objective.
            val_obj, std_ll = validate(gen_val, model, f, report_freq=None)

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
            print("epoch use time: %0.4f seconds", toc - tic)

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
        test_obj, std_ll = validate(gen_test, model, f, report_freq=None)
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
