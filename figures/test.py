import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import argparse

import torch
import time
import math
from wavecnp2.cnp2 import RegressionANP2 as ANP2
from wavecnp2.cnp2 import RegressionCNP2 as CNP2

from wavecnp2.data import(
    translate_imgs,
    preprocss_img_resolution_sample,
    preprocss_img_resolution_task,
    generate_rand_2d_cnt_mask_convcnp,
    generate_rand_2d_tgt_mask,
    generate_rand_2d_mask_cnp
)
from wavecnp2.utils import device, gaussian_logpdf
from wavecnp2.wave_conv_2d import ConvCNP2, WaveCNP2
from wavecnp2.experiment import (
    report_loss,
    generate_root,
    generate_root_group,
    WorkingDirectory,
    save_checkpoint
)

# image
import torchvision

# image
import torchvision

random_seed = 5
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic=True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_numpy(x):
    """Convert a PyTorch tensor to NumPy."""
    return x.squeeze().detach().cpu().numpy()



#### data prepare

batch_size = 5
r_dim = 128


y_dim = 1
# y_dim = 3
# cnt_num = 100 # 100
# tgt_num = 300

num_training_tasks = 500 # 1000
num_validation_tasks = 200 # 2000
num_test_tasks = 300 # 5000

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),  # Resize images
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor()           # Convert images to PyTorch tensors
])

trainds = torchvision.datasets.ImageFolder(root='./data/xray/',transform=transform)
img_size = 64

train_set, val_set, test_set = torch.utils.data.random_split(trainds, [num_training_tasks, num_validation_tasks, num_test_tasks])

trainldr = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           drop_last=True,
                                           shuffle=True)
valldr = torch.utils.data.DataLoader(dataset=val_set,
                                           batch_size=batch_size,
                                           drop_last=True,
                                           shuffle=True)
testldr = torch.utils.data.DataLoader(dataset=test_set,
                                           batch_size=batch_size,
                                           drop_last=True,
                                           shuffle=False)


### load model

wd = WorkingDirectory(root=generate_root_group("xray", "xray"))

w_model = WaveCNP2(y_dim, r_dim, img_size,
                   TASK_DWT=True,
                   ADAPT=False,
                   level=3,
                   smooth=1.0)

c_model = ConvCNP2(y_dim, r_dim)

cnp_model = CNP2(r_dim)

anp_model = ANP2(r_dim)


load_dict = torch.load('results/pretrained_model/wcnp_model_best.pth.tar', map_location=torch.device('cpu'), weights_only=False)
w_model.load_state_dict(load_dict['state_dict'])
load_dict = torch.load('results/pretrained_model/ccnp_model_best.pth.tar', map_location=torch.device('cpu'), weights_only=False)
c_model.load_state_dict(load_dict['state_dict'])
load_dict = torch.load('results/pretrained_model/cnp_model_best.pth.tar', map_location=torch.device('cpu'), weights_only=False)
cnp_model.load_state_dict(load_dict['state_dict'])
load_dict = torch.load('results/pretrained_model/anp_model_best.pth.tar', map_location=torch.device('cpu'), weights_only=False)
anp_model.load_state_dict(load_dict['state_dict'])


### test model

fig, axs = plt.subplots(6, 1, figsize=(6, 9))

for x, (imgs, lbls) in enumerate(trainldr):
    image_size_1 = imgs.shape[2]
    image_size_2 = imgs.shape[3]


    axs[0].imshow(np.transpose(torchvision.utils.make_grid(imgs, padding=0), (1, 2, 0)))
    # axs[0].text(-25, 15, 'Original images', horizontalalignment='center')
    axs[0].text(0, -2, 'Original images', fontsize=15)


    imgs = translate_imgs(20, imgs)


    axs[1].imshow(np.transpose(torchvision.utils.make_grid(imgs, padding=0), (1, 2, 0)))
    # axs[0].text(-25, 15, 'Original images', horizontalalignment='center')
    axs[1].text(0, -2, 'Translate images', fontsize=15)


    break


for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])


plt.subplots_adjust(wspace=0, hspace=0)
fig.tight_layout()
plt.show()
plt.draw()
# axs.legend(['Observed Data', 'Mean', 'Confidence'])
# fig.savefig('_experiments/figs/2d_new.pdf',bbox_inches='tight')

print('finished!')
