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
    preprocss_img_resolution_sample,
    preprocss_img_resolution_task,
    generate_rand_2d_cnt_mask_convcnp,
    generate_rand_2d_tgt_mask,
    generate_rand_2d_mask_cnp
)
from wavecnp2.utils import device, gaussian_logpdf
from wavecnp2.wave_conv_2d import WaveCNP2, ConvCNP2
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

batch_size = 6
r_dim = 128


y_dim = 1
# y_dim = 3
# cnt_num = 100 # 100
# tgt_num = 300

# Load datasets
num_training_tasks = 6 # 1000
num_validation_tasks = 6 # 2000
num_test_tasks = 6 # 5000


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),  # Resize images
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor()           # Convert images to PyTorch tensors
])

trainds = torchvision.datasets.ImageFolder(root='./data/xray/',transform=transform)
img_size = 64

train_set, val_set, test_set, empty_set = torch.utils.data.random_split(trainds, [num_training_tasks, num_validation_tasks, num_test_tasks, 1000-num_training_tasks-num_validation_tasks-num_test_tasks])

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

wd = WorkingDirectory(root=generate_root_group("nothing", "nothing"))

w_model = WaveCNP2(y_dim, r_dim, img_size,
                   TASK_DWT=True,
                   ADAPT=True)

load_dict = torch.load('results/pretrained_model_mlp/model_best_old.pth.tar', map_location=torch.device('cpu'), weights_only=False)
w_model.load_state_dict(load_dict['state_dict'])


### test model

fig, axs = plt.subplots(6, 1, figsize=(6, 8))

for x, (imgs, lbls) in enumerate(trainldr):
    image_size_1 = imgs.shape[2]
    image_size_2 = imgs.shape[3]

    axs[0].imshow(np.transpose(torchvision.utils.make_grid(imgs, padding=0), (1, 2, 0)))
    # axs[0].text(-25, 15, 'Original images', horizontalalignment='center')
    axs[0].text(0, -2, 'Original images', fontsize=15)

    cnt_num = 200
    cnt_mask_maxtrix = generate_rand_2d_cnt_mask_convcnp(imgs, batch_size, image_size_1, image_size_2,
                                                         cnt_num)

    tgt_mask_maxtrix = torch.ones_like(imgs)
    tgt_num = image_size_1 * image_size_2

    tgt_mask_vec = torch.ones(batch_size * image_size_1 * image_size_2)
    tgt_mask_vec = tgt_mask_vec > 0

    axs[1].imshow(np.transpose(torchvision.utils.make_grid(cnt_mask_maxtrix*imgs, padding=0), (1, 2, 0)))
    # axs[1].text(-25, 15, 'Context set', horizontalalignment='center')
    axs[1].text(0, -2, 'Context set', fontsize=15)

    # count time
    # toc2 = time.perf_counter()
    # print("in epoch: ", x, " ---- generate_rand_2d_mask_cnp use time: %0.4f seconds", toc2 - toc)

    mean_list, std_list = w_model(imgs, cnt_mask_maxtrix, tgt_mask_maxtrix, tgt_mask_vec, cnt_num, tgt_num)

    ### 0
    y_mean = mean_list[0]
    y_mean = (y_mean).reshape(batch_size, 1, image_size_1, image_size_2)

    axs[2].imshow(np.transpose(torchvision.utils.make_grid(y_mean, padding=0), (1, 2, 0)))
    # axs[2].text(-25, 15, 'WaveCNP', horizontalalignment='center')
    axs[2].text(0, -2, 'Level 1', fontsize=15)

    ### 1
    y_mean = mean_list[1]
    y_mean = (y_mean).reshape(batch_size, 1, image_size_1, image_size_2)
    axs[3].imshow(np.transpose(torchvision.utils.make_grid(y_mean, padding=0), (1, 2, 0)))
    # axs[2].text(-25, 15, 'WaveCNP', horizontalalignment='center')
    axs[3].text(0, -2, 'Level 2', fontsize=15)

    ### 2
    y_mean = mean_list[2]
    y_mean = (y_mean).reshape(batch_size, 1, image_size_1, image_size_2)
    axs[4].imshow(np.transpose(torchvision.utils.make_grid(y_mean, padding=0), (1, 2, 0)))
    # axs[2].text(-25, 15, 'WaveCNP', horizontalalignment='center')
    axs[4].text(0, -2, 'Level 3', fontsize=15)

    ### 3
    y_mean = mean_list[3]
    y_mean = (y_mean).reshape(batch_size, 1, image_size_1, image_size_2)
    axs[5].imshow(np.transpose(torchvision.utils.make_grid(y_mean, padding=0), (1, 2, 0)))
    # axs[2].text(-25, 15, 'WaveCNP', horizontalalignment='center')
    axs[5].text(0, -2, 'Level 4', fontsize=15)


    break


for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])


plt.subplots_adjust(wspace=0, hspace=0)
fig.tight_layout()
plt.show()
plt.draw()
# axs.legend(['Observed Data', 'Mean', 'Confidence'])
# fig.savefig('_experiments/figs/2d_mlp_new_xray_new.pdf',bbox_inches='tight')

print('finished!')
