"""
Reproduce Omniglot results of Snell et al Prototypical networks.
"""
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse
import torch
import torchvision.transforms as transforms

from few_shot.datasets import OmniglotDataset, MiniImageNet, QMBDDataset
from few_shot.models import get_few_shot_encoder
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.proto import proto_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH

from PIL import Image

setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
import pdb

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='qmbd')
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-train', default=1, type=int)
parser.add_argument('--n-test', default=1, type=int)
parser.add_argument('--k-train', default=5, type=int)
parser.add_argument('--k-test', default=2, type=int)
parser.add_argument('--q-train', default=5, type=int)
parser.add_argument('--q-test', default=1, type=int)
args = parser.parse_args()

evaluation_episodes = 1000
episodes_per_epoch = 100

if args.dataset == 'omniglot':
    n_epochs = 40
    dataset_class = OmniglotDataset
    num_input_channels = 1
    drop_lr_every = 20
elif args.dataset == 'miniImageNet':
    n_epochs = 80
    dataset_class = MiniImageNet
    num_input_channels = 3
    drop_lr_every = 40
elif args.dataset == 'qmbd':
    n_epochs = 40
    dataset_class = QMBDDataset
    num_input_channels = 1
    drop_lr_every = 20
else:
    raise (ValueError, 'Unsupported dataset')

param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
            f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'

print(param_str)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        # print("hh: size = ", size)
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img, padding_value=None):
        # print("hh: img.size = ", img.size)
        # sys.exit()
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class alignCollate(object):

    def __init__(self, imgH=28, imgW=28, padding_value=255):
        self.imgH = imgH
        self.imgW = imgW
        self.padding_value = padding_value

    def __call__(self, batch):
        # print(batch[0][0].shape, batch[0][1])

        imgH = self.imgH
        imgW = self.imgW

        new_images = []
        new_labels = []
        transform = resizeNormalize((imgW, imgH))
        for image, label in batch:
            image = np.array(image)
            _, h, w = image.shape
            if w >= h:
                new_image = np.ones((w, w), dtype=np.uint8) * self.padding_value
                padding = int((w - h) / 2)
                new_image[padding:padding + h, :] = image[0]
            else:
                new_image = np.ones((h, h), dtype=np.uint8) * self.padding_value
                padding = int((h - w) / 2)
                new_image[:, padding:padding + w] = image[0]
            # print(new_image.shape)
            new_images.append(Image.fromarray(new_image))
            new_labels.append(label)

        out_images = [transform(image, self.padding_value) for image in new_images]
        images = torch.cat([t.unsqueeze(0) for t in out_images], 0)
        labels = torch.tensor(new_labels)
        print(images.shape, labels.shape)
        return [images, labels]


###################
# Create datasets #
###################
background = dataset_class('background')
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, episodes_per_epoch, args.n_train, args.k_train, args.q_train),
    num_workers=4,
    collate_fn=alignCollate(imgH=28, imgW=28, padding_value=255)
)
evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
    num_workers=4,
    collate_fn=alignCollate(imgH=28, imgW=28, padding_value=255)
)

#########
# Model #
#########
model = get_few_shot_encoder(num_input_channels)
model.to(device, dtype=torch.double)
# pdb.set_trace()

############
# Training #
############
print(f'Training Prototypical network on {args.dataset}...')
optimiser = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.NLLLoss().cuda()


def lr_schedule(epoch, lr):
    # Drop lr every 2000 episodes
    if epoch % drop_lr_every == 0:
        return lr / 2
    else:
        return lr


callbacks = [
    EvaluateFewShot(
        eval_fn=proto_net_episode,
        num_tasks=evaluation_episodes,
        n_shot=args.n_test,
        k_way=args.k_test,
        q_queries=args.q_test,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
        distance=args.distance
    ),
    ModelCheckpoint(
        filepath=PATH + f'/models/proto_nets/{param_str}.pth',
        monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc'
    ),
    LearningRateScheduler(schedule=lr_schedule),
    CSVLogger(PATH + f'/logs/proto_nets/{param_str}.csv'),
]

fit(
    model,
    optimiser,
    loss_fn,
    epochs=n_epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=proto_net_episode,
    fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                         'distance': args.distance},
)
