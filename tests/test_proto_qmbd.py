import unittest

import torch

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

import sys
sys.path.append('/projects/open_sources/classify/few-shot')

from few_shot.core import NShotTaskSampler
from few_shot.datasets import DummyDataset, OmniglotDataset, MiniImageNet, QMBDDataset
from few_shot.models import get_few_shot_encoder
from few_shot.proto import compute_prototypes
from utils.qmbd import alignCollate
import pdb

# model_path = 'models/proto_nets/omniglot_nt=1_kt=60_qt=5_nv=1_kv=5_qv=1.pth'
model_path = 'models/proto_nets/qmbd_nt=1_kt=5_qt=1_nv=1_kv=2_qv=1.pth'

class TestProtoNets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = DummyDataset(samples_per_class=1000, n_classes=20)

    def _test_n_k_q_combination(self, n, k, q):
        n_shot_taskloader = DataLoader(self.dataset,
                                       batch_sampler=NShotTaskSampler(self.dataset, 100, n, k, q))

        # Load a single n-shot, k-way task
        for batch in n_shot_taskloader:
            x, y = batch
            break

        support = x[:n * k]
        support_labels = y[:n * k]
        prototypes = compute_prototypes(support, k, n)

        # By construction the second feature of samples from the
        # DummyDataset is equal to the label.
        # As class prototypes are constructed from the means of the support
        # set items of a particular class the value of the second feature
        # of the class prototypes should be equal to the label of that class.
        for i in range(k):
            self.assertEqual(
                support_labels[i * n],
                prototypes[i, 1],
                'Prototypes computed incorrectly!'
            )

    def test_compute_prototypes(self):
        test_combinations = [
            (1, 5, 5),
            (5, 5, 5),
            (1, 20, 5),
            (5, 20, 5)
        ]

        for n, k, q in test_combinations:
            self._test_n_k_q_combination(n, k, q)

    def test_create_model(self):
        # Check output of encoder has shape specified in paper
        encoder = get_few_shot_encoder(num_input_channels=1).float()


        encoder.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))




        # omniglot = OmniglotDataset('background')
        data = QMBDDataset('background')
        align_func = alignCollate(imgH=28, imgW=28, padding_value=255)
        result1 = []
        # for i in range(len(data)):
        #     print(data[i][1])
        # pdb.set_trace()
        for i in range(2):
            result1.append(encoder(align_func([data[i]])[0]))
        result2 = []
        for i in range(7):
            result2.append(encoder(align_func([data[i + 2]])[0]))
        result3 = []
        for i in range(29):
            result3.append(encoder(align_func([data[i + 9]])[0]))
        for i in range(2):
            for j in range(7):
                print(np.sum(np.power((result1[i] - result2[j]).detach().numpy(), 2)))
        print("-------------------")
        for i in range(6):
            for j in range(i + 1, 7):
                print(np.sum(np.power((result2[i] - result2[j]).detach().numpy(), 2)))

        # a = encoder(omniglot[0][0].unsqueeze(0).float())
        # b = encoder(omniglot[4][0].unsqueeze(0).float())
        # print((a - b).detach().numpy())
        # print(np.sum(np.power((a - b).detach().numpy(), 2)))
        # print(b)




        # self.assertEqual(
        #     encoder(data[0][0].unsqueeze(0).float()).shape[1],
        #     64,
        #     'Encoder network should produce 64 dimensional embeddings on Omniglot dataset.'
        # )

        # encoder = get_few_shot_encoder(num_input_channels=3).float()
        # omniglot = MiniImageNet('background')
        # self.assertEqual(
        #     encoder(omniglot[0][0].unsqueeze(0).float()).shape[1],
        #     1600,
        #     'Encoder network should produce 1600 dimensional embeddings on miniImageNet dataset.'
        # )


if __name__ == '__main__':
    t = TestProtoNets()
    t.setUpClass()
    t.test_create_model()
    # t.test_compute_prototypes()