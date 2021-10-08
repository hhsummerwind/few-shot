import unittest

import torch

from torch.utils.data import DataLoader

import sys
sys.path.append('/projects/open_sources/classify/few-shot')

from few_shot.core import NShotTaskSampler
from few_shot.datasets import DummyDataset, OmniglotDataset, MiniImageNet
from few_shot.models import get_few_shot_encoder
from few_shot.proto import compute_prototypes


model_path = 'models/proto_nets/omniglot_nt=1_kt=60_qt=5_nv=1_kv=5_qv=1.pth'


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
        import pdb
        import math
        import numpy as np
        # Check output of encoder has shape specified in paper
        encoder = get_few_shot_encoder(num_input_channels=1).float()


        encoder.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))




        omniglot = OmniglotDataset('background')
        # pdb.set_trace()
        result1 = []
        for i in range(20):
            result1.append(encoder(omniglot[i][0].unsqueeze(0).float()))
        result2 = []
        for i in range(20):
            result2.append(encoder(omniglot[i + 20][0].unsqueeze(0).float()))
        result3 = []
        for i in range(20):
            result3.append(encoder(omniglot[i + 40][0].unsqueeze(0).float()))
        # for i in range(19):
        #     for j in range(i + 1, 20):
        #         print(np.sum(np.power((result1[i] - result1[j]).detach().numpy(), 2)))
        for i in range(20):
            for j in range(20):
                print(np.sum(np.power((result1[i] - result3[j]).detach().numpy(), 2)))

        # a = encoder(omniglot[0][0].unsqueeze(0).float())
        # b = encoder(omniglot[4][0].unsqueeze(0).float())
        # print((a - b).detach().numpy())
        # print(np.sum(np.power((a - b).detach().numpy(), 2)))
        # print(b)




        self.assertEqual(
            encoder(omniglot[0][0].unsqueeze(0).float()).shape[1],
            64,
            'Encoder network should produce 64 dimensional embeddings on Omniglot dataset.'
        )

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
    t.test_compute_prototypes()