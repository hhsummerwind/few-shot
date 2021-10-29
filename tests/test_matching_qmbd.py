import unittest

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

import sys
sys.path.append('/projects/open_sources/classify/few-shot')

from few_shot.core import NShotTaskSampler
from few_shot.datasets import DummyDataset, OmniglotDataset, MiniImageNet, QMBDDataset
from few_shot.matching import matching_net_predictions
from few_shot.utils import pairwise_distances
from utils.qmbd import resizeNormalize, alignCollate
from few_shot.models import MatchingNetwork

model_path = 'models/matching_nets/qmbd_n=1_k=5_q=1_nv=1_kv=2_qv=1_dist=cosine_fce=True.pth'


class TestMatchingNets(unittest.TestCase):
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

        # Take just dummy label features and a little bit of noise
        # So distances are never 0
        support = x[:n * k, 1:]
        queries = x[n * k:, 1:]
        support += torch.rand_like(support)
        queries += torch.rand_like(queries)

        distances = pairwise_distances(queries, support, 'cosine')

        # Calculate "attention" as softmax over distances
        attention = (-distances).softmax(dim=1).cuda()

        y_pred = matching_net_predictions(attention, n, k, q)

        self.assertEqual(
            y_pred.shape,
            (q * k, k),
            'Matching Network predictions must have shape (q * k, k).'
        )

        y_pred_sum = y_pred.sum(dim=1)
        self.assertTrue(
            torch.all(
                torch.isclose(y_pred_sum, torch.ones_like(y_pred_sum).double())
            ),
            'Matching Network predictions probabilities must sum to 1 for each '
            'query sample.'
        )

    def test_matching_net_predictions(self):
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
        n_test = 1
        k_test = 2
        q_test = 1
        fce = eval(os.path.basename(os.path.splitext(model_path)[0]).split('_')[-1].split('=')[-1])
        num_input_channels = 1
        lstm_layers = 1
        lstm_input_size = 64
        unrolling_steps = 2
        device = torch.device('cuda')

        encoder = MatchingNetwork(n_test, k_test, q_test, fce, num_input_channels,
                        lstm_layers=lstm_layers,
                        lstm_input_size=lstm_input_size,
                        unrolling_steps=unrolling_steps,
                        device=device)


        encoder.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))




        # omniglot = OmniglotDataset('background')
        data = QMBDDataset('background')
        align_func = alignCollate(imgH=28, imgW=28, padding_value=255)
        result1 = []
        # for i in range(len(data)):
        #     print(data[i][1])
        # pdb.set_trace()
        for i in range(2):
            result1.append(encoder.encoder(align_func([data[i]])[0]))
        result2 = []
        for i in range(7):
            result2.append(encoder.encoder(align_func([data[i + 2]])[0]))
        result3 = []
        for i in range(29):
            result3.append(encoder.encoder(align_func([data[i + 9]])[0]))
        for i in range(2):
            for j in range(7):
                print(np.sum(np.power((result1[i] - result2[j]).detach().numpy(), 2)))
        print("-------------------")
        for i in range(6):
            for j in range(i + 1, 7):
                print(np.sum(np.power((result2[i] - result2[j]).detach().numpy(), 2)))

if __name__ == '__main__':
    t = TestMatchingNets()
    t.setUpClass()
    t.test_create_model()
