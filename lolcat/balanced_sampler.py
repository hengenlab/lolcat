from typing import Iterator, Sequence

import torch
from torch.utils.data.sampler import Sampler
from torch_geometric.nn import global_mean_pool


class MySampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, labels, num_samples=None) -> None:
        self.labels = labels
        self.num_samples = num_samples

        # get inverse map for labels
        num_classes = torch.max(self.labels) + 1
        self.class_idx = []
        for i in range(num_classes):
            self.class_idx.append(torch.where(self.labels == i)[0])

        # init factors
        self.factors = self.init_oversampling_factor()
        self.indices = self.oversample(self.factors)

        self._step = 0

    def init_oversampling_factor(self):
        _, counts = torch.unique(self.labels, return_counts=True)
        majority_class_count = torch.max(counts)

        oversampling_factors = majority_class_count / counts
        oversampling_factors = torch.clip(oversampling_factors, 1., 200.)
        return oversampling_factors

    def oversample(self, factors):
        indices = []
        for i in range(len(self.class_idx)):
            class_idx = self.class_idx[i]
            num_samples = class_idx.size(0)

            class_factor = factors[i]
            class_factor_n, class_factor_f = int(class_factor), class_factor % 1.

            indices.append(torch.repeat_interleave(class_idx, class_factor_n))
            indices.append(class_idx[torch.randperm(num_samples)[:int(num_samples*class_factor_f)]])
        return torch.cat(indices)

    def __iter__(self) -> Iterator[int]:
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self) -> int:
        return self.num_samples if self.num_samples is not None else len(self.indices)

    def step(self, train_loss, train_labels, val_loss, val_labels):
        self._step += 1
        # compute class stats
        avg_train_loss = global_mean_pool(train_loss, train_labels)
        global_std_train_loss, global_avg_train_loss = torch.std_mean(train_loss)
        avg_val_loss = global_mean_pool(val_loss, val_labels)

        # readjust factors
        for i in range(self.factors.size(0)):
            undertrained_score = (avg_train_loss[i] - global_avg_train_loss) / global_std_train_loss
            overfitting_score = avg_val_loss[i] - avg_train_loss[i]

            # print(i, ' - ', 'global_avg_train_loss:', global_avg_train_loss,  'global_std_train_loss:', global_std_train_loss,
            #      'loss:', avg_train_loss[i], 'undertrained_score:', undertrained_score, 'overfit:', overfitting_score, 'factor:', self.factors[i])

            if undertrained_score > 0.:
                if overfitting_score < 2 * global_std_train_loss:
                    self.factors[i] = self.factors[i] / 0.99
            elif undertrained_score < 0.:
                self.factors[i] = self.factors[i] / 1.01

        if self._step % 50 == 0:
            # random exploration
            # self.factors = self.factors * (0.5 + torch.rand_like(self.factors))
            pass

        self.factors = torch.clip(self.factors, 0.8, 100.)
        self.indices = self.oversample(self.factors)
