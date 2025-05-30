import random
from typing import Iterator, List
from torch.utils.data.dataloader import default_collate
from main import pad_spans_tensors
from torch.utils.data import Sampler


class SampleCollator:
    def __init__(self, tokenizer: object):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[dict]):
        outputs = pad_spans_tensors(batch, self.tokenizer)
        return default_collate(outputs)


class MultiDatasetSampler(Sampler[int]):
    def __init__(
        self,
        num_samples: List[int],
        batch_size: int,
        train_shuffle: bool,
        datset_batch_size: int = None,
        limit_num_samples: bool = False,
    ) -> None:
        self.num_samples = num_samples
        self.min_num_samples = min(num_samples)
        self.limit_num_samples = limit_num_samples
        self.batch_size = batch_size if datset_batch_size is None else datset_batch_size
        self.train_shuffle = train_shuffle
        if self.limit_num_samples:
            num_samples_batch = [self.min_num_samples // self.batch_size for _ in self.num_samples]
        else:
            num_samples_batch = [num_sample // self.batch_size for num_sample in self.num_samples]
        self.num_samples_batch = sum(num_samples_batch) * self.batch_size

    def __iter__(self) -> Iterator[int]:
        batch_samples_ids = []
        dataset_start = 0
        for num_sample in self.num_samples:
            instance_ids = [i for i in range(dataset_start, dataset_start + num_sample)]
            if self.train_shuffle:
                random.shuffle(instance_ids)
            dataset_start += num_sample
            if self.limit_num_samples:
                instance_ids = instance_ids[: self.min_num_samples]
                num_sample = self.min_num_samples
            for i in range(num_sample // self.batch_size):
                batch_samples_ids.append(
                    instance_ids[i * self.batch_size : (i + 1) * self.batch_size]
                )
        if self.train_shuffle:
            random.shuffle(batch_samples_ids)
        samples_ids = [sample_id for batch in batch_samples_ids for sample_id in batch]
        return iter(samples_ids)

    def __len__(self) -> int:
        return self.num_samples_batch
