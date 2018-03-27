import torch, queue
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
from .imports import *
from .core import *
import collections,sys,traceback,threading

string_classes = (str, bytes)


def get_tensor(batch, pin):
    if isinstance(batch, (np.ndarray, np.generic)):
        batch = T(batch).contiguous()
        return batch.pin_memory() if pin else batch
    elif isinstance(batch, string_classes): return batch
    elif isinstance(batch, collections.Mapping):
        return {k: get_tensor(sample, pin) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [get_tensor(sample, pin) for sample in batch]
    raise TypeError("batch must contain numbers, dicts or lists; found {}"
                     .format(type(batch)))


class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, pad_idx=0,
                 num_workers=None, pin_memory=False, drop_last=False, pre_pad=True,
                 transpose=False, transpose_y=False):
        self.dataset,self.batch_size,self.num_workers = dataset,batch_size,num_workers
        self.pin_memory,self.drop_last,self.pre_pad = pin_memory,drop_last,pre_pad
        self.transpose,self.transpose_y,self.pad_idx = transpose,transpose_y,pad_idx

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with '
                                 'batch_size, shuffle, sampler, and drop_last')

        if sampler is not None and shuffle:
            raise ValueError('sampler is mutually exclusive with shuffle')

        if batch_sampler is None:
            if sampler is None:
                sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.lazy_loader = False

    def __len__(self): return len(self.batch_sampler)

    def jag_stack(self, b):
        if len(b[0].shape) not in (1,2): return np.stack(b)
        ml = max(len(o) for o in b)
        if min(len(o) for o in b)==ml: return np.stack(b)
        res = np.zeros((len(b), ml), dtype=b[0].dtype) + self.pad_idx
        for i,o in enumerate(b):
            if self.pre_pad: res[i, -len(o):] = o
            else:            res[i,  :len(o)] = o
        return res

    def np_collate(self, batch):
        b = batch[0]
        if isinstance(b, (np.ndarray, np.generic)): return self.jag_stack(batch)
        elif isinstance(b, (int, float)): return np.array(batch)
        elif isinstance(b, string_classes): return batch
        elif isinstance(b, collections.Mapping):
            return {key: self.np_collate([d[key] for d in batch]) for key in b}
        elif isinstance(b, collections.Sequence):
            return [self.np_collate(samples) for samples in zip(*batch)]
        raise TypeError(("batch must contain numbers, dicts or lists; found {}".format(type(b))))

    def get_batch(self, indices):
        res = self.np_collate([self.dataset[i] for i in indices])
        if self.transpose:   res[0] = res[0].T
        if self.transpose_y: res[1] = res[1].T
        return res

    def __iter__(self):
        if self.num_workers==0:
            for batch in map(self.get_batch, iter(self.batch_sampler)):
                yield get_tensor(batch, self.pin_memory)
        else:
            with LazyThreadPoolExecutor(max_workers=self.num_workers) as e:
                if self.lazy_loader:
                    print('Using lazy threadpool- no batch')
                    for batch in e.map(self.get_batch, iter(self.batch_sampler)):
                        yield get_tensor(batch, self.pin_memory)
                else:
                    print('Chunking instead')
                    # avoid py3.6 issue where queue is infinite and can result in memory exhaustion
                    for c in chunk_iter(iter(self.batch_sampler), self.num_workers*10):
                        for batch in e.map(self.get_batch, c): yield get_tensor(batch, self.pin_memory)



import collections
import itertools
import logging
import threading
import time

class LazyThreadPoolExecutor(ThreadPoolExecutor):
    def map(self, fn, *iterables, timeout=None, chunksize=1, prefetch=None):
        """Returns an iterator equivalent to map(fn, iter).
        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: The size of the chunks the iterable will be broken into
                before being passed to a child process. This argument is only
                used by ProcessPoolExecutor; it is ignored by
                ThreadPoolExecutor.
        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.
        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        if timeout is not None:
            end_time = timeout + time.time()
        if prefetch is None:
            prefetch = self._max_workers
        if prefetch < 0:
            raise ValueError("prefetch count may not be negative")

        argsiter = zip(*iterables)

        fs = collections.deque(self.submit(fn, *args) for args in itertools.islice(argsiter, self._max_workers + prefetch))

        # Yield must be hidden in closure so that the futures are submitted
        # before the first iterator value is required.
        def result_iterator():
            nonlocal argsiter
            try:
                while fs:
                    if timeout is None:
                        res = fs[0].result()
                    else:
                        res = fs[0].result(end_time - time.time())

                    # Got a result, future needn't be cancelled
                    del fs[0]

                    # Dispatch next task before yielding to keep
                    # pipeline full
                    if argsiter:
                        try:
                            args = next(argsiter)
                        except StopIteration:
                            argsiter = None
                        else:
                            fs.append(self.submit(fn, *args))

                    yield res
            finally:
                for future in fs:
                    future.cancel()
        return result_iterator()
