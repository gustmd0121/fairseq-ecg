# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import torch 
import soundfile as sf
import torch
import torchaudio
from scipy.io import loadmat
from feature_utils import get_path_iterator, dump_feature

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_mfcc_feature")


class MfccFeatureReader(object):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate = 500,
            n_filter = 128,
            n_lfcc = 312
        )

    def read_audio(self, path, ref_len=None):
        tensor = loadmat(path)
        tensor = torch.from_numpy(tensor['val'])
        tensor = tensor.float()
        data_mean = torch.mean(tensor, 1, keepdim=True)
        data_std = torch.std(tensor, 1, keepdim=True)
        tensor -= data_mean
        tensor /= data_std
        
        return tensor

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len) 
        with torch.no_grad():
            # x = x.view(1, -1) #[1, 70400]

            lfccs = self.lfcc(x)
            lfccs = lfccs.transpose(1, 2)
            deltas = torchaudio.functional.compute_deltas(lfccs)
            ddeltas = torchaudio.functional.compute_deltas(deltas)
            concat = torch.cat([lfccs, deltas, ddeltas], dim=1)
            concat = concat.view(-1, concat.shape[2])
            concat = concat.transpose(0,1).contiguous()           
            
            return concat


def main(tsv_dir, split, nshard, rank, feat_dir, sample_rate):
    reader = MfccFeatureReader(sample_rate)
    generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank)
    dump_feature(reader, generator, num, split, nshard, rank, feat_dir)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir")
    parser.add_argument("split")
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("feat_dir")
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()
    logger.info(args)

    main(**vars(args))
