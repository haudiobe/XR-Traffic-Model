#!/usr/bin/python3
import csv
from typing import List
import random
import logging
import argparse
from pathlib import Path
import re

from xrtm.models import (
    VTraceTx,
    STraceTx
)

from xrtm.encoder import (
    EncoderConfig,
    ErrorResilienceMode,
    MultiViewEncoder,
    VTraceIterator
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='STrace encoder configuration')
    parser.add_argument('-c', '--config', help='encoder config', type=str, required=True)
    parser.add_argument('-u', '--user_id', help='user id', type=int, required=False, default=-1)
    parser.add_argument('-l', '--log_level', type=int, default=0, help='default=0 - set log level. 0:CRITICAL, 1:INFO, 2:DEBUG', required=False)
    args = parser.parse_args()

    log_level = {
        0: logging.CRITICAL,
        1: logging.INFO,
        2: logging.DEBUG,
    }
    logging.basicConfig(level=log_level[args.log_level])
    logger = logging.getLogger(__name__)

    cfg_path = Path(args.config)
    assert cfg_path.exists(), f'config file not found: {cfg_path.resolve()}'

    cfg = EncoderConfig.load(cfg_path)
    encoder = MultiViewEncoder(cfg, args.user_id)

    with open(cfg.get_strace_output(args.user_id), 'w', newline='') as f:
        writer = STraceTx.get_csv_writer(f)
        for traces in VTraceIterator(cfg):
            for s in encoder.process(traces, feedback=[]):
                writer.writerow(s.get_csv_dict())
