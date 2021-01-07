#!/usr/bin/python3
import csv
from typing import List
import random
import logging
import argparse
from pathlib import Path
import re

from xrtm.utils import (
    read_csv_vtraces,
    plot
)

from xrtm.models import (
    VTraceTx,
    STraceTx
)

from xrtm.exceptions import (
    EncoderConfigException
)

from xrtm.feedback import (
    RandomFeedbackGenerator,
    RandomStereoFeedback
)

from xrtm.encoder import (
    MonoEncoder,
    StereoEncoder,
    EncoderConfig,
    ErrorResilienceMode
)

logger = logging.getLogger(__name__)

def stereo_encoder(cfg):
    feedback_provider = None
    if cfg.error_resilience_mode >= ErrorResilienceMode.FEEDBACK_BASED:
        feedback_provider = RandomStereoFeedback(
            full_intra_ratio=0.1,
            referenceable_ratio=0.05,
            referenceable_default=cfg.error_resilience_mode != ErrorResilienceMode.FEEDBACK_BASED_ACK
        )
    return StereoEncoder(cfg, feedback=feedback_provider)

def mono_encoder(cfg):
    feedback_provider = None
    if cfg.error_resilience_mode >= ErrorResilienceMode.FEEDBACK_BASED:
        feedback_provider = RandomFeedbackGenerator(
            full_intra_ratio=0.1,
            referenceable_ratio=0.05,
            referenceable_default=cfg.error_resilience_mode != ErrorResilienceMode.FEEDBACK_BASED_ACK
        )
    return MonoEncoder(cfg, feedback_provider=feedback_provider)

def main(cfg, vtrace_in:Path, strace_out:Path, stereo=True):

    encoder = stereo_encoder(cfg) if stereo else mono_encoder(cfg)
    
    with open(strace_out, 'w', newline='') as f:
        writer = STraceTx.get_csv_writer(f)
        vtraces = VTraceTx.iter_csv_file(vtrace_in)
        for s in encoder.process(vtraces, getattr(cfg,'num_frames_to_process', 5) ):
            writer.writerow(s.get_csv_dict())

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='STrace encoder configuration')
    parser.add_argument('-v', '--v_trace', type=str, help='v-trace *.csv', required=True )
    parser.add_argument('-s', '--s_trace', help='default to *.strace.csv', type=str, required=False )
    parser.add_argument('-c', '--config', help='encoder config', type=str, required=True )
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
    assert cfg_path.exists(), 'config file not found'
    cfg = EncoderConfig.load(cfg_path)

    vtrace_in = Path(args.v_trace)
    assert vtrace_in.exists(), 'vtrace file not found'

    if args.s_trace == None:
        basename = re.sub('\.vtrace$', '', vtrace_in.stem)
        strace_out = vtrace_in.parent / f'{basename}.strace.csv'
        cfg.frames_dir = vtrace_in.parent / f'{basename}_frames'
    else:
        strace_out = Path(args.s_trace)
        cfg.frames_dir = strace_out.parent / f'{strace_out.stem}_frames'

    cfg.frames_dir.mkdir(parents=True, exist_ok=True)
    main(cfg, vtrace_in, strace_out, stereo=True)
