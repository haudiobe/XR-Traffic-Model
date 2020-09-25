
import csv
from typing import List
import random
import logging
import argparse


from xrtm.utils import (
    read_csv_vtraces,
    plot
)

from xrtm.models import (
    VTrace,
    STrace
)

from xrtm.exceptions import (
    VTraceException,
    EncoderConfigException
)

from xrtm.feedback import (
    ReferenceableList,
    FeedbackProvider,
    Feedback
)

from xrtm.encoder import (
    Encoder,
    EncoderConfig,
    ErrorResilienceMode
)

logger = logging.getLogger(__name__)

class RandomFeedbackProvider(FeedbackProvider):
    
    def __init__(self, full_intra_ratio:float, referenceable_ratio:float, referenceable_default:bool):
        self.full_intra_ratio = int(full_intra_ratio * 100)
        self.referenceable_ratio = int(referenceable_ratio * 100)
        self.referenceable_default = referenceable_default

    def handle_feedback(self, payload:Feedback):
        raise NotImplementedError()
    
    @property
    def rc_max_bits(self) -> int:
        return -1

    def get_full_intra_refresh(self) -> bool:
        return random.randint(0, 100) < self.full_intra_ratio

    def set_full_intra_refresh(self, idr):
        pass
    
    full_intra_refresh = property(get_full_intra_refresh, set_full_intra_refresh)

    def apply_feedback(self, rpl:ReferenceableList) -> ReferenceableList:
        for rp in rpl.pics:
            for s in rp.slices:
                if s.referenceable == self.referenceable_default and random.randint(0, 100) < self.referenceable_ratio:
                    s.referenceable = not self.referenceable_default
        return rpl


def main(cfg, vtrace_csv, csv_out, plot_stats=False):

    feedback_provider = None
    if cfg.error_resilience_mode >= ErrorResilienceMode.FEEDBACK_BASED:
        feedback_provider = RandomFeedbackProvider(
            full_intra_ratio=0.1, # to disable INTRA_REFRESH, set this to 0 
            referenceable_ratio=0.05, # to disable ACK/NACK, set this to 0 and set default to True
            referenceable_default=cfg.error_resilience_mode != ErrorResilienceMode.FEEDBACK_BASED_ACK
        )
    encoder = Encoder(cfg, feedback_provider=feedback_provider)
    vtraces = []
    straces = []

    with open(csv_out, 'w', newline='') as strace_csv:

        writer = csv.DictWriter(strace_csv, fieldnames=STrace.csv_headers())
        writer.writeheader()
        
        for vt in read_csv_vtraces(vtrace_csv):
            vtraces.append(vt)
            for s in encoder.encode(vt):
                strace = STrace.to_csv_dict(s)
                writer.writerow(strace)
                straces.append(strace)
    
    if plot_stats:
        plot(vtraces, straces, cfg.slices_per_frame)


def parse_args():

    parser = argparse.ArgumentParser(description='Model encoder configuration')

    parser.add_argument('-i', '--v_trace', type=str,
                         help='v-trace *.csv', required=True )

    parser.add_argument('-o', '--s_trace', 
                        help='default to *.strace.csv', type=str, required=False )

    parser.add_argument('-W', '--width', type=int,
                         help='default=2048 - frame width', default=2048)

    parser.add_argument('-H', '--height', type=int,
                         help='default=2048 - frame height', default=2048)

    parser.add_argument('-s', '--slices', type=int,
                         help='default=1 - slices per v-trace, typically 1, 4, 8, 16', default=1)

    parser.add_argument('--crf', type=int, default=None, required=False,
                         help='default=None - tune bitrate with a custom CRF')

    parser.add_argument('-e', '--erm', type=int,
                         help='default=0 - Error Resilience Mode: \
                            DISABLE = 0, \
                            PERIODIC_FRAME = 1, \
                            PERIODIC_SLICE = 2, \
                            FEEDBACK_BASED (NACK) = 3, \
                            FEEDBACK_BASED_ACK = 4' , default=0, required=False)

    parser.add_argument('-g', '--gop_size', type=int,
                         help='default=-1 - intra refresh period when using --erm=1. -1 just follows v-trace GOP pattern' , default=-1)

    parser.add_argument('--plot', action="store_true", default=False, help='run the plot function after generating the output csv')
    
    parser.add_argument('-l', '--log_level', type=int, default=0, help='default=0 - set log level. 0:CRITICAL, 1:INFO, 2:DEBUG', required=False)

    args = parser.parse_args()
    
    cfg = EncoderConfig()
    
    # frame config
    cfg.frame_width = args.width
    cfg.frame_height = args.height
    cfg.slices_per_frame = args.slices
    
    # rc config
    cfg.crf = -1 if args.crf == None else args.crf

    # error resilience
    cfg.error_resilience_mode = args.erm
    cfg.gop_size = args.gop_size
    
    return cfg, args


if __name__ == '__main__':

    cfg, args = parse_args()
    log_level = {
        0: logging.CRITICAL,
        1: logging.INFO,
        2: logging.DEBUG,
    }
    logging.basicConfig(level=log_level[args.log_level])
    logger = logging.getLogger(__name__)

    if args.s_trace == None:
        basename = args.v_trace.split('.')[0]
        csv_out = f'{basename}.strace.csv'
    else:
        csv_out = args.s_trace

    try:
        main(cfg, args.v_trace, csv_out, plot_stats=args.plot)
    except EncoderConfigException as ee:
        logger.critical(ee)
    except FileNotFoundError as fe:
        logger.critical(fe)