import sys
import os
import re
import time
import csv
from enum import Enum, IntEnum
from typing import List

import random
import logging
import argparse
import matplotlib

XRTM_CSV_PTS = 'pts'
XRTM_CSV_POC = 'poc'
XRTM_CSV_SLICETYPE = 'slice_type'
XRTM_CSV_QP = 'qp'
XRTM_CSV_BITS = 'bits'
XRTM_CSV_PSNR = 'psnr'
XRTM_CSV_SSIM = 'ssim'
XRTM_CSV_REFS = 'refs'
XRTM_CSV_TOTAL_FRAME_TIME = 'total_frame_time'
XRTM_CSV_CU_INTRA = 'ctu_intra_pct'
XRTM_CSV_CU_INTER = 'ctu_inter_pct'
XRTM_CSV_CU_SKIP = 'ctu_skip_pct'
XRTM_CSV_CU_MERGE = 'ctu_merge_pct'
XRTM_CSV_RF = 'rate_factor'

I_SLICE = 'I-SLICE'
P_SLICE = 'P-SLICE'

CTU_SIZE = 64
INTRA_CTU_VARIANCE = 0.1
INTER_CTU_VARIANCE = 0.2

global logger


class EncoderConfigException(Exception):
    pass

class VTraceException(Exception):
    pass

class RateControlException(Exception):
    pass

class FeedbackException(Exception):
    pass


class VTrace:
    pts:int = 0
    poc:int = 0
    slice_type:str = ''
    qp:float = .0
    bits:int = 0
    psnr:float = .0
    ssim:float = .0
    total_frame_time:int = 0
    latency:int = 0
    refs:List[List[int]] = []
    ctu_intra_pct:float = .0
    ctu_inter_pct:float = .0
    ctu_skip_pct:float = .0
    ctu_merge_pct:float = .0
    rate_factor:float = .0

    @staticmethod
    def csv_fields():
        return [
            XRTM_CSV_PTS,
            XRTM_CSV_POC,
            XRTM_CSV_SLICETYPE,
            XRTM_CSV_QP,
            XRTM_CSV_BITS,
            XRTM_CSV_PSNR,
            XRTM_CSV_SSIM,
            XRTM_CSV_REFS,
            XRTM_CSV_TOTAL_FRAME_TIME,
            XRTM_CSV_RF,
            XRTM_CSV_CU_INTRA,
            XRTM_CSV_CU_INTER,
            XRTM_CSV_CU_SKIP,
            XRTM_CSV_CU_MERGE
        ]

    def to_csv_dict(self) -> dict:
        return self.__dict__

    def from_csv_dict(self, row:dict):
        for (k, v) in row.items():
            C = type(self.__getattribute__(k))
            self.__setattr__(k, C(v))
        return self


def validates_ctu_distribution(vt:VTrace, raise_exception=True) -> bool:
    if vt.slice_type == I_SLICE:
        total = vt.ctu_intra_pct
    else:
        total = vt.ctu_intra_pct + vt.ctu_inter_pct + vt.ctu_skip_pct + vt.ctu_merge_pct
    valid = round(total * 100) == 100
    if not valid:
        e_msg = f'Invalid CTU distribution on frame {vt.poc} - sum of all CTUs is {total}%'
        if raise_exception:
            raise VTraceException(e_msg)
        logger.critical(e_msg)
    return valid


class STrace:

    pts:int
    poc:int
    bits:int
    qp:float
    qp_new:float
    slice_type:int

    intra_ctu_count:int
    intra_ctu_bits:int
    inter_ctu_count:int
    inter_ctu_bits:int
    skip_ctu_count:int
    skip_ctu_bits:int
    merge_ctu_count:int
    skip_ctu_bits:int
    
    intra_mean:int
    inter_mean:int

    @staticmethod
    def csv_fields():
        return [
            'pts',
            'poc',
            'slice_type',
            'bits',
            'qp',
            'qp_new',
            'intra_ctu_count',
            'intra_ctu_bits',
            'inter_ctu_count',
            'inter_ctu_bits',
            'skip_ctu_count',
            'skip_ctu_bits',
            'merge_ctu_count',
            'merge_ctu_bits',
            'intra_mean',
            'inter_mean'
        ]

    def to_csv_dict(self) -> dict:
        # custom formating here as needed
        return self.__dict__


class RATE_CONTROL(IntEnum):
    CONSTANT = 0
    DYNAMIC = 1
    CRF = 2

class ERROR_RESILIENCE_MODE(IntEnum):
    PERIODIC_FRAME = 1
    PERIODIC_SLICE = 2
    FEEDBACK_BASED = 3

class FEEDBACK_MODE(IntEnum):
    INTRA_REFRESH = 1
    ACK = 2
    NACK = 3

class EncoderConfig:
    # frame config
    frame_width:int = 2048
    frame_height:int = 2048
    slices_per_frame:int = 8
    # rc config
    crf:int # None => will not apply CRF adjustment
    # error resilience
    error_resilience_mode:ERROR_RESILIENCE_MODE = ERROR_RESILIENCE_MODE.PERIODIC_FRAME
    gop_size = -1
    feedback_mode:FEEDBACK_MODE = None
    
class Feedback:
    rc_max_bits:int
    frame_idx:int
    slice_idx:int

class FeedbackStatus:
    """
    a class to report/retrieve feedback,
    this is a draft / work in progress
    it will need to be designed/used carefully so that client/encoder do not block each other
    """
    _status:List[Feedback] = []

    def add(payload:Feedback):
        # a method for incoming feedback
        pass

    def remove(*args, **kwargs):
        # a method to cleanup when no longer referenced by encoder
        pass

    def needs_refresh(self, frame_idx, slice_idx):
        # a method to retrieve the feedback
        intra_refresh = random.randint(0,10) < 3
        return intra_refresh


class CTUEncodingMode(IntEnum):
    INTRA = 1
    INTER = 2
    MERGE = 3
    SKIP = 4

class CTU:
    _mode:CTUEncodingMode
    _refs:List[int]
    
    def __init__(self, mode, refs=None):
        self._mode = mode
        self._refs = refs
    
    @property
    def mode(self):
        return self._mode

    
class SliceStats:
    intra_ctu_count = 0
    intra_ctu_bits = 0
    inter_ctu_count = 0
    inter_ctu_bits = 0
    skip_ctu_count = 0
    skip_ctu_bits = 0
    merge_ctu_count = 0
    merge_ctu_bits = 0

    def add_intra_ctu(self, bits):
        self.intra_ctu_count += 1
        self.intra_ctu_bits += bits

    def add_inter_ctu(self, bits):
        self.inter_ctu_count += 1
        self.inter_ctu_bits += bits

    def add_skip_ctu(self):
        self.skip_ctu_count += 1
        self.skip_ctu_bits += 8

    def add_merge_ctu(self, bits):
        self.merge_ctu_count += 1
        self.merge_ctu_bits += bits
    
    @property
    def total_size(self):
        return self.intra_ctu_bits + self.inter_ctu_bits + self.skip_ctu_bits + self.merge_ctu_bits

    @property
    def ctu_count(self):
        return self.intra_ctu_count + self.inter_ctu_count + self.skip_ctu_count + self.merge_ctu_count




def draw_ctus(intra:float, inter:float, skip:float, merge:float, size:int=4096, refs=[]):
    return random.choices([
        CTU(CTUEncodingMode.INTRA, None),
        CTU(CTUEncodingMode.INTER, refs),
        CTU(CTUEncodingMode.SKIP, None),
        CTU(CTUEncodingMode.MERGE, refs),
        ], weights=[intra, inter, skip, merge], k=size)

def get_intra_mean(vtrace, vtrace_ctu_count):
    if vtrace.slice_type != I_SLICE:
        raise RateControlException(f'can not compute "intra_mean" for {vtrace.slice_type}. I-SLICE required')
    return vtrace.bits / vtrace_ctu_count

def get_inter_mean(vtrace, vtrace_ctu_count, intra_mean):
    if vtrace.slice_type == I_SLICE:
        return 0.0
    intra_size = vtrace.ctu_intra_pct * vtrace_ctu_count * intra_mean
    skip_size = vtrace.ctu_skip_pct * vtrace_ctu_count
    inter_size = vtrace.bits - intra_size - skip_size
    return inter_size / (vtrace_ctu_count * (1 - vtrace.ctu_intra_pct - vtrace.ctu_skip_pct))

def apply_crf_adjustment(bits, CRF, CRFref, qp):
    y = CRFref - CRF
    final_bits = bits * pow(2, y/6)
    QPnew = qp - y
    return final_bits, QPnew

def ctu_bits(mean, variance):
    # isolates common CTU function adjustments
    return random.gauss(mean, variance)

def encode_intra_slice(intra_mean, ctu_count):
    s = SliceStats()
    for ctu in range(ctu_count):
        s.add_intra_ctu(ctu_bits(intra_mean, INTRA_CTU_VARIANCE))
    return s

def encode_inter_slice_1(intra_mean, inter_mean, ctus):
    s = SliceStats()
    for ctu in ctus:
        if ctu.mode == CTUEncodingMode.INTRA:
            s.add_intra_ctu(
                ctu_bits(intra_mean, INTRA_CTU_VARIANCE)
            )
        elif ctu.mode == CTUEncodingMode.INTER:
            s.add_inter_ctu(
                ctu_bits(inter_mean, INTER_CTU_VARIANCE)
            )
        elif ctu.mode == CTUEncodingMode.MERGE:
            s.add_merge_ctu(
                ctu_bits(inter_mean, INTER_CTU_VARIANCE)
            )
        elif ctu.mode == CTUEncodingMode.SKIP:
            s.add_skip_ctu()
    return s

def encode_inter_slice_X(means, X):
    return ctu_bits(mean * X)

def round_to_int(bits):
    return int(round(bits))

def is_perodic_intra_frame(frame_poc, gop_size, gop_start_poc=0):
    # not optimal to compute this for each frame
    return ((frame_poc - gop_start_poc)) % gop_size == 0

def is_perodic_intra_slice(slice_idx, slices_per_frame, frame_poc, gop_start_poc=0):
    # not optimal to compute this for each slice
    return ((frame_poc - gop_start_poc) % slices_per_frame) == slice_idx

def get_slice_ctus(frame_ctus, slices_per_frame, slice_idx):
    sl = len(frame_ctus) / slices_per_frame
    start = int(slice_idx * sl)
    end = int(start + sl)
    return frame_ctus[start:end]

#########

def validate_encoder_config(cfg):
    err_modes = [
        ERROR_RESILIENCE_MODE.PERIODIC_FRAME,
        ERROR_RESILIENCE_MODE.PERIODIC_SLICE,
        ERROR_RESILIENCE_MODE.FEEDBACK_BASED,
    ]
    if not cfg.error_resilience_mode in err_modes:
        raise EncoderConfigException(f'invalid resilience mode. must be one of {err_modes}')

    if cfg.error_resilience_mode != ERROR_RESILIENCE_MODE.PERIODIC_FRAME and cfg.gop_size != -1:
        raise EncoderConfigException('custom gop is only valid for PERIODIC_FRAME error resilience')

    if cfg.error_resilience_mode == ERROR_RESILIENCE_MODE.PERIODIC_SLICE and cfg.slices_per_frame == 1:
        raise EncoderConfigException('PERIODIC_SLICE error resilience requires multiple slices per frame')

    if (cfg.frame_width % CTU_SIZE) != 0:
        raise EncoderConfigException(f'frame width must be a multiple of {CTU_SIZE}')
    
    if (cfg.frame_height % CTU_SIZE) != 0:
        raise EncoderConfigException(f'frame height must be a multiple of {CTU_SIZE}')

    if ((cfg.frame_height / CTU_SIZE ) % cfg.slices_per_frame) != 0 :
        raise EncoderConfigException(f'multiple of {CTU_SIZE}')



class Encoder:

    _cfg:EncoderConfig = None
    _ctus_per_slices:int
    _ctus_per_frame:int
    
    _feedback:FeedbackStatus = None

    # track mean intra CTU size
    # we can not estimate intra size directly from V-trace for Inter, see below
    _gop_start_poc:int = 0
    _intra_mean:float = 0
    _inter_mean:float = 0
    
    def __init__(self, cfg, feedback_provider:FeedbackStatus=None):
        
        validate_encoder_config(cfg)

        if cfg.error_resilience_mode == ERROR_RESILIENCE_MODE.FEEDBACK_BASED and feedback_provider == None:
            raise EncoderConfigException('Feedback based error resilience requires a feedback provider')

        self._cfg = cfg
        self._feedback = feedback_provider

        w = cfg.frame_width
        h = cfg.frame_height
        self._ctus_per_frame = int(( w * h ) / ( CTU_SIZE * CTU_SIZE ))
        self._ctus_per_slice = int(( h / CTU_SIZE ) * h / ( cfg.slices_per_frame * CTU_SIZE ))
        logger.info(f'frame WxH: {w} x {h}')
        logger.info(f'slice height: {int(h / cfg.slices_per_frame)}')
        logger.info(f'ctus p. frame: {self._ctus_per_frame}')
        logger.info(f'ctus p. slice: {self._ctus_per_slice}')
    
    
    def do_frame(self, vtrace:VTrace):
        
        #############################################################
        # vtrace sanity check
        validates_ctu_distribution(vtrace, raise_exception=True)

        #############################################################
        # draw frame CTUs, shuffled, taking into account weights from V-trace
        frame_bits = 0
        frame_ctus = draw_ctus(
                vtrace.ctu_intra_pct,
                vtrace.ctu_inter_pct,
                vtrace.ctu_skip_pct,
                vtrace.ctu_merge_pct,
                size = self._ctus_per_frame
        )

        ################################################################
        # Compute mean Intra / Inter CU size from V-trace
        if vtrace.slice_type == I_SLICE:
            self._gop_start_poc = vtrace.poc
            self._intra_mean = get_intra_mean(vtrace, self._ctus_per_frame)
            self._inter_mean = 0.0

        else:
            # preserve intra value through the GOP, that may be inaccurate !
            self._inter_mean = get_inter_mean(vtrace, self._ctus_per_frame, self._intra_mean)

        #############################################################
        # slices generator, yields S-traces as we iterate do_frames
        for slice_idx in range(self._cfg.slices_per_frame):

            slice_bits = 0
            slice_type = vtrace.slice_type

            #############################################################
            # Based on error resilience mode, take decision on the Slice type
            if self._cfg.error_resilience_mode == ERROR_RESILIENCE_MODE.PERIODIC_FRAME:
                
                if self._cfg.gop_size != -1:
                    if is_perodic_intra_frame(vtrace.poc, self._cfg.gop_size, self._gop_start_poc):
                        slice_type = I_SLICE
                    else:
                        slice_type = P_SLICE
                        
            elif self._cfg.error_resilience_mode == ERROR_RESILIENCE_MODE.PERIODIC_SLICE:

                if is_perodic_intra_slice(slice_idx, self._cfg.slices_per_frame, vtrace.poc, self._gop_start_poc):
                    slice_type = I_SLICE
                else:
                    slice_type = P_SLICE

            elif self._cfg.error_resilience_mode == ERROR_RESILIENCE_MODE.FEEDBACK_BASED:
                if self._feedback.needs_refresh(vtrace.poc, slice_idx):
                    slice_type = I_SLICE

            #############################################################
            # Now encode the Slice             
            slice_ctus = get_slice_ctus(frame_ctus, self._cfg.slices_per_frame, slice_idx)

            if slice_type == I_SLICE:
                s = encode_intra_slice(self._intra_mean, len(slice_ctus))
                slice_bits = s.total_size

            elif slice_type == P_SLICE:
                s = encode_inter_slice_1(self._intra_mean, self._inter_mean, slice_ctus)
                slice_bits = s.total_size
                """
                @TODO: if NACK reports a lost referenced frame ... 
                if X:
                    option_A = encode_inter_slice_X(self._inter_mean, X)
                    option_B = encode_intra_slice(self._intra_mean, slice_ctus)
                    slice_bits = min(option_A, option_B)
                
                Note: option_A doesn't provide CTU stats for debug !
                """

            if self._cfg.crf != None:
                (final_bits, qp_new) = apply_crf_adjustment(slice_bits, self._cfg.crf, vtrace.rate_factor, vtrace.qp)
                slice_bits = final_bits
            else:
                qp_new = None
            
            strace = STrace()
            strace.pts = vtrace.pts
            strace.poc = vtrace.poc
            strace.slice_type = slice_type
            strace.qp = vtrace.qp
            strace.qp_new = qp_new

            strace.bits = int(round(slice_bits))

            strace.intra_ctu_count = s.intra_ctu_count
            strace.inter_ctu_count = s.inter_ctu_count
            strace.skip_ctu_count = s.skip_ctu_count
            strace.merge_ctu_count = s.merge_ctu_count

            strace.intra_ctu_bits = s.intra_ctu_bits
            strace.inter_ctu_bits = s.inter_ctu_bits
            strace.skip_ctu_bits = s.skip_ctu_bits
            strace.merge_ctu_bits = s.merge_ctu_bits
            
            strace.intra_mean = float(self._intra_mean)
            strace.inter_mean = float(self._inter_mean)

            frame_bits += slice_bits
            # @TODO: add S-trace to the ref list for error resilience
            # @TODO: track refs on S-trace (currently not listed in S-trace model specs)
            
            yield strace



def write_csv_vtraces(vtraces:List[VTrace], csv_filename:str):
    with open(csv_filename, 'w', newline='') as X265_csv_out:
        writer = csv.DictWriter(X265_csv_out, fieldnames=VTrace.csv_fields())
        writer.writeheader()
        for vt in vtraces:
            writer.writerow(
                VTrace.to_csv_dict(vt)
            )

def read_csv_vtraces(csv_filename):
    with open(csv_filename, newline='') as csvfile:
        vtrace_reader = csv.DictReader(csvfile)
        for row in vtrace_reader:
            yield VTrace().from_csv_dict(row)


def plot(vtraces, straces):
    """
    @TODO: 
    - plot from csv files directly ?
    - improve slice count != 1 
    """

    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(5, sharex=True)
    
    def filter(traces, key):
        return [t.__dict__[key] for t in traces]

    vtx = filter(vtraces, 'poc')
    vty = filter(vtraces, 'bits')

    stx = filter(straces, 'poc')
    sty = filter(straces, 'bits')

    axs[0].plot( vtx, vty, label='V-traces', color='green')
    axs[0].plot( stx, sty, label='S-traces', color = 'red')
    axs[0].set_ylabel('bit size')
    axs[0].set_xlabel('poc')
    axs[0].legend()

    intra_bits = filter(straces, 'intra_ctu_bits')
    inter_bits = filter(straces, 'inter_ctu_bits')
    merge_bits = filter(straces, 'merge_ctu_bits')
    skip_bits = filter(straces, 'skip_ctu_bits')

    axs[1].plot( stx, intra_bits, label='intra', color='yellowgreen')
    axs[1].plot( stx, inter_bits, label='inter', color='darkorange')
    axs[1].plot( stx, merge_bits, label='merge', color='chocolate')
    axs[1].plot( stx, skip_bits, label='skip', color='steelblue')
    axs[1].set_ylabel('total CTU bits')
    axs[1].set_xlabel('poc')
    axs[1].legend()

    intra_count = filter(straces, 'intra_ctu_count')
    inter_count = filter(straces, 'inter_ctu_count')
    merge_count = filter(straces, 'merge_ctu_count')
    skip_count = filter(straces, 'skip_ctu_count')

    axs[2].plot( stx, intra_count, label='intra', color='yellowgreen')
    axs[2].plot( stx, inter_count, label='inter', color='darkorange')
    axs[2].plot( stx, merge_count, label='merge', color='chocolate')
    axs[2].plot( stx, skip_count, label='skip', color='steelblue')
    axs[2].set_ylabel('CTU count')
    axs[2].set_xlabel('poc')
    axs[2].legend()

    intra_mean = filter(straces, 'intra_mean')
    inter_mean = filter(straces, 'inter_mean')
    axs[3].plot( stx, intra_mean, label='intra medium value', color='grey')
    axs[3].plot( stx, inter_mean, label='inter medium value', color='teal')
    axs[3].set_ylabel('bits')
    axs[3].set_xlabel('poc')
    axs[3].legend()

    qp = filter(straces, 'qp')
    qp_new = filter(straces, 'qp_new')
    axs[4].plot( stx, qp, label='qp', color='grey')
    axs[4].plot( stx, qp_new, label='qp_new', color='teal')
    axs[4].set_xlabel('poc')
    axs[4].legend()
    plt.show()

        
def main(cfg, vtrace_csv, strace_csv, plot_stats=False):

    feedback_provider = None
    if cfg.error_resilience_mode == ERROR_RESILIENCE_MODE.FEEDBACK_BASED:
        feedback_provider = FeedbackStatus()
    encoder = Encoder(cfg, feedback_provider=feedback_provider)
    vtraces = []
    straces = []

    with open(csv_out, 'w', newline='') as strace_csv:

        writer = csv.DictWriter(strace_csv, fieldnames=STrace.csv_fields())
        writer.writeheader()
        
        for vt in read_csv_vtraces(vtrace_csv):
            vtraces.append(vt)
            
            for st in encoder.do_frame(vt):
                writer.writerow(
                    STrace.to_csv_dict(st)
                )
                straces.append(st)
                
    if plot_stats:
        plot(vtraces, straces)


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
                         help='default=1 - Error Resilience Mode: \
                            PERIODIC_FRAME = 1, \
                            PERIODIC_SLICE = 2, \
                            FEEDBACK_BASED = 3' , default=1, required=False)

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
    cfg.crf = args.crf

    # error resilience
    cfg.error_resilience_mode = args.erm
    
    if cfg.error_resilience_mode == ERROR_RESILIENCE_MODE.PERIODIC_FRAME:
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