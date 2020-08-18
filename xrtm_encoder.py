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

from xrtm_models import (
    VTrace,
    STrace,
    read_csv_vtraces,
    VTraceException,
    plot
)

from xrtm_feedback import (
    FeedbackStatus,
    Feedback,
    FEEDBACK_MODE
)

I_SLICE = 'I-SLICE'
P_SLICE = 'P-SLICE'

CTU_SIZE = 64
INTRA_CTU_VARIANCE = 0.1
INTER_CTU_VARIANCE = 0.2
MAX_REFS = 3

logger = logging.getLogger(__name__)

class EncoderConfigException(Exception):
    pass

class RateControlException(Exception):
    pass

class FeedbackException(Exception):
    pass

def validates_ctu_distribution(vt:VTrace, raise_exception=True) -> bool:
    total = vt.ctu_intra_pct + vt.ctu_inter_pct + vt.ctu_skip_pct + vt.ctu_merge_pct
    valid = round(total * 100) == 100
    if not valid:
        e_msg = f'Invalid CTU distribution on frame {vt.poc} - sum of all CTUs is {total}%'
        if raise_exception:
            raise VTraceException(e_msg)
        logger.critical(e_msg)
    return valid

class ERROR_RESILIENCE_MODE(IntEnum):
    PERIODIC_FRAME = 1
    PERIODIC_SLICE = 2
    FEEDBACK_BASED = 3

class EncoderConfig:
    # frame config
    frame_width:int = 2048
    frame_height:int = 2048
    slices_per_frame:int = 8
    # rc config
    crf:int # None
    # error resilience
    error_resilience_mode:ERROR_RESILIENCE_MODE = ERROR_RESILIENCE_MODE.PERIODIC_FRAME
    gop_size = -1
    feedback_mode:FEEDBACK_MODE = None


class CTUEncodingMode(IntEnum):
    INTRA = 1
    INTER = 2
    MERGE = 3
    SKIP = 4

class CTU:
    _mode:CTUEncodingMode

    def __init__(self, mode, refs=None):
        self._mode = mode
    
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
   return vtrace.intra_total_bits / vtrace_ctu_count

def get_inter_mean(vtrace, vtrace_ctu_count):
    if vtrace.ctu_intra_pct == 1.:
        return 0.
    inter_bits = vtrace.inter_total_bits - (vtrace.ctu_intra_pct * vtrace.intra_total_bits)
    inter_ctu_count = vtrace_ctu_count * (1 - vtrace.ctu_intra_pct - vtrace.ctu_skip_pct)
    if inter_bits < 0:
        logger.critical(f'poc={vtrace.poc} - invalid meam inter CTU bit size - check V-trace data')

    return inter_bits / inter_ctu_count

def apply_crf_adjustment(bits, CRF, CRFref, qp):
    y = CRFref - CRF
    final_bits = bits * pow(2, y/6)
    QPnew = qp - y
    return final_bits, QPnew

def ctu_bits(mean, variance):
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

def encode_inter_slice_X(mean, X):
    return mean * X

def get_slice_ctus(frame_ctus, slices_per_frame, slice_idx):
    sl = len(frame_ctus) / slices_per_frame
    start = int(slice_idx * sl)
    end = int(start + sl)
    return frame_ctus[start:end]

class InvalidSliceDecision(Exception):
     def __init__(self, message, slice_type, slice_stats):
        super().__init__(message)
        self.slice_type = slice_type
        self.slice_stats = slice_stats

def encode_inter_slice(vtrace, intra_mean, inter_mean, slice_ctus:List[CTU], refs:List[int]=None):
    X = 0
    if refs != None and len(refs) > 0:
        X = vtrace.poc - refs[0]
    if X <= 1:
        s = encode_inter_slice_1(intra_mean, inter_mean, slice_ctus)
        slice_bits = s.total_size
        return s.total_size, s
    else:
        inter_size = encode_inter_slice_X(inter_mean, X)
        s = encode_intra_slice(intra_mean, len(slice_ctus))
        if inter_size >= s.total_size:
            raise InvalidSliceDecision(str(I_SLICE), I_SLICE, s)
        else:
            return inter_size, None

def is_perodic_intra_frame(frame_poc, gop_size, gop_start_poc=0):
    return ((frame_poc - gop_start_poc)) % gop_size == 0

def is_perodic_intra_slice(slice_idx, slices_per_frame, frame_poc, gop_start_poc=0):
    return ((frame_poc - gop_start_poc) % slices_per_frame) == slice_idx

def slice_type_decision(cfg:EncoderConfig, feedback_provider:FeedbackStatus, vtrace:VTrace, slice_idx:int, gop_start_idx:int=0):

    if vtrace.poc == 0:
        return I_SLICE, None

    elif cfg.error_resilience_mode == ERROR_RESILIENCE_MODE.PERIODIC_FRAME:
        if is_perodic_intra_frame(vtrace.poc, cfg.gop_size, gop_start_idx):
            return I_SLICE, None
        else:
            return P_SLICE, None
    
    elif cfg.error_resilience_mode == ERROR_RESILIENCE_MODE.PERIODIC_SLICE:
        if is_perodic_intra_slice(slice_idx, cfg.slices_per_frame, vtrace.poc, gop_start_idx):
            return I_SLICE, None
        else:
            return P_SLICE, None
    
    elif cfg.error_resilience_mode == ERROR_RESILIENCE_MODE.FEEDBACK_BASED:
        f = feedback_provider.get_status(slice_idx, vtrace.poc)
        if f != None:
            if f.mode == FEEDBACK_MODE.INTRA_REFRESH:
                return I_SLICE, f
            elif f.mode == FEEDBACK_MODE.NACK:
                return P_SLICE, f
        else:
            return P_SLICE, None

    raise InvalidSliceDecision('no slice type decision', None, None)


def validate_encoder_config(cfg):
    err_modes = [
        ERROR_RESILIENCE_MODE.PERIODIC_FRAME,
        ERROR_RESILIENCE_MODE.PERIODIC_SLICE,
        ERROR_RESILIENCE_MODE.FEEDBACK_BASED,
    ]
    if not cfg.error_resilience_mode in err_modes:
        raise EncoderConfigException(f'invalid resilience mode. must be one of {err_modes}')
    
    if cfg.error_resilience_mode != ERROR_RESILIENCE_MODE.PERIODIC_FRAME.value and cfg.gop_size != -1:
        raise EncoderConfigException('custom gop is only valid for PERIODIC_FRAME error resilience')

    if cfg.error_resilience_mode == ERROR_RESILIENCE_MODE.PERIODIC_FRAME and cfg.gop_size == -1:
        raise EncoderConfigException('PERIODIC_FRAME error resilience requires an explicit gop size, eg. -g=60')

    if cfg.error_resilience_mode == ERROR_RESILIENCE_MODE.PERIODIC_SLICE and cfg.slices_per_frame == 1:
        raise EncoderConfigException('PERIODIC_SLICE error resilience requires multiple slices per frame')

    if (cfg.frame_width % CTU_SIZE) != 0:
        raise EncoderConfigException(f'frame width must be a multiple of {CTU_SIZE}')
    
    if (cfg.frame_height % CTU_SIZE) != 0:
        raise EncoderConfigException(f'frame height must be a multiple of {CTU_SIZE}')

    if ((cfg.frame_height / cfg.slices_per_frame ) % CTU_SIZE) != 0 :
        raise EncoderConfigException(f'({cfg.frame_height}px / {cfg.slices_per_frame} slices) is not a multiple of {CTU_SIZE}')


class Encoder:

    _cfg:EncoderConfig = None
    _ctus_per_slices:int
    _ctus_per_frame:int
    
    _feedback:FeedbackStatus = None
    _gop_start_poc:int = 0

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
        
        validates_ctu_distribution(vtrace, raise_exception=True)

        frame_bits = 0
        frame_ctus = draw_ctus(
                vtrace.ctu_intra_pct,
                vtrace.ctu_inter_pct,
                vtrace.ctu_skip_pct,
                vtrace.ctu_merge_pct,
                size = self._ctus_per_frame
        )

        intra_mean = get_intra_mean(vtrace, self._ctus_per_frame)
        inter_mean = get_inter_mean(vtrace, self._ctus_per_frame)

        for slice_idx in range(self._cfg.slices_per_frame):

            slice_type = None
            slice_bits = 0
            nack_idx = None
            slice_feedback = None
            qp_new = None

            (slice_type, feedback) = slice_type_decision(self._cfg, self._feedback, vtrace, slice_idx, self._gop_start_poc)

            if self._cfg.error_resilience_mode == ERROR_RESILIENCE_MODE.PERIODIC_FRAME and slice_type == I_SLICE:
                self._gop_start_poc = vtrace.poc

            if feedback != None:
                if feedback.mode == FEEDBACK_MODE.INTRA_REFRESH:
                    slice_feedback = 'INTRA_REFRESH'
                elif feedback.mode == FEEDBACK_MODE.NACK:
                    nack_idx = feedback.frame_idx
                    slice_feedback = f'NACK:{nack_idx} X:{vtrace.poc - nack_idx}'
            
            
            slice_ctus = get_slice_ctus(frame_ctus, self._cfg.slices_per_frame, slice_idx)
            if slice_type == I_SLICE:
                s = encode_intra_slice(intra_mean, len(slice_ctus))
                slice_bits = s.total_size

            elif slice_type == P_SLICE:
                try:
                    refs = None if nack_idx == None else [nack_idx]
                    slice_bits, s = encode_inter_slice(vtrace, intra_mean, inter_mean, slice_ctus, refs)
                    if s == None:
                        s = SliceStats()
                except InvalidSliceDecision as e:
                    slice_type = e.slice_type
                    s = e.slice_stats
                    slice_bits = s.total_size
            
            if self._cfg.crf != None:
                qp_ref = vtrace.intra_qp_ref if slice_type == I_SLICE else vtrace.inter_qp_ref
                (slice_bits, qp_new) = apply_crf_adjustment(slice_bits, self._cfg.crf, vtrace.crf_ref, qp_ref)
            
            strace = STrace()
            strace.bits = int(round(slice_bits))
            strace.pts = vtrace.pts
            strace.poc = vtrace.poc
            strace.slice_type = slice_type
            if slice_type == I_SLICE:
                strace.qp = vtrace.intra_qp_ref
            elif slice_type == P_SLICE:
                strace.qp = vtrace.inter_qp_ref
            strace.qp_new = qp_new
            strace.intra_ctu_count = s.intra_ctu_count
            strace.inter_ctu_count = s.inter_ctu_count
            strace.skip_ctu_count = s.skip_ctu_count
            strace.merge_ctu_count = s.merge_ctu_count
            strace.intra_ctu_bits = s.intra_ctu_bits
            strace.inter_ctu_bits = s.inter_ctu_bits
            strace.skip_ctu_bits = s.skip_ctu_bits
            strace.merge_ctu_bits = s.merge_ctu_bits
            strace.intra_mean = float(intra_mean)
            strace.inter_mean = float(inter_mean)
            strace.feedback = slice_feedback

            frame_bits += strace.bits
            yield strace


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
                writer.writerow(st.__dict__)
                straces.append(st)
                
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