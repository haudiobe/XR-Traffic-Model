import sys
import os
import re
import time
import csv
from enum import Enum, IntEnum
from typing import List, Iterator

import random
import logging
import argparse

from .exceptions import (
    VTraceException,
    EncoderConfigException,
    RateControlException,
    FeedbackException
)

from .models import (
    VTrace,
    STrace,
    SliceStats,
    SliceType,
    Slice,
    Frame,
    CTU,
    CTUEncodingMode,
    validates_ctu_distribution
)

from .utils import (
    read_csv_vtraces
)

from .feedback import (
    FeedbackProvider,
    FeedbackType,
    ReferenceableList
)


CTU_SIZE = 64
INTRA_CTU_VARIANCE = 0.1
INTER_CTU_VARIANCE = 0.2
MAX_REFS = 16

logger = logging.getLogger(__name__)


class ErrorResilienceMode(IntEnum):
    DISABLED = 0
    PERIODIC_FRAME = 1
    PERIODIC_SLICE = 2
    FEEDBACK_BASED = 3 # feedback INTRA, BITRATE + NACK
    FEEDBACK_BASED_ACK = 4 # feedback INTRA, BITRATE + ACK

class EncoderConfig:
    # frame config
    frame_width:int = 2048
    frame_height:int = 2048
    slices_per_frame:int = 8
    crf:int = -1
    # error resilience
    error_resilience_mode:ErrorResilienceMode = ErrorResilienceMode.DISABLED
    gop_size = -1

class RefPicList(ReferenceableList):
    max_size:int
    pics:List[Frame]
    
    def __init__(self, max_size:int=16):
        self.max_size = max_size
        self.pics = []

    def add_frame(self, pic:Frame):
        if pic.is_idr_frame:
            self.pics = [pic]
        else:
            self.pics.append(pic)
            self.pics = self.pics[-self.max_size:]

    def slice_refs(self, slice_idx): 
        for frame in reversed(self.pics):
            if frame.slices[slice_idx].referenceable:
                yield frame.poc
            if frame.slices[slice_idx].slice_type == SliceType.IDR:
                return

    def set_referenceable_status(self, frame_idx:int, slice_idx:int, status:bool):
        for frame in reversed(self.pics):
            if frame.poc == frame_idx:
                frame.slices[slice_idx].referenceable = status
                return

    def reset(self):
        self.pics = []

def draw_ctus(intra:float, inter:float, skip:float, merge:float, size:int=4096, refs=[]):
    return random.choices([
        CTU(CTUEncodingMode.INTRA, None),
        CTU(CTUEncodingMode.INTER, refs),
        CTU(CTUEncodingMode.SKIP, None),
        CTU(CTUEncodingMode.MERGE, refs),
        ], weights=[intra, inter, skip, merge], k=size)

def get_slice_ctus(frame_ctus, slices_per_frame, slice_idx):
    sl = len(frame_ctus) / slices_per_frame
    start = int(slice_idx * sl)
    end = int(start + sl)
    return frame_ctus[start:end]

def get_intra_mean(vtrace, vtrace_ctu_count):
   return vtrace.intra_total_bits / vtrace_ctu_count

def get_inter_mean(vtrace, vtrace_ctu_count):
    if vtrace.ctu_intra_pct == 1.:
        return 0.
    intra_mean = vtrace.intra_total_bits / vtrace_ctu_count
    inter_bits = vtrace.inter_total_bits - (vtrace.ctu_intra_pct * intra_mean) - (vtrace.ctu_skip_pct * 8)
    inter_ctu_count = vtrace_ctu_count * (1 - vtrace.ctu_intra_pct - vtrace.ctu_skip_pct)
    return inter_bits / inter_ctu_count

def model_crf_adjustment(bits, CRF, CRFref, qp):
    y = CRFref - CRF
    final_bits = bits * pow(2, y/6)
    QPnew = qp - y
    return final_bits, QPnew

def model_pnsr_adjustment(qp_new, qp_ref, psnr):
    qp_delta = qp_new - qp_ref
    return psnr + qp_delta

def model_encoding_time(s:Slice, v:VTrace, slices_per_frame:int):
    return v.get_encoding_time(s.slice_type) / slices_per_frame

def ctu_bits(mean, variance):
    return random.gauss(mean, variance)

def encode_intra_slice(intra_mean, ctu_count):
    s = SliceStats()
    for _ in range(ctu_count):
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

class SliceTypeOverride(Exception):
     def __init__(self, message:str, t:SliceType, s:SliceStats):
        self.slice_type = t
        self.stats = s
        super().__init__(message)

def encode_inter_slice(frame_idx, intra_mean, inter_mean, slice_ctus:List[CTU], refs:Iterator[int]):

    delta = 0
    for first_ref_idx in refs:
        delta = frame_idx - first_ref_idx
        break

    if delta > 1:
        # fake retransmission bandwidth overhead, 
        # and estimates if intra has lighter payload
        inter_bits = encode_inter_slice_X(inter_mean, delta)
        intra_stats = encode_intra_slice(intra_mean, len(slice_ctus))
        if inter_bits >= intra_stats.total_size:
            raise SliceTypeOverride(str(SliceType.IDR), SliceType.IDR, intra_stats)
        else:
            stats = SliceStats()
            stats.inter_ctu_count = len(slice_ctus)
            stats.inter_ctu_bits = inter_bits
            return stats 
    else:
        return encode_inter_slice_1(intra_mean, inter_mean, slice_ctus)

def is_perodic_intra_frame(frame_poc, gop_size, prev_idr_idx=0):
    return (frame_poc == 0) | (((frame_poc - prev_idr_idx) % gop_size) == 0)

def is_perodic_intra_slice(slice_idx, slices_per_frame, frame_poc, prev_idr_idx=0):
    return (frame_poc == 0) | (((frame_poc - prev_idr_idx) % slices_per_frame) == slice_idx)

def validate_encoder_config(cfg):

    err_modes = [
        ErrorResilienceMode.DISABLED,
        ErrorResilienceMode.PERIODIC_FRAME,
        ErrorResilienceMode.PERIODIC_SLICE,
        ErrorResilienceMode.FEEDBACK_BASED,
        ErrorResilienceMode.FEEDBACK_BASED_ACK,
    ]

    if not cfg.error_resilience_mode in err_modes:
        raise EncoderConfigException(f'invalid resilience mode. must be one of {err_modes}')

    if cfg.error_resilience_mode != ErrorResilienceMode.PERIODIC_FRAME.value and cfg.gop_size != -1:
        raise EncoderConfigException('custom gop is only valid for PERIODIC_FRAME error resilience')

    if cfg.error_resilience_mode == ErrorResilienceMode.PERIODIC_FRAME and cfg.gop_size == -1:
        raise EncoderConfigException('PERIODIC_FRAME error resilience requires an explicit gop size, eg. -g=60')

    if cfg.error_resilience_mode == ErrorResilienceMode.PERIODIC_SLICE and cfg.slices_per_frame == 1:
        raise EncoderConfigException('PERIODIC_SLICE error resilience requires multiple slices per frame')

    if (cfg.frame_width % CTU_SIZE) != 0:
        raise EncoderConfigException(f'frame width must be a multiple of {CTU_SIZE}')

    if (cfg.frame_height % CTU_SIZE) != 0:
        raise EncoderConfigException(f'frame height must be a multiple of {CTU_SIZE}')

    if ((cfg.frame_height / cfg.slices_per_frame ) % CTU_SIZE) != 0:
        raise EncoderConfigException(f'({cfg.frame_height}px / {cfg.slices_per_frame} slices) is not a multiple of {CTU_SIZE}')


class Encoder:

    _cfg:EncoderConfig = None
    _ctus_per_slices:int
    _ctus_per_frame:int
    
    _feedback:FeedbackProvider = None
    _prev_idr_idx:int = 0
    _refs:RefPicList = RefPicList(max_size=MAX_REFS)
    _rc_max_bits:int = -1

    @property
    def refs_list_0(self):
        return self._refs

    def __init__(self, cfg:EncoderConfig, feedback_provider:FeedbackProvider=None, view_idx:int=0):
        
        validate_encoder_config(cfg)

        if cfg.error_resilience_mode >= ErrorResilienceMode.FEEDBACK_BASED and feedback_provider == None:
            raise EncoderConfigException('Feedback based error resilience requires a feedback provider')

        self._cfg = cfg
        self._view_idx = view_idx
        self._feedback = feedback_provider
        self._default_referenceable_status = cfg.error_resilience_mode != ErrorResilienceMode.FEEDBACK_BASED_ACK
        
        w = cfg.frame_width
        h = cfg.frame_height
        self._ctus_per_frame = int(( w * h ) / ( CTU_SIZE * CTU_SIZE ))
        self._ctus_per_slice = int(( h / CTU_SIZE ) * h / ( cfg.slices_per_frame * CTU_SIZE ))

        logger.info(f'frame WxH: {w} x {h}')
        logger.info(f'slice height: {int(h / cfg.slices_per_frame)}')
        logger.info(f'ctus p. frame: {self._ctus_per_frame}')
        logger.info(f'ctus p. slice: {self._ctus_per_slice}')


    def encode(self, vtrace:VTrace) -> Iterator[Slice]:
        
        validates_ctu_distribution(vtrace, raise_exception=True)

        intra_mean = get_intra_mean(vtrace, self._ctus_per_frame)
        inter_mean = get_inter_mean(vtrace, self._ctus_per_frame)

        # frame type decision, based on error resilience mode
        is_idr_frame = False
        
        if self._cfg.error_resilience_mode == ErrorResilienceMode.DISABLED:
            is_idr_frame = vtrace.is_full_intra

        elif self._cfg.error_resilience_mode == ErrorResilienceMode.PERIODIC_FRAME:
            if is_perodic_intra_frame(vtrace.poc, self._cfg.gop_size, self._prev_idr_idx):
                is_idr_frame = True

        elif self._cfg.error_resilience_mode >= ErrorResilienceMode.FEEDBACK_BASED:
            is_idr_frame = (vtrace.poc == 0) or self._feedback.intra_refresh_status()
            if is_idr_frame: 
                self._feedback.clear_intra_refresh_status()
            self._rc_max_bits = self._feedback.rc_max_bits
        
        frame = Frame(vtrace.poc, idr=is_idr_frame, view_idx=self._view_idx)

        if frame.is_idr_frame:
            self._prev_idr_idx = vtrace.poc
            frame.ctu_map = draw_ctus( 100, 0, 0, 0, size = self._ctus_per_frame )
        else:
            frame.ctu_map = draw_ctus(
                vtrace.ctu_intra_pct,
                vtrace.ctu_inter_pct,
                vtrace.ctu_skip_pct,
                vtrace.ctu_merge_pct,
                size = self._ctus_per_frame
            )
        
        for slice_idx in range(self._cfg.slices_per_frame):
            
            # @TODO: make referenceable configuration to False, eg. ACK mode
            S:Slice = Slice(vtrace.pts, vtrace.poc, slice_idx, intra_mean, inter_mean, referenceable=self._default_referenceable_status, view_idx=self._view_idx) 
            
            # slice type decision, based on error resilience mode
            if self._cfg.error_resilience_mode == ErrorResilienceMode.DISABLED:
                if frame.is_idr_frame:
                    S.slice_type = SliceType.IDR
                else:
                    S.slice_type = SliceType.P
                    
            if self._cfg.error_resilience_mode == ErrorResilienceMode.PERIODIC_SLICE:
                if is_perodic_intra_slice(slice_idx, self._cfg.slices_per_frame, vtrace.poc, self._prev_idr_idx):
                    S.slice_type = SliceType.IDR
                else:
                    S.slice_type = SliceType.P

            elif self._cfg.error_resilience_mode >= ErrorResilienceMode.FEEDBACK_BASED:
                if frame.is_idr_frame:
                    S.slice_type = SliceType.IDR
                else:
                    S.slice_type = SliceType.P
                self._refs = self._feedback.apply_feedback(self._refs)

            # model slice size based on Vtrace
            if S.slice_type == SliceType.IDR:
                S.stats = encode_intra_slice(intra_mean, self._ctus_per_slice)
                S.refs = []
            
            else:
                try:
                    slice_ctus = get_slice_ctus(frame.ctu_map, self._cfg.slices_per_frame, slice_idx)
                    S.stats = encode_inter_slice(vtrace.poc, intra_mean, inter_mean, slice_ctus, self._refs.slice_refs(slice_idx))
                    for ref_pic_idx in self._refs.slice_refs(slice_idx):
                        S.refs.append(ref_pic_idx)
                    S.refs.reverse()
                    S.slice_type = SliceType.P

                except SliceTypeOverride as s:
                    S.stats = s.stats
                    S.slice_type = s.slice_type

            # sanity check
            assert S.slice_type != None
            assert S.bits_ref > 0

            # apply bitrate adjustment model
            S.qp_ref =  vtrace.inter_qp_ref if not S.slice_type == SliceType.IDR else vtrace.intra_qp_ref
            size_new, qp_new = None, S.qp_ref
            
            if self._rc_max_bits > 0 :
                crf_new = self._cfg.crf if self._cfg.crf else vtrace.crf_ref
                while size_new > self._rc_max_bits:
                    crf_new += 1
                    size_new, qp_new = model_crf_adjustment(S.bits_ref, crf_new, vtrace.crf_ref, S.qp_ref)
                    if crf_new == 51: 
                        break
                S.qp_new = qp_new
                S.bits_new = size_new

            elif self._cfg.crf >= 0:
                size_new, qp_new = model_crf_adjustment(S.bits_ref, self._cfg.crf, vtrace.crf_ref, S.qp_ref)
                S.qp_new = qp_new
                S.bits_new = size_new

            # apply PSNR adjustment model
            for c, psnr_ref in vtrace.get_psnr_ref(S.slice_type).items():
                psnr_new = model_pnsr_adjustment(S.qp_new, S.qp_ref, psnr_ref)
                S.__setattr__(c, psnr_new)

            # apply encoding time adjustment model
            S.total_time = model_encoding_time(S, vtrace, self._cfg.slices_per_frame)
            
            frame.slices.append(S)
            yield S

        self._refs.add_frame(frame)
        