import sys
import os
import re
import time
import csv
import json
import math
from enum import Enum, IntEnum
from typing import List, Iterator

import random
import logging
import argparse

from abc import ABC, abstractmethod

from .exceptions import (
    VTraceTxException,
    EncoderConfigException,
    RateControlException,
    FeedbackException
)

from .models import (
    VTraceTx,
    STraceTx,
    SliceType,
    Slice,
    Frame,
    CTU,
    RateControl,
    model_pnsr_adjustment
)

from .utils import (
    read_csv_vtraces
)

from .feedback import (
    FeedbackProvider,
    FeedbackType,
    ReferenceableList,
    RandomStereoFeedback
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

    @classmethod
    def get(cls, i:int):
        for e in cls.__members__.values():
            if e.value == i:
                return e
    

class EncoderConfig:
    
    def __init__(self, data:dict):
        self.frame_width = data.get('frame_width', 2048) 
        self.frame_height = data.get('frame_height', 2048)
        self.frame_rate = data.get('frame_rate', 60.0)
        self.slices_per_frame = data.get('slices_per_frame', 8)
        self.crf = data.get('crf', -1)
        self.rc_max_bits = data.get('rc_max_bits', -1)
        self.rc_window = data.get('rc_window', -1) # TODO
        self.gop_size = data.get('gop_size', -1)
        self.error_resilience_mode = ErrorResilienceMode.get(data.get('error_resilience_mode', 0))
        self.start_time = data.get('start_time', 0)
        self.slice_delay = data.get('slice_delay', 1)
        self.interleaved = data.get('interleaved', True)
        self.render_jitter_min = data.get('render_jitter_min', 15)
        self.render_jitter_max = data.get('render_jitter_max', 25)
        self.eye_buffer_delay = int(data.get('eye_buffer_delay', 1e+6 / (2 * self.frame_rate)))
        
        # CTU map config
        self.cu_size = data.get('cu_size', 64)
        self.cu_count = int(( self.frame_width * self.frame_height ) / self.cu_size**2 )
        self.cu_per_slice = int(( self.frame_width / self.cu_size ) * self.frame_height / ( self.slices_per_frame * self.cu_size ))
        # slices/CU refs
        self.max_refs = data.get('max_refs', 16)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
            return cls(data)


class RefPicList(ReferenceableList):
    max_size:int
    pics:List[Frame]
    
    def __init__(self, max_size:int=16):
        self.max_size = max_size
        self.pics = []

    def add_frame(self, pic:Frame):
        self.pics.append(pic)
        self.pics = self.pics[-self.max_size:]


    def iter(self, slice_idx): 
        for frame in reversed(self.pics):
            if frame.slices[slice_idx].referenceable:
                yield frame.frame_idx
            if frame.slices[slice_idx].slice_type == SliceType.IDR:
                return

    def set_referenceable_status(self, frame_idx:int, slice_idx:int, status:bool):
        for frame in reversed(self.pics):
            if frame.frame_idx == frame_idx:
                frame.slices[slice_idx].referenceable = status
                return

    def reset(self):
        self.pics = []

def get_intra_mean(vtrace, vtrace_ctu_count):
   return vtrace.intra_total_bits / vtrace_ctu_count

def get_inter_mean(vtrace, vtrace_ctu_count):
    if vtrace.ctu_intra_pct == 1.:
        return 0.
    intra_mean = vtrace.intra_total_bits / vtrace_ctu_count
    inter_bits = vtrace.inter_total_bits - (vtrace.ctu_intra_pct * intra_mean) - (vtrace.ctu_skip_pct * 8)
    inter_ctu_count = vtrace_ctu_count * (1 - vtrace.ctu_intra_pct - vtrace.ctu_skip_pct)
    return inter_bits / inter_ctu_count

def model_encoding_time(s:Slice, v:VTraceTx, slices_per_frame:int):
    return v.get_encoding_time(s.slice_type) / slices_per_frame

def ctu_bits(mean, variance):
    return random.gauss(mean, variance)

def validates_ctu_distribution(vt:VTraceTx, raise_exception=True) -> bool:
    total = vt.ctu_intra_pct + vt.ctu_inter_pct + vt.ctu_skip_pct + vt.ctu_merge_pct
    valid = round(total * 100) == 100
    if not valid:
        e_msg = f'Invalid CTU distribution on frame {vt.encode_order} - sum of all CTUs is {total}%'
        if raise_exception:
            raise VTraceTxException(e_msg)
        logger.critical(e_msg)
    return valid

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

def is_perodic_intra_frame(frame_idx:int, gop_size:int, offset=0):
    return (frame_idx == 0) | (((frame_idx - offset) % gop_size) == 0)

def is_perodic_intra_slice(slice_idx:int, slices_per_frame:int, frame_idx:int, offset=0):
    return (frame_idx == 0) | (((frame_idx - offset) % slices_per_frame) == slice_idx)


class AbstracEncoder(ABC):

    @abstractmethod
    def encode(self, vtrace:VTraceTx) -> Iterator[Slice]:
        raise NotImplementedError()


class BaseEncoder(AbstracEncoder):

    @property
    def refs_list_0(self):
        return self._refs

    def __init__(self, cfg:EncoderConfig, feedback_provider:FeedbackProvider=None, view_idx:int=0):
        
        validate_encoder_config(cfg)
        self.view_idx = view_idx
        self.cfg = cfg
        self.rc = RateControl(self.cfg.rc_max_bits, self.cfg.crf)

        if cfg.error_resilience_mode >= ErrorResilienceMode.FEEDBACK_BASED and feedback_provider == None:
            raise EncoderConfigException('Feedback based error resilience requires a feedback provider')
        self._feedback = feedback_provider
        self._default_referenceable_status = self.cfg.error_resilience_mode != ErrorResilienceMode.FEEDBACK_BASED_ACK
        self._refs = RefPicList(max_size=cfg.max_refs)

    def encode(self, vtrace:VTraceTx) -> Iterator[STraceTx]:
        
        # frame type decision, based on error resilience mode
        is_idr_frame = vtrace.encode_order == 0
        
        if self.cfg.error_resilience_mode == ErrorResilienceMode.DISABLED:
            is_idr_frame = vtrace.is_full_intra

        elif self.cfg.error_resilience_mode == ErrorResilienceMode.PERIODIC_FRAME:
            is_idr_frame = is_perodic_intra_frame(vtrace.encode_order, self.cfg.gop_size)
        
        elif self.cfg.error_resilience_mode >= ErrorResilienceMode.FEEDBACK_BASED:
            is_idr_frame = vtrace.is_full_intra
            self.rc.max_bits = self._feedback.rc_max_bits

        validates_ctu_distribution(vtrace, raise_exception=True)
        intra_mean = get_intra_mean(vtrace, self.cfg.cu_count)
        inter_mean = get_inter_mean(vtrace, self.cfg.cu_count)
        frame = Frame(vtrace.encode_order, intra_mean, inter_mean, self.cfg.slices_per_frame, self.cfg.cu_per_slice, self.cfg.cu_count, self.cfg.cu_size, view_idx=self.view_idx)
        frame.draw(*vtrace.get_cu_distribution())

        for S in frame.slices:
            # slice type decision, based on error resilience mode
            if self.cfg.error_resilience_mode == ErrorResilienceMode.DISABLED:
                S.slice_type = SliceType.IDR if is_idr_frame else SliceType.P
            
            if self.cfg.error_resilience_mode == ErrorResilienceMode.PERIODIC_SLICE:
                if is_perodic_intra_slice(S.slice_idx, self.cfg.slices_per_frame, vtrace.encode_order):
                    S.slice_type = SliceType.IDR
                else:
                    S.slice_type = SliceType.P
            
            elif self.cfg.error_resilience_mode >= ErrorResilienceMode.FEEDBACK_BASED:
                S.slice_type = SliceType.P if not is_idr_frame else SliceType.IDR 
                self._refs = self._feedback.apply_feedback(self._refs)

            if S.slice_type == SliceType.P:
                try:
                    nearest = next(self._refs.iter(S.slice_idx))
                    delta = frame.frame_idx - nearest
                    if (delta > 1) and (inter_mean * delta) > (intra_mean * frame.cu_per_slice):
                        S.slice_type == SliceType.IDR
                except StopIteration:
                        pass
            
            # encode CTUs & bitrate adjustment 
            bits, qp = frame.encode_slice(S, self.rc, [*self._refs.iter(S.slice_idx)])
            S.qp = qp
            S.size = math.ceil(bits/8)
            
            # apply PSNR adjustment model
            for comp, ref in vtrace.get_psnr_ref(S.slice_type).items():
                psnr_new = model_pnsr_adjustment(S.qp, vtrace.get_qp_ref(S.slice_type), ref)
                setattr(S, comp, psnr_new)
            
            frame.slices[S.slice_idx] = S
            yield STraceTx.from_slice(S)

        self._refs.add_frame(frame)
        frame.cu_map 


def render_jitter(cfg:EncoderConfig) -> int:
    return random.randint(cfg.render_jitter_min, cfg.render_jitter_max)

class MonoEncoder:
    
    def __init__(self, cfg:EncoderConfig, feedback_provider:FeedbackProvider=None):
        self.frame_idx = 0
        self.cfg = cfg
        self.mono = BaseEncoder(cfg, feedback_provider=feedback_provider)

    def process(self, frames:Iterator[VTraceTx]) -> Iterator[Slice]:

        fps = 1e+6 / self.cfg.frame_rate

        for vtrace in frames:
            ts = int(vtrace.encode_order * fps) + self.cfg.start_time
            render_timing = ts + render_jitter(self.cfg)

            for i, s in enumerate(self.mono.encode(vtrace)):
                slice_delay = render_timing + (i * self.cfg.slice_delay)
                s.render_timing = render_timing
                s.time_stamp_in_micro_s = slice_delay
                yield s
            self.frame_idx += 1


class StereoEncoder:

    def __init__(self, cfg:EncoderConfig, feedback:RandomStereoFeedback=None):
        self.frame_idx = 0
        self.cfg = cfg # TODO: could be one per eye, has not been needed so far
        
        if feedback != None:
            self._enc0 = BaseEncoder(cfg, feedback.enc0, view_idx=1)
            self._enc1 = BaseEncoder(cfg, feedback.enc1, view_idx=2)
        else:
            self._enc0 = BaseEncoder(cfg, None, view_idx=1)
            self._enc1 = BaseEncoder(cfg, None, view_idx=2)

    def process(self, frames:Iterator[VTraceTx]) -> Iterator[Slice]:
        
        fps = 1e+6 / self.cfg.frame_rate
        
        for vtrace in frames:
            ts = int(vtrace.encode_order * fps) + self.cfg.start_time
            render_timing = ts + render_jitter(self.cfg)
            
            eye_buff0 = self._enc0.encode(vtrace)
            eye_buff1 = self._enc1.encode(vtrace)

            if self.cfg.interleaved:
                for i, s in enumerate(eye_buff0):
                    slice_delay = (i * self.cfg.slice_delay)
                    s.render_timing = render_timing
                    s.time_stamp_in_micro_s = render_timing + slice_delay
                    yield s
                render_timing += self.cfg.eye_buffer_delay
                for i, s in enumerate(eye_buff1):
                    slice_delay = (i * self.cfg.slice_delay)
                    s.render_timing = render_timing
                    s.time_stamp_in_micro_s = render_timing + slice_delay
                    yield s
    
            else:
                for i, s0 in enumerate(eye_buff0):
                    slice_delay = (i * self.cfg.slice_delay)
                    s0.render_timing = render_timing
                    s0.time_stamp_in_micro_s = render_timing + slice_delay
                    yield s0
                    s1 = next(eye_buff1)
                    s1.render_timing = render_timing
                    s1.time_stamp_in_micro_s = render_timing + slice_delay
                    yield s1

