import sys
import os
import re
import time
import csv
import json
import math
from pathlib import Path
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
    CU,
    RateControl,
    EncoderConfig,
    ErrorResilienceMode,
    model_pnsr_adjustment
)

from .utils import (
    read_csv_vtraces
)

from .feedback import (
    FeedbackProvider,
    FeedbackType,
    RandomStereoFeedback,
    ReferenceableList
)

logger = logging.getLogger(__name__)

def validates_ctu_distribution(vt:VTraceTx, raise_exception=True) -> bool:
    total = vt.ctu_intra_pct + vt.ctu_inter_pct + vt.ctu_skip_pct + vt.ctu_merge_pct
    valid = round(total * 100) == 100
    if not valid:
        e_msg = f'Invalid CU distribution on frame {vt.encode_order} - sum of all CUs is {total}%'
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

    if (cfg.frame_width % cfg.cu_size) != 0:
        raise EncoderConfigException(f'frame width must be a multiple of {cfg.cu_size}')

    if (cfg.frame_height % cfg.cu_size) != 0:
        raise EncoderConfigException(f'frame height must be a multiple of {cfg.cu_size}')

    if ((cfg.frame_height / cfg.slices_per_frame ) % cfg.cu_size) != 0:
        raise EncoderConfigException(f'({cfg.frame_height}px / {cfg.slices_per_frame} slices) is not a multiple of {cfg.cu_size}')

def is_perodic_intra_frame(frame_idx:int, gop_size:int, offset=0) -> bool:
    return (frame_idx == 0) | (((frame_idx - offset) % gop_size) == 0)

def is_perodic_intra_slice(slice_idx:int, slices_per_frame:int, frame_idx:int, offset=0) -> bool:
    return (frame_idx == 0) | (((frame_idx - offset) % slices_per_frame) == slice_idx)

class RefPicList(ReferenceableList):
    
    def __init__(self, max_size:int=16):
        self.max_size = max_size
        self.pics = []

    def is_empty(self) -> bool:
        return len(self.pics) == 0

    def push(self, pic:Frame):
        self.pics.append(pic)
        self.pics = self.pics[-self.max_size:]

    def get_rpl(self, slice_idx:int) -> List[int]:
        return [*self.iter(slice_idx)]

    def iter(self, slice_idx:int) -> Iterator[int]:
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


class AbstracEncoder(ABC):

    @abstractmethod
    def encode(self, vtrace:VTraceTx) -> Iterator[Slice]:
        raise NotImplementedError()


class BaseEncoder(AbstracEncoder):

    def __init__(self, cfg:EncoderConfig, feedback_provider:FeedbackProvider=None, view_idx:int=0):
        
        validate_encoder_config(cfg)
        self.view_idx = view_idx
        self.cfg = cfg
        self.rc = RateControl(self.cfg.rc_max_bits, self.cfg.crf)

        if cfg.error_resilience_mode >= ErrorResilienceMode.FEEDBACK_BASED and feedback_provider == None:
            raise EncoderConfigException('Feedback based error resilience requires a feedback provider')
        self._feedback = feedback_provider
        self._default_referenceable_status = self.cfg.error_resilience_mode != ErrorResilienceMode.FEEDBACK_BASED_ACK
        self.refs = RefPicList(max_size=cfg.max_refs)

    def encode(self, vtrace:VTraceTx) -> Iterator[STraceTx]:
        
        validates_ctu_distribution(vtrace, raise_exception=True)
        
        if self.cfg.error_resilience_mode == ErrorResilienceMode.DISABLED:
            is_idr_frame = vtrace.is_full_intra
        elif self.cfg.error_resilience_mode == ErrorResilienceMode.PERIODIC_FRAME:
            is_idr_frame = is_perodic_intra_frame(vtrace.encode_order, self.cfg.gop_size)
        else:
            is_idr_frame = vtrace.encode_order == 0
        
        if self.cfg.error_resilience_mode >= ErrorResilienceMode.FEEDBACK_BASED:
            self._feedback.update_rpl(self.refs)
            self.rc.max_bits = self._feedback.rc_max_bits

        frame = Frame(vtrace, self.cfg, view_idx=self.view_idx)
        frame.draw(intra_refresh=is_idr_frame)

        for S in frame.slices:
            
            if self.cfg.error_resilience_mode == ErrorResilienceMode.PERIODIC_SLICE:
                if is_perodic_intra_slice(S.slice_idx, self.cfg.slices_per_frame, frame.frame_idx):
                    S.slice_type = SliceType.IDR
                else:
                    S.slice_type = SliceType.P
            else:
                S.slice_type = SliceType.IDR if is_idr_frame else SliceType.P

            rpl = self.refs.get_rpl(S.slice_idx)
            refresh = False

            if S.slice_type == SliceType.P:
                if len(rpl) == 0:
                    refresh = True
                else:
                    delta = frame.frame_idx - rpl[0]
                    refresh = (delta > 1) and (frame.inter_mean * delta) > (frame.intra_mean * frame.cu_per_slice)

            if refresh:
                S.slice_type = SliceType.IDR
                frame.draw(address=S.cu_address, count=S.cu_count, intra_refresh=True)
                rpl = []

            # encode CUs & bitrate adjustment
            size_ref = frame.encode(address=S.cu_address, count=S.cu_count, rpl=rpl)
            size_new, qp = self.rc.adjust(size_ref, S.slice_type == SliceType.IDR)
            S.size = int(size_new)
            S.qp = qp
            
            # apply PSNR adjustment model
            for comp, ref in vtrace.get_psnr_ref(S.slice_type).items():
                psnr_new = model_pnsr_adjustment(S.qp, vtrace.get_qp_ref(S.slice_type), ref)
                setattr(S, comp, psnr_new)
            
            yield STraceTx.from_slice(S)

        self.refs.push(frame)
        frame.cu_map.dump(self.cfg.frames_dir / frame.frame_file)


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
        self.cfg = cfg
        
        if feedback != None:
            self._enc0 = BaseEncoder(cfg, feedback.enc0, view_idx=1)
            self._enc1 = BaseEncoder(cfg, feedback.enc1, view_idx=2)
        else:
            self._enc0 = BaseEncoder(cfg, None, view_idx=1)
            self._enc1 = BaseEncoder(cfg, None, view_idx=2)

    def process(self, frames:Iterator[VTraceTx], count=-1) -> Iterator[Slice]:
        
        fps = 1e+6 / self.cfg.frame_rate
        idx = 0

        for vtrace in frames:
            if count != -1 and vtrace.encode_order > count:
                break

            ts = int(vtrace.encode_order * fps) + self.cfg.start_time
            render_timing = ts + render_jitter(self.cfg)
            
            eye_buff0 = self._enc0.encode(vtrace)
            eye_buff1 = self._enc1.encode(vtrace)

            if self.cfg.interleaved:
                for i, s in enumerate(eye_buff0):
                    slice_delay = (i * self.cfg.slice_delay)
                    s.render_timing = render_timing
                    s.time_stamp_in_micro_s = render_timing + slice_delay
                    s.index = idx
                    yield s
                    idx += 1
                    
                for i, s in enumerate(eye_buff1):
                    slice_delay = (i * self.cfg.slice_delay)
                    s.render_timing = render_timing
                    s.time_stamp_in_micro_s = self.cfg.eye_buffer_delay + render_timing + slice_delay
                    s.index = idx
                    yield s
                    idx += 1

            else:
                for i, s0 in enumerate(eye_buff0):
                    slice_delay = (i * self.cfg.slice_delay)
                    s0.render_timing = render_timing
                    s0.time_stamp_in_micro_s = render_timing + slice_delay
                    s.index = idx
                    yield s0
                    idx += 1

                    s1 = next(eye_buff1)
                    s1.render_timing = render_timing
                    s1.time_stamp_in_micro_s = render_timing + slice_delay
                    s.index = idx
                    yield s1
                    idx += 1

