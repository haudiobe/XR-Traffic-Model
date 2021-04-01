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

from .models import (
    VTraceTx,
    STraceTx,
    SliceType,
    Slice,
    Frame,
    CU,
    RateControl,
    RC_mode,
    EncoderConfig,
    ErrorResilienceMode,
    RefPicList
)


from .feedback import (
    Feedback,
    FeedbackType,
)

from .utils import ConfigException

logger = logging.getLogger(__name__)


def validate_encoder_config(cfg):

    err_modes = [
        ErrorResilienceMode.DISABLED,
        ErrorResilienceMode.PERIODIC_INTRA,
        ErrorResilienceMode.FEEDBACK_BASED,
        ErrorResilienceMode.FEEDBACK_BASED_ACK,
        ErrorResilienceMode.FEEDBACK_BASED_NACK,
    ]

    if not cfg.error_resilience_mode in err_modes:
        raise ConfigException(f'invalid resilience mode. must be one of {err_modes}')

    if cfg.error_resilience_mode == ErrorResilienceMode.PERIODIC_INTRA and cfg.intra_refresh_period == -1:
        raise ConfigException('PERIODIC_FRAME error resilience requires an explicit intra refresh preriod')

    if (cfg.frame_width % cfg.cu_size) != 0:
        raise ConfigException(f'frame width must be a multiple of {cfg.cu_size}')

    if (cfg.frame_height % cfg.cu_size) != 0:
        raise ConfigException(f'frame height must be a multiple of {cfg.cu_size}')

    if ((cfg.frame_height / cfg.slices_per_frame ) % cfg.cu_size) != 0:
        raise ConfigException(f'({cfg.frame_height}px / {cfg.slices_per_frame} slices) is not a multiple of {cfg.cu_size}')

def is_perodic_intra_refresh(frame_idx:int, intra_refresh_period:int, offset=0) -> bool: 
    return (frame_idx > 0) and (((frame_idx - offset) % intra_refresh_period) == 0)

def get_importance_index(refs:RefPicList, s:Slice, cfg:EncoderConfig) -> int:
    assert s.frame == refs.frames[-1]
    delta_max = cfg.slices_per_frame * cfg.intra_refresh_period
    delta = refs.get_previous_intra_delta(s)
    importance = delta_max - delta
    # print(f'[{s.frame_idx}][{s.view_idx}][{s.slice_idx}].type:{s.slice_type} - .delta:{delta} - .importance:{importance}')
    return importance


class AbstracEncoder(ABC):

    @abstractmethod
    def encode(self, vtrace:VTraceTx) -> Iterator[Slice]:
        raise NotImplementedError()


class BaseEncoder(AbstracEncoder):

    def __init__(self, cfg:EncoderConfig, rc:RateControl, user_idx=0, view_idx=0):
        
        validate_encoder_config(cfg)
        self._user_idx = user_idx
        self._view_idx = view_idx
        self.cfg = cfg
        self.rc = rc
        self._default_referenceable_status = self.cfg.error_resilience_mode != ErrorResilienceMode.FEEDBACK_BASED_ACK
        self.refs = RefPicList()
        self.frame_idx = 0
        self.periodic_intra_slice_idx = 0

    def dump_frame_file(self, p:Path):
        frame = self.refs.frames[-1]
        assert len(self.refs.frames) > 0, 'nothing to dump'
        frame.cu_map.dump( p )

    def encode(self, vtrace:VTraceTx, feedback:List[Feedback] = None) -> List[STraceTx]:
        
        is_intra_frame = False
        is_perodic_intra = False
        
        if self.frame_idx == 0:
            is_intra_frame = True
        elif self.cfg.error_resilience_mode == ErrorResilienceMode.DISABLED:
            is_intra_frame = vtrace.is_full_intra
        elif self.cfg.error_resilience_mode == ErrorResilienceMode.PERIODIC_INTRA:
            is_perodic_intra = (self.frame_idx % self.cfg.intra_refresh_period) == 0
            is_intra_frame == is_perodic_intra and (self.cfg.slices_per_frame == 1)
        
        # handle feedback
        if len(feedback) > 0:
            assert self.cfg.error_resilience_mode >= ErrorResilienceMode.FEEDBACK_BASED, 'got feedback but the encoder is not using it'
            for fb in feedback:
                if fb.type == FeedbackType.SLICE_ACK:
                    self.refs.set_referenceable_status(fb.frame_idx, fb.slice_idx, True)
                elif fb.type == FeedbackType.SLICE_NACK:
                    self.refs.set_referenceable_status(fb.frame_idx, fb.slice_idx, False)
                else:
                    # self.rc.handle_feedack(fb)
                    pass

        # draw CU map
        frame = Frame(self.cfg, view_idx=self._view_idx, frame_idx=self.frame_idx, vtrace=vtrace)
        frame.draw(intra_refresh=is_intra_frame)

        # slice type decision 
        for S in frame.slices:

            S.referenceable = self._default_referenceable_status
            
            if is_intra_frame:
                S.slice_type = SliceType.IDR
            elif is_perodic_intra and S.slice_idx == self.periodic_intra_slice_idx:
                S.slice_type = SliceType.IDR
            else:
                S.slice_type = SliceType.P
            
            # take feedback into account
            if self.cfg.error_resilience_mode >= ErrorResilienceMode.FEEDBACK_BASED:
                # rpl is the list of pictures that can be referenced by this slice
                rpl = self.refs.get_rpl(S)
                refresh = False
                if S.slice_type == SliceType.P:
                    if len(rpl) == 0:
                        refresh = True
                    else:
                        # gap is larger than 1, check if intra refresh is more efficient
                        delta = frame.frame_idx - rpl[0]
                        if delta > 1:
                            # TODO: this is wrong, needs clarification 
                            refresh = (frame.inter_mean * delta) > (frame.intra_mean * frame.cu_per_slice)
                if refresh:
                    S.slice_type = SliceType.IDR

            if S.slice_type == SliceType.IDR and not is_intra_frame:
                frame.draw(address=S.cu_address, count=S.cu_count, intra_refresh=True)
        
        # encode CU map, no bitrate constraints
        if self.rc.mode == RC_mode.VBR:
            if self.rc.target_qp < 0:
                size = frame.encode(self.refs) * 8
            else:
                size = frame.encode(self.refs, i_qp=self.rc.target_qp, p_qp=self.rc.target_qp) * 8
        else: # with bitrate constraints [cVBR, CBR]
            i_qp, p_qp = self.rc.estimate_qp(frame, intra=is_intra_frame)
            size = frame.encode(self.refs, i_qp=i_qp, p_qp=p_qp) * 8
            self.rc.add_frame_bits(size)

        self.refs.push(frame)
        self.frame_idx += 1

        if is_perodic_intra:
            self.periodic_intra_slice_idx = (self.periodic_intra_slice_idx + 1) % self.cfg.slices_per_frame
        
        traces = []
        for s in frame.slices:
            s.importance = get_importance_index(self.refs, s, self.cfg)
            traces.append(STraceTx.from_slice(s))
        
        return traces


class VTraceIterator:
    
    def __init__(self, cfg:EncoderConfig):
        self.cfg = cfg
        buffer_files = [Path(b["V-Trace"]) for b in self.cfg.buffers]
        self.buffers = [[*VTraceTx.iter_csv_file(csv)] for csv in buffer_files]
        if len(self.buffers) > 1:
            for b in self.buffers:
                assert len(b) == len(self.buffers[0]), "V-Trace csv files for all buffers must have matching row count"

    def __iter__(self):
        self.iterators = [self.iter_vtrace(b, self.cfg.start_frame, self.cfg.total_frames) for b in self.buffers]
        return self

    def __next__(self):
        return [next(it) for it in self.iterators]

    @classmethod
    def iter_vtrace(cls, vtraces:List[VTraceTx], start_frame=0, count:int=-1) -> Iterator[VTraceTx]:
        size = len(vtraces)
        start_frame = start_frame % size
        if count < 0:
            count == size
        i = 0
        while i < count:
            idx = (start_frame + i) % size
            if idx == 0:
                # skip V-trace #0 which has irrelevant QP 
                i += 1
                count += 1
                idx = (start_frame + i) % size
            yield vtraces[idx]
            i += 1
        

class MultiViewEncoder:

    def __init__(self, cfg:EncoderConfig, user_idx=-1):
        self.cfg = cfg
        self.user_idx = user_idx
        self.frame_idx = 0
        self.rc = []
        self.buffers = []
        for view_idx, _ in enumerate(self.cfg.buffers):
            rctl = RateControl(cfg)
            self.rc.append(rctl)
            enc = BaseEncoder(cfg, rctl, user_idx, view_idx)
            self.buffers.append(enc)
        self.frame_idx = 0
        self.frame_duration = cfg.get_frame_duration()
        self.slice_idx = 0
        self.frames_dir = self.cfg.get_frames_dir(user_idx, mkdir=True, overwrite=True)
    
    def process(self, vtraces:List[VTraceTx], feedback:List[Feedback]=None) -> List[STraceTx]:
        """
        call process(vtraces, feedback) to encode all buffers. a frame file is saved for each buffer.

        :vtraces: a V-Trace record is expected for each Buffer definition found in the encoder config
        :feedback: the feedback collected since the previous call
        :returns: the list of all straces generated
        """
        assert len(vtraces) == len(self.buffers), "expecting synchornous buffers (one vtrace p. buffer)"
        t0 = time.perf_counter()
        # render_timing 
        timestamp = self.frame_idx * self.frame_duration
        straces = []
        buff_idx = 0
        pre_delay = self.cfg.get_pre_delay()
        
        for vtrace, enc in zip(vtraces, self.buffers):
            frame_file = self.cfg.get_frame_file(self.frame_idx, buff_idx)
            
            if self.cfg.buffer_interleaving:
                render_timing = round(timestamp + self.cfg.get_buffer_delay(buff_idx))
                pre_delay = self.cfg.get_pre_delay()
            else:
                render_timing = round(timestamp)

            slice_delay = 0 # slice delay is reset for each buffer

            for s in enc.encode(vtrace, feedback):
                s.frame_idx = self.frame_idx
                s.render_timing = render_timing
                slice_delay += self.cfg.get_encoding_delay()
                s.time_stamp_in_micro_s = round(render_timing + pre_delay + slice_delay)
                s.index = self.slice_idx
                s.buffer = buff_idx
                s.frame_file = str(self.frames_dir / frame_file)
                straces.append(s)
                self.slice_idx += 1

            enc.dump_frame_file( self.frames_dir / frame_file )
            buff_idx += 1

        elapsed = round((time.perf_counter()-t0) * 1e3)
        print(f'frame: {self.frame_idx} - processing: {elapsed}ms')

        self.frame_idx += 1
        return straces