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
        self.refs = RefPicList(max_size=cfg.max_refs)
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
        frame = Frame(vtrace, self.cfg, view_idx=self._view_idx)
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
                rpl = self.refs.get_rpl(S.slice_idx)
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
                    frame.draw(address=S.cu_address, count=S.cu_count, intra_refresh=True)

        # encode CU map, computes the frame size in bytes
        qp = self.rc.target_qp
        size = frame.encode(qp, self.refs) * 8
        budget = self.rc.get_frame_budget()

        # adjust if too large
        if self.rc.mode in [RC_mode.cVBR, RC_mode.CBR] and (size > budget):
            qp = int(min(frame.i_qp, frame.p_qp))
            while size > budget:
                if self.rc.qp_max > 0 and qp > self.rc.qp_max:
                    break
                size = frame.encode(qp, self.refs) * 8
                qp += 1

        # adjust if too small
        elif self.rc.mode == RC_mode.CBR and (size < budget):
            qp = int(max(frame.i_qp, frame.p_qp))
            while size < budget:
                if self.rc.qp_min > 0 and qp < self.rc.qp_min:
                    break
                size = frame.encode(qp, self.refs) * 8
                qp -= 1
        
        self.refs.push(frame)
        self.frame_idx += 1
        if is_perodic_intra:
            self.periodic_intra_slice_idx = (self.periodic_intra_slice_idx+1) % self.cfg.slices_per_frame
        return [STraceTx.from_slice(S) for S in frame.slices]


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
            yield vtraces[idx]
            i += 1
        

class MultiViewEncoder:

    def __init__(self, cfg:EncoderConfig, user_idx=-1):
        self.cfg = cfg
        self.user_idx = user_idx
        self.frame_idx = 0
        self.rc = RateControl(self.cfg)
        self.buffers = [ BaseEncoder(cfg, self.rc, user_idx, view_idx) for view_idx, buff in enumerate(self.cfg.buffers) ]
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

        timestamp = self.frame_idx * self.frame_duration + self.cfg.get_pre_delay()
        straces = []
        buff_idx = 0
        
        for vtrace, enc in zip(vtraces, self.buffers):
            refresh = ''
            frame_file = f'{self.frame_idx}_{buff_idx}.csv'
            slice_delay = 0
            for s in enc.encode(vtrace, feedback):
                s.frame_idx = self.frame_idx
                s.render_timing = round(timestamp)
                slice_delay += self.cfg.get_encoding_delay()
                s.time_stamp_in_micro_s = round(timestamp + slice_delay + self.cfg.get_buffer_delay(buff_idx))
                s.index = self.slice_idx
                s.eye_buffer = buff_idx
                s.frame_file = str(self.frames_dir / frame_file)
                refresh += str(s.type.value)
                straces.append(s)
                self.slice_idx += 1
            enc.dump_frame_file( self.frames_dir / frame_file )
            print(f'{buff_idx}: {refresh}')
            buff_idx += 1

        elapsed = round((time.perf_counter()-t0) * 1e3)
        print(f'frame: {self.frame_idx} - ts: {round(timestamp/1e3)}ms - processing: {elapsed}ms')

        self.frame_idx += 1
        return straces