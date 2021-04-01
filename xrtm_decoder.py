#!/usr/bin/python3
import sys
import os
import shutil
import json
import random
import logging
import argparse
from pathlib import Path
from functools import reduce

from typing import Iterable, Tuple, List

from xrtm.models import (
        STraceRx,
        FrameConfig,
        Frame,
        CuMap,
        CU_status,
        CU_mode,
        VTraceRx,
        VTraceTx
    )
from xrtm.feedback import Feedback

logger = logging.getLogger(__name__)

CU_LEFT = 128
CU_TOP = 64
CU_RIGHT = 32
CU_BOTTOM = 16
CU_LEFTTOP = 8
CU_TOPRIGHT = 4
CU_RIGHTBOTTOM = 2
CU_BOTTOMLEFT = 1

def model_cu_references() -> int:
    # 4 random bits: Left, Top, Right, Bottom
    # folowed by 4 bits for corners: LeftTop, TopRight, RightBottom, BottomLeft
    ltrb = random.getrandbits(4)
    def corner(ltrb, c):
        cv = (ltrb & c)
        if cv == c:
            return 1
        elif cv == 0:
            return 0
        else:
            return random.getrandbits(1)
    return ltrb << 4 | corner(ltrb, 12) << 3 | corner(ltrb, 6) << 2 | corner(ltrb, 3) << 1 | corner(ltrb, 9)

def iter_cu_references(refs:int, address:int, width:int, height:int) -> Iterable[int]:
    
    x, y = address % width, address // width
    left = x > 0
    right = x < (width-1)
    top = y > 0
    bottom = y < (height-1)
    cu_address = lambda xx, yy: yy * width + xx

    if left and (refs & CU_LEFT):
        yield cu_address(x-1, y)

    if top:
        if left and (refs & CU_LEFTTOP):
            yield cu_address(x-1, y-1)
        if refs & CU_TOP:
            yield cu_address(x, y-1)
        if right and (refs & CU_TOPRIGHT):
            yield cu_address(x+1, y-1)

    if right and (refs & CU_RIGHT):
        yield cu_address(x+1, y)

    if bottom:
        if right and (refs & CU_RIGHTBOTTOM):
            yield cu_address(x+1, y+1)
        if refs & CU_BOTTOM:
            yield cu_address(x, y+1)
        if left and (refs & CU_BOTTOMLEFT):
            yield cu_address(x-1, y+1)
    
    yield address


def strace_rx_stream(fp:Path, timestep_micro_s:int) -> List[STraceRx]:
    """
    Stream Sp-Trace from csv in a timestep increment.
    returns a list of STraceRx which may be empty.
    """
    # raw_traces = [ s for s in STraceRx.iter_csv_file(fp) ]
    # sp_stream = sorted( raw_traces, key=lambda x: x.time_stamp_in_micro_s )
    sp_stream = sorted( STraceRx.iter_csv_file(fp), key=lambda x: x.time_stamp_in_micro_s )
    timestamp = sp_stream[0].time_stamp_in_micro_s
    step_slices = []
    for sp in sp_stream:
        if sp.time_stamp_in_micro_s > (timestamp + timestep_micro_s):
            yield timestamp, step_slices
            timestamp += timestep_micro_s
            step_slices = []
        step_slices.append(sp)
    yield timestamp, step_slices


class DecoderConfig(FrameConfig):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strace_input = './S-Trace.csv'
        self.sptrace_input = './Sp-Trace.csv'
        self.vptrace_output = './Vp-Trace.csv'
        self.delay_budget = 50 * 1e3 # 50ms as micro_s

    def get_frames_dir(self, user_idx=-1, mkdir=False, overwrite=False) -> Path:
        p = Path(self.strace_input)
        if user_idx < 0:
            p = p.parent / f'{p.stem}.frames/'
        else:
            p = p.parent / f'{p.stem}[{str(user_idx)}].frames/'
        if mkdir:
            if overwrite and p.exists() and p.is_dir:
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=False)
        return p

    def get_strace_input(self, user_idx=-1) -> Path:
        p = Path(self.strace_input)
        if user_idx < 0:
            return p
        else:
            return p.parent / f'{p.stem}[{str(user_idx)}]{p.suffix}'

    def get_sptrace_input(self, user_idx=-1, buff_idx=-1) -> Path:
        p = Path(self.sptrace_input)
        name = f'{p.stem}'
        if user_idx >= 0:
            name += f'[{str(user_idx)}]'
        if buff_idx >= 0:
            name += f'-{buff_idx}'
        return p.parent / f'{name}{p.suffix}'

    def get_vptrace_output(self, user_idx=-1, buff_idx=-1) -> Path:
        p = Path(self.vptrace_output)
        name = f'{p.stem}'
        if user_idx >= 0:
            name += f'[{str(user_idx)}]'
        if buff_idx >= 0:
            name += f'-{buff_idx}'
        return p.parent / f'{name}{p.suffix}'

    @classmethod
    def parse_config(cls, data:dict) -> 'DecoderConfig':
        cfg = cls()
        cfg.strace_input = data["Input"]["S-Trace"]
        cfg.sptrace_input = data["Input"]["Sp-Trace"]
        cfg.vptrace_output = data["Vp-Trace"]
        cfg.delay_budget = int(data["maxDelay"]) * 1e3
        return cfg

    @classmethod
    def load(cls, p:Path) -> 'DecoderConfig':
        with open(p, 'r') as f:
            cfg = json.load(f)
            return cls.parse_config(cfg)



class Decoder:

    def __init__(self, cfg:DecoderConfig, t0=0, fps=60, frames_dir=None):
        self.t0 = t0
        self.frame_duration = 1e6 // fps
        self.buffer_framecount = 1 + int(cfg.delay_budget // self.frame_duration)
        
        self.cfg = cfg
        self.frames_dir = Path(frames_dir) if frames_dir else None

        self.dt = t0
        self.curr_frame_idx = 0
        # TODO: these buffers are not purged for frames no longer referenced, 
        #   could grow too large and slow down the simulation
        self.frames = {}
        self.frame_timing = {}
        
    def process(self, dt:int, slices:Iterable[STraceRx]) -> Tuple[VTraceRx, Iterable[Feedback]]:
        
        assert dt < self.frame_duration, "'dt' timestep must be shorter than frame duration"
        self.dt += dt

        for s in slices:
            # load frame if it hasn't already been loaded
            frame_idx = self.cfg.parse_frame_idx(s.frame_file)
            frame = self.frames.get(frame_idx, None)
            if frame == None:
                frame_file = self.frames_dir/s.frame_file if self.frames_dir else s.frame_file
                frame = Frame(self.cfg, None, frame_idx, frame_file=frame_file)
                self.frames[frame_idx] = frame
                self.frame_timing[frame_idx] = [s.render_timing, s.time_stamp_in_micro_s]
            else:
                assert self.frame_timing[frame_idx][0] == s.render_timing, "'render_timing' must be the same in all slices of a frame"
                self.frame_timing[frame_idx][1] = max(self.frame_timing[frame_idx][1], s.time_stamp_in_micro_s) # frame timing is that of the latest slice
            
            if (not s.is_decodable()) or s.is_late(self.cfg.delay_budget):
                # TODO: Indicate the slice loss for feedback
                for cu in frame.cu_map.get_slice(s.start_address_cu, s.number_cus):
                    # Mark all CUs as unavailable
                    cu.status = CU_status.UNAVAILABLE
            else:
                # assert (self.curr_frame_idx * self.frame_duration) < (s.render_timing + self.cfg.max_delay), 'slice is late'
                # TODO: Indicate the slice received for “feedback”
                for cu in frame.cu_map.get_slice(s.start_address_cu, s.number_cus):
                    if cu.mode == CU_mode.INTRA:
                        # If it is an intra CU, mark it correct
                        cu.status = CU_status.OK
                    else:
                        # If it is an inter CU and it references a damaged or unavailable CU, 
                        # mark it as damaged, otherwise mark it as correct
                        ref_frame_idx = cu.reference # frame_idx - cu.reference
                        ref_frame = self.frames.get(ref_frame_idx, None)
                        if ref_frame == None:
                            cu.status = CU_status.DAMAGED
                        else:
                            # TODO: move reference modeling to the encoder
                            for cu_address in iter_cu_references(model_cu_references(), cu.address, self.cfg.cu_count_width, self.cfg.cu_count_height):
                                if ref_frame.cu_map.cus[cu_address].status != CU_status.OK:
                                    cu.status = CU_status.DAMAGED
                                    break # at least one modeled reference is DAMAGED or UNAVAILABLE
                                else:
                                    cu.status = CU_status.OK

        trace = None
        feedback = []
        idx = int(self.dt // self.frame_duration)

        if idx > (self.curr_frame_idx + self.buffer_framecount):
            trace = VTraceRx({})
            frame = self.frames.get(self.curr_frame_idx, None)
            if frame != None:
                trace.set_frame_stats(frame)
                trace.render_timing = self.frame_timing[self.curr_frame_idx][0]
                trace.time_stamp_in_micro_s = self.frame_timing[self.curr_frame_idx][1]
            else:
                count = self.cfg.get_cu_per_frame()
                trace.total_CUs = count
                trace.lost_CUs = count
                trace.correct_CUs = 0
                trace.damaged_CUs = 0
                trace.bits = 0
                trace.render_timing = -1
                trace.time_stamp_in_micro_s = 0
            self.curr_frame_idx = (idx - self.buffer_framecount)

        return trace, feedback


    def flush(self):
        last_frame_idx = max(self.frames.keys())
        while self.curr_frame_idx <= last_frame_idx:
            trace = VTraceRx({})
            frame = self.frames.get(self.curr_frame_idx, None)
            if frame != None:
                trace.set_frame_stats(frame)
                trace.render_timing = self.frame_timing[self.curr_frame_idx][0]
                trace.time_stamp_in_micro_s = self.frame_timing[self.curr_frame_idx][1]
            else:
                count = self.cfg.get_cu_per_frame()
                trace.total_CUs = count
                trace.lost_CUs = count
                trace.correct_CUs = 0
                trace.damaged_CUs = 0
                trace.bits = 0
                trace.render_timing = -1
                trace.time_stamp_in_micro_s = 0
            self.curr_frame_idx += 1
            yield trace


if __name__ == "__main__":

    print('#', *sys.argv)

    parser = argparse.ArgumentParser(description='STrace encoder configuration')
    parser.add_argument('-c', '--config', help='encoder config', type=str, required=True)
    parser.add_argument('-u', '--user_id', help='user id', type=int, required=False, default=-1)
    parser.add_argument('-b', '--buffer_id', help='buffer id', type=int, required=False, default=-1)
    parser.add_argument('-l', '--log_level', type=int, default=0, help='default=0 - set log level. 0:CRITICAL, 1:INFO, 2:DEBUG', required=False)
    parser.add_argument('-t', '--timestep', help='process data in (1e6/timestep) increments', type=int, required=False, default=120)
    parser.add_argument('-d', '--frames-dir', help='frames dir to look up for frame files', type=str, required=False, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    assert cfg_path.exists(), f'config file not found: {cfg_path.resolve()}'

    cfg = DecoderConfig.load(cfg_path)
    dec = Decoder(cfg, t0=0, frames_dir=args.frames_dir)
    dt = 1e6/args.timestep
    
    x = 0
    with open(cfg.get_vptrace_output(args.user_id, args.buffer_id), 'w') as vtf:
        vtof = VTraceRx.get_csv_writer(vtf)
        for t, s in strace_rx_stream(cfg.get_sptrace_input(args.user_id, args.buffer_id), dt):
            vtrx, feedback = dec.process(dt, s)
            if vtrx != None:
                x += 1
                vtof.writerow(vtrx.get_csv_dict())
        for vtrx in dec.flush():
            x += 1
            vtof.writerow(vtrx.get_csv_dict())
    
    print(f'# processed {x} frames - DONE')
