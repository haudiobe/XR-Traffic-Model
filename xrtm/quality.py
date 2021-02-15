from typing import Iterable
from pathlib import Path
import json
from xrtm.models import CSV, CsvRecord, VTraceTx, VTraceRx, STraceTx, STraceRx, PTraceTx, PTraceRx


class QTrace(CsvRecord):
    attributes = [
        CSV("user", int),
        CSV("buffer", int),
        CSV("total_packets", int),
        CSV("duration", int),
        CSV("PLoR", float),
        CSV("PLaR", float),
        CSV("SLR", float)
    ]
    """
        ,
        CSV("ALR", float),
        CSV("DAR", float),
        CSV("PSNR", float),
        CSV("PSNRyuv", float),
        CSV("RPSNR", float),
        CSV("RPSNRyuv", float)
    ]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.duration = -1
        self.total_packets = 0
        self.lost_packets = 0
        self.late_packets = 0
        self.total_slices = 0
        self.lost_slices = 0
        

class QualEvalCfg:

    def __init__(self, cfg:dict={}):
        self.users = cfg.get("Users")
        self.output = cfg.get("Q-Trace")
    
    def iter_inputs(self) -> Iterable['QualEvalInput']:
        for u in self.users:
            for b in u["Inputs"]:
                yield QualEvalInput(user_idx=u["Id"], buffer=b)

    @classmethod
    def load(cls, fp) -> 'QualEvalCfg':
        cfg = json.load(fp)
        return cls(cfg)


class QualEvalInput:

    def __init__(self, user_idx:int, buffer:dict):
        self.user_idx = user_idx
        self.buffer_idx = buffer["buffer"]
        self.pp_trace = buffer.get("Pp-Trace", None)
        self.sp_trace = buffer.get("Sp-Trace", None)
        self.vp_trace = buffer.get("Vp-Trace", None)



def q_eval(max_delay:int, qi:QualEvalInput) -> QTrace:

    q = QTrace({
        "user": qi.user_idx,
        "buffer": qi.buffer_idx,
        "duration": -1
    })
    
    q.lost_packets = 0
    q.late_packets = 0
    q.total_packets = 0
    with open(qi.pp_trace, 'r') as pp_trace:
        for pp in pp_trace:
            if pp.time_stamp_in_micro_s <= 0:
                q.lost_packets += 1
            elif (pp.render_timing + max_delay) > pp.time_stamp_in_micro_s:
                q.late_packets += 1
            q.total_packets += 1
    
    q.lost_slices = 0
    q.total_slices = 0
    with open(qi.sp_trace, 'r') as sp_trace:
        for sp in sp_trace:
            if sp.slice_recovery_position < sp.size:
                q.lost_slices += 1
            q.total_slices += 1
            if t < 0:
                t = pp.render_timing
        t = pp.render_timing - t
    
    q.PLoR = q.lost_packets / q.total_packets
    q.PLaR = q.late_packets / q.total_packets
    q.SLR = q.lost_slices / q.total_slices
    return q