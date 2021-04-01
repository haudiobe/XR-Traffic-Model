from typing import Iterable
from pathlib import Path
import json
from xrtm.models import CSV, CsvRecord, VTraceTx, VTraceRx, STraceTx, STraceRx, PTraceTx, PTraceRx, parse_and_scale_1000, scale_and_serialize_1000


class QTrace(CsvRecord):
    attributes = [
        CSV("user", lambda x: -1 if x == "Avg." else int(x), lambda x: "Avg." if x < 0 else x, 0),
        CSV("buffer", lambda x: -1 if x == "Avg." else int(x), lambda x: "Avg." if x < 0 else x, 0),
        CSV("total_packets", int),
        CSV("duration", int),
        CSV("PLoR", float),
        CSV("PLaR", float),
        CSV("SLR", float),
        CSV("CAR", float),
        CSV("ALR", float),
        CSV("DAR", float),
        CSV("LDR", float),
        CSV("PSNR", parse_and_scale_1000, scale_and_serialize_1000, 0),
        CSV("PSNRyuv", parse_and_scale_1000, scale_and_serialize_1000, 0),
        CSV("RPSNR", parse_and_scale_1000, scale_and_serialize_1000, 0),
        CSV("RPSNRyuv", parse_and_scale_1000, scale_and_serialize_1000, 0)
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.duration = -1
        self.total_packets = 0
        self.lost_packets = 0
        self.late_packets = 0
        self.total_slices = 0
        self.lost_slices = 0
        self.slice_recovery = 0
        
        self.lost_CUs = 0
        self.damaged_CUs = 0
        self.correct_CUs = 0
        self.total_CUs = 0

        self.vp_count = 0


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
    for pp in PTraceTx.iter_csv_file(qi.pp_trace):
        if pp.buffer != q.buffer:
            continue
        if pp.time_stamp_in_micro_s <= 0:
            q.lost_packets += 1
        elif pp.time_stamp_in_micro_s > (pp.render_timing + max_delay):
            q.late_packets += 1
        q.total_packets += 1

    q.lost_slices = 0
    q.total_slices = 0
    for sp in STraceRx.iter_csv_file(qi.sp_trace):
        if sp.recovery_position < sp.size:
            q.lost_slices += 1
        q.slice_recovery += (sp.recovery_position / sp.size)
        q.total_slices += 1
    
    q.slice_recovery /= q.total_slices

    q.PLoR = q.lost_packets / q.total_packets
    q.PLaR = q.late_packets / q.total_packets
    q.SLR = q.lost_slices / q.total_slices
    
    q.vp_count = 0
    for vp in VTraceRx.iter_csv_file(qi.vp_trace):
        q.correct_CUs += vp.correct_CUs
        q.damaged_CUs += vp.damaged_CUs
        q.lost_CUs += vp.lost_CUs
        q.total_CUs += vp.total_CUs

        q.PSNR += vp.psnr_y
        q.PSNRyuv += vp.psnr_yuv
        q.RPSNR += vp.rpsnr_y
        q.RPSNRyuv += vp.rpsnr_yuv
        q.vp_count += 1
    
    if q.total_CUs == 0:
        print( "#" * 96, "# ERROR; 0 CUs\n", qi.__dict__, "\n", q.__dict__, "\n#")
    else:
        q.CAR = q.correct_CUs / q.total_CUs
        q.DAR = q.damaged_CUs /q.total_CUs
        q.ALR = q.lost_CUs / q.total_CUs
        q.LDR = q.ALR + q.DAR
    
    q.PSNR /= q.vp_count
    q.PSNRyuv /= q.vp_count
    q.RPSNR /= q.vp_count
    q.RPSNRyuv /= q.vp_count
    
    return q