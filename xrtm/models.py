import os
import shutil
import csv
from typing import Any, Iterator, List, Dict, Callable, Tuple
from enum import Enum, IntEnum
import json
import zlib
import math
from pathlib import Path
from collections import OrderedDict
import random
import scipy.stats as stats

import logging as logger

from .feedback import Referenceable, ReferenceableList
from .utils import ConfigException, _requires


INTRA_VARIANCE = 0.1
INTER_VARIANCE = 0.2

class ErrorResilienceMode(IntEnum):
    DISABLED = 0
    PERIODIC_INTRA = 1
    FEEDBACK_BASED = 2
    FEEDBACK_BASED_ACK = 3
    FEEDBACK_BASED_NACK = 4
    
    @classmethod
    def get(cls, i:int):
        for e in cls.__members__.values():
            if e.value == i:
                return e

class Delay:

    CONSTANT = "constant"
    EQUALLY = "equally"
    GAUSSIANTRUNC = "gaussiantrunc"

    def __init__(self, **params):
        self.mode = str(params.get("mode", "constant")).lower()
        assert self.mode in [self.CONSTANT, self.EQUALLY, self.GAUSSIANTRUNC]
        # timestamps are expressed in micro seconds
        self.parameter1 = float(params.get("parameter1", 0)) * 1e3
        self.parameter2 = float(params.get("parameter2", 0)) * 1e3
        self.parameter3 = float(params.get("parameter3", 0)) * 1e3

    def get_delay(self):
        if self.mode == self.CONSTANT:
            return self.parameter1
        
        if self.mode == self.EQUALLY:
            return random.uniform(self.parameter1, self.parameter2)
        
        # if self.mode == self.GAUSSIANTRUNC:
        #     mean, std = self.parameter1, self.parameter2
        #     lower, upper = mean - self.parameter3, mean + self.parameter3
        #     a, b = (lower - mean) / std, (upper - mean) / std
        #     return stats.truncnorm( a, b, loc=mean, scale=std).rvs()

class EncodingDelay(Delay):
    
    def get_delay(self, no_slices:float, frame_interval_micro_sec:float):
        if self.mode == self.GAUSSIANTRUNC:
            mean, std = self.parameter1 / no_slices, self.parameter2 / no_slices
            lower, upper = 0, frame_interval_micro_sec / no_slices
            a, b = (lower - mean) / std, (upper - mean) / std
            r = stats.truncnorm(a, b, loc=mean, scale=std).rvs()
            return r
        else:
            return super().get_delay()

class EncoderConfig:
    
    def __init__(self):
        self.rc_mode = RC_mode.VBR
        self.rc_bitrate = 10000000
        self.rc_window_size = 1
        self.rc_target_qp = -1
        self.rc_qp_min = -1
        self.rc_qp_max = -1

        self.error_resilience_mode = ErrorResilienceMode.DISABLED
        self.intra_refresh_period = -1
        self.start_time = 0
        self.total_frames = -1
        
        self.buffers = []
        self.frame_width = 2048
        self.frame_height = 2048
        self.frame_rate = 60.0
        self.slices_per_frame = 1

        self.buffer_interleaving = True
        self._pre_delay = None
        self._encoding_delay = None

        # CU map config
        self.cu_size = 64
        
        # slices/CU refs
        self.strace_output = './S-Trace.csv'

    def get_cu_per_frame(self):
        return int(( self.frame_width * self.frame_height ) / self.cu_size**2 )

    def get_cu_per_slice(self):
        return int(( self.frame_width / self.cu_size ) * self.frame_height / ( self.slices_per_frame * self.cu_size ))

    def get_frame_duration(self, unit=1e6) -> float:
        return unit / self.frame_rate

    def get_pre_delay(self):
        if self._pre_delay == None:
            return 0
        return self._pre_delay.get_delay()

    def get_encoding_delay(self):
        if self._encoding_delay == None:
            return 0
        no_slices = len(self.buffers) * self.slices_per_frame
        return self._encoding_delay.get_delay(no_slices, self.get_frame_duration())
    
    def get_buffer_delay(self, buff_idx:int):
        return 0 if buff_idx == 0 else self.get_frame_duration() / len(self.buffers)

    def get_frames_dir(self, user_idx=-1, mkdir=False, overwrite=False):
        p = Path(self.strace_output)
        if user_idx < 0:
            p = p.parent / f'{p.stem}.frames/'
        else:
            p = p.parent / f'{p.stem}[{str(user_idx)}].frames/'
        if mkdir:
            if overwrite and p.exists() and p.is_dir:
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=False)
        return p

    def get_strace_output(self, user_idx=-1):
        p = Path(self.strace_output)
        if user_idx < 0:
            return p
        else:
            return p.parent / f'{p.stem}[{str(user_idx)}]{p.suffix}'

    @classmethod
    def parse_config(cls, data:dict):
        cfg = cls()
        ########################################
        # Rate config
        rc = data.get("Bitrate", { "mode": "VBR" })
        if rc["mode"] == "VBR":
            cfg.rc_mode = RC_mode.VBR
            cfg.rc_target_qp = int(rc.get("QP", -1))
        elif rc["mode"] == "cVBR":
            cfg.rc_mode = RC_mode.cVBR
            _requires("bitrate", rc, "cVBR requires an explicit bitrate")
            cfg.rc_bitrate = int(rc["bitrate"]) # bits/s
            cfg.rc_window_size = int(rc.get("window_framerate", 1)) # frames
            cfg.rc_qp_min = int(rc.get("QPmin", -1)) # default: -1
            cfg.rc_qp_max = int(rc.get("QPmax", -1)) # optional
        elif rc["mode"] == "CBR":
            cfg.rc_mode = RC_mode.CBR
            _requires("bitrate", rc, "CBR requires an explicit bitrate")
            cfg.rc_bitrate = int(rc["bitrate"])
            cfg.rc_window_size = int(rc.get("window_framerate", 1))
            cfg.rc_qp_min = int(rc.get("QPmin", -1))
            cfg.rc_qp_max = int(rc.get("QPmax", -1))
        elif rc["mode"] == "fVBR":
            """
            ◦ fVBR: Feedback-based Variable Bitrate 
                ▪ Details are tbd
            """
            cfg.rc_mode = RC_mode.VBR
        ########################################
        # Slice config
        slicing = data.get("Slice", { "mode": "no" })
        if slicing["mode"] == "no":
            cfg.slices_per_frame = 1
        elif slicing["mode"] == "fixed":
            cfg.slices_per_frame = int(slicing.get("parameter", 8))
        elif slicing["mode"] == "max":
            raise ConfigException("Slice mode 'max' not implemented")
        ########################################
        # Error resilience
        erm = data.get("ErrorResilience", { "mode": "no" })
        if erm["mode"] == "no":
            cfg.error_resilience_mode = ErrorResilienceMode.DISABLED
            cfg.intra_refresh_period = -1
        elif erm["mode"] == "pIntra":
            cfg.error_resilience_mode = ErrorResilienceMode.PERIODIC_INTRA
            _requires("parameter", erm, "missing required parameter for ErrorResilience mode = pIntra")
            cfg.intra_refresh_period = int(erm["parameter"])
        elif erm["mode"] in ["fIntra", "ACK", "NACK"]:
            cfg.error_resilience_mode = ErrorResilienceMode.FEEDBACK_BASED
            cfg.intra_refresh_period = -1
            print("feedback mode under development")
        ########################################
        # Delay config 
        pre_default = {
            "mode": "equally",
            "parameter1": 10,
            "parameter2": 30
        }
        cfg._pre_delay = Delay(**data.get("PreDelay", pre_default))
        
        encoding_default = {
            "mode": "GaussianTrunc",
            "parameter1": 8,
            "parameter2": 3,
            "parameter3": 1
        }
        cfg._encoding_delay = EncodingDelay(**data.get("EncodingDelay", encoding_default))
        cfg.buffers = data.get("Buffers", [])
        if len(cfg.buffers) == 0 or len(cfg.buffers) > 2:
            raise ConfigException("invalid Buffers configuration")
        ########################################
        # Buffers/Frames
        for i, buff in enumerate(cfg.buffers):
            _requires("V-Trace", buff, f'"V-Trace" not defined on buffer {i}')
            p = buff["V-Trace"]
            assert Path(p).exists(), f'V-Trace file not found {p}'
        cfg.frame_width = data.get("frame_width", 2048)
        cfg.frame_height = data.get("frame_height", 2048)
        cfg.start_frame = data.get("start_frame", 0)
        cfg.total_frames = data.get("total_frames", -1)
        cfg.buffer_interleaving = data.get("buffer_interleaving", False)
        cfg.cu_size = data.get("cu_size", 64)
        _requires("S-Trace", data, f'"S-Trace" output path definition missing')
        cfg.strace_output = data["S-Trace"]
        return cfg

    @classmethod
    def load(cls, p:Path) -> 'EncoderConfig':
        with open(p, 'r') as f:
            cfg = json.load(f)
            return cls.parse_config(cfg)


#################################################################################################################################
# CSV PARSING

def parse_list(cls=str, separator=",") -> Callable:
    def parse(s):
        if s == '':
            return []
        else:
            return [cls(item) for item in s.split(separator)]
    return parse

def serialize_list(l:List[Any], separator=",") -> str:
    return str(separator).join([str(i) for i in l])

def parse_and_scale(div=100):
    return lambda x: None if x in ['None', ''] else float(x) / div

def scale_and_serialize(mul=100):
    return lambda x: x if x == None else str(int(x*mul))

parse_and_scale_100 = parse_and_scale(100)
scale_and_serialize_100 = scale_and_serialize(100)
parse_and_scale_1000 = parse_and_scale(1000)
scale_and_serialize_1000 = scale_and_serialize(1000)
    

class CSV:

    def __init__(self, name:str, parse:Callable=None, serialize:Callable=None, default:Any=None):
        self.name = name
        self.default = default
        self._parse = parse
        self._serialize = serialize
    
    def parse(self, data:str) -> Any:
        if self._parse != None:
            return self._parse(data)
        if not data:
            data = self.default
        return data

    def serialize(self, data:Any) -> str:
        if data == None:
            data = self.default
        if self._serialize != None:
            return self._serialize(data)
        return str(data)


class CsvRecord:

    attributes:List[CSV] = []

    def __init__(self, data:dict):
        for h in self.attributes:
            setattr(self, h.name, data.get(h.name, h.default))

    @classmethod
    def from_csv_row(cls, row:dict):
        data = {}
        for h in cls.attributes:
            data[h.name] = h.parse(row.get(h.name, None))
        return cls(data)

    @classmethod
    def get_csv_fieldnames(cls) -> List[str]:
        return [h.name for h in cls.attributes]

    def get_csv_dict(self) -> dict:
        r = {}
        for h in self.attributes:
            r[h.name] = h.serialize(getattr(self, h.name, h.default))
        return r

    @classmethod
    def iter_csv_file(cls, fp:Path):
        with open(fp, newline='') as f:
            vtrace_reader = csv.DictReader(f)
            for row in vtrace_reader:
                yield cls.from_csv_row(row)

    @classmethod
    def get_csv_writer(cls, f, writeheaders=True):
        w =  csv.DictWriter(f, fieldnames=cls.get_csv_fieldnames())
        if writeheaders:
            w.writeheader()
        return w


#################################################################################################################################
# model attributes

class CsvEnum(Enum):

    @classmethod
    def parse(cls, t:str) -> Enum:
        for T in cls.__members__.values():
            if t == str(T.value):
                return T
        raise ValueError(f'parser error - invalid data: "{t}"')

    @classmethod
    def serialize(cls, t:Enum) -> str:
        return str(t.value)


class SliceType(CsvEnum):
    IDR = 1
    P = 2

class CU_mode(CsvEnum):
    UNDEFINED = 0
    INTRA = 1
    INTER = 2
    MERGE = 3
    SKIP = 4

class CU_status(CsvEnum):
    OK              = 0
    DAMAGED         = 1
    UNAVAILABLE     = 2


class CuMap:
    
    def __init__(self, count:int):
        self.count = count
        self._map:List[CU] = [None] * count

    def count(self) -> List[int]:
        intra, inter, merge, skip = [0] * 4
        for cu in self._map:
            if CU_mode.INTRA:
                intra += 1
            elif CU_mode.INTER:
                inter += 1
            elif CU_mode.MERGE:
                merge += 1
            elif CU_mode.SKIP:
                skip += 1
            else:
                raise ValueError('invalid CU type')
        return intra, inter, merge, skip

    @classmethod
    def draw_ctus(cls, count:int, weights:List[float]) -> List['CU']:
        return [CU({'mode':m}) for m in random.choices( 
            [CU_mode.INTRA, CU_mode.INTER, CU_mode.SKIP, CU_mode.MERGE], 
            weights=weights, 
            k=count)]

    @classmethod
    def draw_intra_ctus(cls, count:int) -> List['CU']:
        return [CU({'mode':CU_mode.INTRA}) for i in range(count)]

    def draw(self, *args, **kwargs):
        self._map = self.draw_ctus(*args, **kwargs)

    def get_slice(self, index:int, count:int) -> List['CU']:
        stop = index + count
        return self._map[index:stop]

    def update_slice(self, i:int, m:List['CU']):
        stop = i + len(m)
        assert stop <= self.count
        self._map[i:stop] = m

    def dump(self, csv_out:str):
        with open(csv_out, 'w') as f:
            writer = CU.get_csv_writer(f)
            for idx, cu in enumerate(self._map):
                cu.address = idx
                writer.writerow(cu.get_csv_dict())

    def load(self, csv_in:str) -> 'CuMap':
        data = []
        with open(csv_in, 'r') as f:
            data = [ctu for ctu in CU.iter_csv_file(csv_in)]
        assert len(data) == self.count
        self._map = data

#################################################################################################################################
# models

class VTraceTx(CsvRecord):

    attributes = [
            CSV("time_stamp_in_micro_s", int, None, -1),
            CSV("encode_order", int),
            CSV("i_qp", parse_and_scale_100, scale_and_serialize_100),
            CSV("i_bits", int),
            CSV("i_y_psnr", parse_and_scale_1000, scale_and_serialize_1000),
            CSV("i_u_psnr", parse_and_scale_1000, scale_and_serialize_1000),
            CSV("i_v_psnr", parse_and_scale_1000, scale_and_serialize_1000),
            CSV("i_yuv_psnr", parse_and_scale_1000, scale_and_serialize_1000),
            CSV("i_ssim", int),
            CSV("i_ssim_db", int),
            CSV("i_total_frame_time_ms", int),
            CSV("p_poc", int),
            CSV("p_qp", parse_and_scale_100, scale_and_serialize_100),
            CSV("p_bits", int),
            CSV("p_y_psnr", parse_and_scale_1000, scale_and_serialize_1000),
            CSV("p_u_psnr", parse_and_scale_1000, scale_and_serialize_1000),
            CSV("p_v_psnr", parse_and_scale_1000, scale_and_serialize_1000),
            CSV("p_yuv_psnr", parse_and_scale_1000, scale_and_serialize_1000),
            CSV("p_ssim", int),
            CSV("p_ssim_db", int),
            CSV("p_total_frame_time_ms", int),
            CSV("intra", parse_and_scale_100, scale_and_serialize_100),
            CSV("merge", parse_and_scale_100, scale_and_serialize_100),
            CSV("skip", parse_and_scale_100, scale_and_serialize_100),
            CSV("inter", parse_and_scale_100, scale_and_serialize_100)
    ]

    def get_intra_mean(self, cu_count):
       return self.i_bits / cu_count
    
    def get_inter_mean(self, cu_count):
        if self.intra == 1.:
            return 0.
        intra_mean = self.i_bits / cu_count
        inter_bits = self.p_bits - (self.intra * intra_mean) - (self.skip * 8)
        inter_ctu_count = cu_count * (1 - self.intra - self.skip)
        return inter_bits / inter_ctu_count

    def get_psnr_ref(self, slice_type:SliceType) -> dict:
        if slice_type == SliceType.IDR:
            return {
                'psnr_y': self.i_y_psnr,
                'psnr_u': self.i_u_psnr,
                'psnr_v': self.i_v_psnr,
                'psnr_yuv': self.i_yuv_psnr
            }
        elif slice_type == SliceType.P:
            return {
                'psnr_y': self.p_y_psnr,
                'psnr_u': self.p_u_psnr,
                'psnr_v': self.p_v_psnr,
                'psnr_yuv': self.p_yuv_psnr
            }
        raise ValueError('invalid argument: slice_type')

    def get_encoding_time(self, slice_type:SliceType):
        if slice_type == SliceType.IDR:
            return self.i_total_frame_time_ms
        elif slice_type == SliceType.P:
            return self.p_total_frame_time_ms
        raise ValueError('invalid argument: slice_type')
        

    def get_cu_distribution(self):
        return self.intra, self.inter, self.skip, self.merge

    @property
    def is_full_intra(self):
        return self.intra == 1.

def validates_cu_distribution(vt:VTraceTx, raise_exception=True) -> bool:
    total = vt.intra + vt.inter + vt.skip + vt.merge
    valid = round(total * 100) == 100
    if not valid:
        e_msg = f'Invalid CU distribution on frame {vt.encode_order} - sum of all CUs is {total}%'
        if raise_exception:
            raise ValueError(e_msg)
        logger.critical(e_msg)
    return valid


class VTraceRx(VTraceTx):
    """
    TODO: implement V'Trace
    """
    pass


class STraceTx(CsvRecord):

    attributes = [
        CSV("index", int, None),
        CSV("time_stamp_in_micro_s", int, None, -1),
        CSV("size", int, None),
        CSV("render_timing", int),
        CSV("buffer", int, None),
        CSV("frame_index", int, None),
        CSV("type", SliceType.parse, SliceType.serialize),
        CSV("importance", int, None, -1),
        CSV('start_address_cu', int, None),
        CSV('number_cus', int, None),
        CSV('frame_file', None, None)
    ]

    @classmethod
    def from_slice(cls, s:'Slice') -> 'STraceTx':
        st = cls({})
        st.time_stamp_in_micro_s = s.time_stamp_in_micro_s
        st.frame_index = s.frame_idx
        st.index = -1
        st.size = s.size
        st.buffer = s.view_idx
        st.render_timing = s.render_timing
        st.type = s.slice_type
        st.importance = s.importance
        st.start_address_cu = s.cu_address
        st.number_cus = s.cu_count
        st.buffer = s.view_idx
        st.frame_file = None
        return st

class STraceRx(STraceTx):
    """
    TODO: implement S'Trace
    """
    pass


class CU(CsvRecord):
    
    attributes = [
        CSV("address", int), # Address of CU in frame.
        CSV("size", int), # Slice size in bytes.
        CSV("mode", CU_mode.parse, CU_mode.serialize, None), # The mode of the CU 1=intra, 2=merge, 3=skip, 4=inter
        CSV("reference", lambda x: None if x == 'None' else int(x)), # The reference frame of the CU 0 n/a, 1=previous, 2=2 in past, etc.
        CSV("qpnew", parse_and_scale_100, scale_and_serialize_100, None), # the QP decided for the CU
        CSV("psnr_y", parse_and_scale_1000, scale_and_serialize_1000), # the estimated Y-PSNR for the CU in db multiplied by 1000
        CSV("psnr_yuv", parse_and_scale_1000, scale_and_serialize_1000) # the estimated weighted YUV-PSNR for the CU db multiplied by 1000
    ]

class PTraceTx(CsvRecord):
    
    attributes = [
            CSV("number", int, None),
            CSV("time_stamp_in_micro_s", int, None, -1),
            CSV("size", int, None),
            CSV("user_id", int, None),
            CSV("buffer", int, None),
            CSV("delay", int, None),
            CSV("render_timing", int),
            CSV("number_in_unit", int, None),
            CSV("last_in_unit", bool, lambda b: 1 if b else 0),
            CSV("type", SliceType.parse, SliceType.serialize, None),
            CSV("importance", int, None, -1),
            CSV("index", int, None),
            CSV("s_trace")
    ]
    
    def is_fragment(self) ->  bool:
        return not (self.number_in_unit == 0 and self.last_in_unit)

    @classmethod
    def from_strace(cls, s:STraceTx, **kwargs) -> 'PTraceTx':
        p = cls(kwargs)
        p.importance = s.importance
        p.render_timing = s.render_timing
        p.buffer = s.buffer
        p.type = s.type
        p.time_stamp_in_micro_s = s.time_stamp_in_micro_s
        return p


class PTraceRx(CsvRecord):
    attributes = [
        *PTraceTx.attributes,
        CSV("inter_arrival_micro_s", int),
    ]


#################################################################################################################################

class RCtrace(CsvRecord):
    attributes = [
        CSV("window_bits", float),
        CSV("window_rate", float),
        CSV("i_qp", float),
        CSV("i_mean", float),
        CSV("i_yuv_psnr", float),
        CSV("p_qp", float),
        CSV("p_mean", float),
        CSV("p_yuv_psnr", float),

        CSV("intra", float),
        CSV("inter", float),
        CSV("merge", float),
        CSV("skip", float)
    ]

class RC_mode(Enum):
    VBR = 0
    cVBR = 1
    CBR = 2
    fVBR = 3

class RateControl:

    def __init__(self, cfg:EncoderConfig):
        self.mode = cfg.rc_mode
        self.bitrate = cfg.rc_bitrate
        self.frame_rate = cfg.frame_rate
        self.target_qp = cfg.rc_target_qp
        self.qp_min = max(0, cfg.rc_qp_min)
        self.qp_max = min(51, cfg.rc_qp_max)
        self._buffer_count = len(cfg.buffers)
        self._frame_bits_ref = (self.bitrate / self._buffer_count) / self.frame_rate
        self._w = []
        self._log = []
        self._factor = 2
        self._w_count_max = cfg.rc_window_size

    @property
    def _w_bits(self):
        return sum(self._w)

    @property
    def _w_count(self):
        return min(len(self._w)+1, self._w_count_max)

    def get_frame_budget(self) -> int:
        """
        available_bits = (self._frame_bits_ref * self._w_count) - self._w_bits
        offset = (available_bits - self._frame_bits_ref) / (self._w_count_max / self._factor)
        return self._frame_bits_ref + offset
        """
        overflow = self._w_count_max * self._frame_bits_ref - sum(self._w[1:])
        bits = self._frame_bits_ref
        if len(self._w) > 0:
            available = (self._frame_bits_ref * len(self._w) - sum(self._w)) / len(self._w)
            offset = available * self._factor * len(self._w) / self._w_count_max
            bits += offset
        return min(bits, overflow)

    def no_overflow(self, bits) -> bool:
        r = sum(self._w[1:]) + bits
        r /= (self._w_count_max * self._frame_bits_ref)
        return r < 1
    
    def _clamp_qp(self, qp):
        if self.qp_min != -1 and qp < self.qp_min:
            return self.qp_min, True
        if self.qp_max != -1 and qp > self.qp_max:
            return self.qp_max, True
        return qp, False

    def weighted_qp(self, frame:'Frame', offset=0):
        i_qp = frame.i_qp + offset
        p_qp = frame.p_qp + offset
        vt = frame._vtrace
        return (i_qp*vt.intra + p_qp*(1-vt.intra))

    def iter_qp_adjustments(self, frame:'Frame', step=1):
        assert self.mode in [RC_mode.cVBR, RC_mode.CBR]
        i_qp_new = frame.i_qp + step
        p_qp_new = frame.p_qp + step
        # offset = step
        while True:
            i_qp_new, i_clamped = self._clamp_qp(i_qp_new)
            p_qp_new, p_clamped = self._clamp_qp(p_qp_new)
            yield i_qp_new, p_qp_new
            # qp, clamped = self._clamp_qp(self.weighted_qp(frame, offset))
            # offset += step
            # yield qp, qp
            if i_clamped and p_clamped:
                print('target QP range exceeded')
                return
            i_qp_new += step
            p_qp_new += step
    
    def estimate_qp(self, frame:'Frame'):
        # qp_init = self.weighted_qp(frame, 0)
        # size = frame.predict_frame_bits(qp_init, qp_init)
        size = frame.predict_frame_bits()
        budget = self.get_frame_budget()
        i_qp, p_qp = frame.i_qp, frame.p_qp
            
        if self.mode == RC_mode.cVBR and (size <= budget):
            return i_qp, p_qp
        
        elif self.mode == RC_mode.CBR and (size <= budget):
            for i, p in self.iter_qp_adjustments(frame, step=-1):
                size = frame.predict_frame_bits(i_qp=i, p_qp=p)
                if size <= budget:
                    i_qp, p_qp = i, p
                else:
                    break
            return i_qp, p_qp

        elif self.mode in [RC_mode.cVBR, RC_mode.CBR] and (size > budget):
            for i_qp, p_qp in self.iter_qp_adjustments(frame, step=1):
                size = frame.predict_frame_bits(i_qp=i_qp, p_qp=p_qp)
                if size <= budget:
                    return i_qp, p_qp

        assert False, f'RC error {self.mode} | frame:{size} | budget:{budget}'

    def add_frame_bits(self, bits:int):
        self._w.append(bits)
        self._w = self._w[-self._w_count_max:]


def model_pnsr_adjustment(qp_new, qp_ref, psnr):
    qp_delta = qp_new - qp_ref
    return psnr + qp_delta

class Slice(Referenceable):
    
    def __init__(self, frame:'Frame', slice_idx:int):
        
        self._frame = frame
        # assuming all slices in frame have an equal cu count
        self.slice_idx = slice_idx
        self.cu_count = frame.cu_per_slice
        self.cu_address = frame.cu_per_slice * slice_idx
        self.slice_type = None

        self.time_stamp_in_micro_s = -1
        self.render_timing = -1

        self._referenceable = True

        self.importance = -1
        self.qp_ref = -1
        self.qp = -1
        self.size = -1
        self.y_psnr = -1
        self.u_psnr = -1
        self.v_psnr = -1
        self.yuv_psnr = -1

    @property
    def frame(self):
        return self._frame
    
    @property
    def view_idx(self):
        return self._frame.view_idx
    
    @property
    def frame_idx(self):
        return self._frame.frame_idx

    def get_referenceable_status(self):
        return self._referenceable

    def set_referenceable_status(self, status:bool):
        self._referenceable = status
    
    referenceable = property(get_referenceable_status, set_referenceable_status)


class Frame:

    def __init__(self, 
            vtrace:VTraceTx,
            cfg:EncoderConfig,
            view_idx=0,
            frame_idx=-1
        ):
        self.frame_idx = frame_idx
        self.view_idx = view_idx
        self.slices_per_frame = cfg.slices_per_frame
        self.cu_per_slice = cfg.get_cu_per_slice()
        self.cu_count = cfg.get_cu_per_frame()
        self.cu_size = cfg.cu_size

        self.intra_mean = vtrace.get_intra_mean(self.cu_count)
        self.inter_mean = vtrace.get_inter_mean(self.cu_count)
        self.i_qp = vtrace.i_qp
        self.i_y_psnr = vtrace.i_y_psnr
        self.i_yuv_psnr = vtrace.i_yuv_psnr
        
        self.p_qp = vtrace.p_qp
        self.p_y_psnr = vtrace.p_y_psnr
        self.p_yuv_psnr = vtrace.p_yuv_psnr

        self.cu_distribution = vtrace.get_cu_distribution()
        self.cu_map = CuMap(self.cu_count)
        self.slices = [Slice(self, slice_idx) for slice_idx in range(self.slices_per_frame)]

        self._vtrace = vtrace
    
    def draw(self, address:int=0, count:int=-1, intra_refresh=False):
        """
        Draw into the frame's CU map, uses the existing CTU distribution unless `intra_refresh=True`
        :param intra_refresh:  
        """
        if count == -1:
            count = self.cu_count
        if intra_refresh:
            cu_list = CuMap.draw_intra_ctus(count)
        else:
            cu_list = CuMap.draw_ctus(count, self.cu_distribution)
        self.cu_map.update_slice(address, cu_list)

    def predict_frame_bits(self, i_qp:int=-1, p_qp:int=-1):
        qp_adjustment = lambda qp, qp_ref : pow(2, (qp_ref - qp)/6)
        
        if i_qp > 0:
            i_qp_factor = qp_adjustment(i_qp, self.i_qp)
        else:
            i_qp = self.i_qp
            i_qp_factor = 1

        if p_qp > 0:
            p_qp_factor = qp_adjustment(p_qp, self.p_qp)
        else:
            p_qp = self.p_qp
            p_qp_factor = 1

        frame_size = 0
        for s in self.slices:
            for cu in self.cu_map.get_slice(s.cu_address, s.cu_count):
                if cu.mode == CU_mode.SKIP:
                    frame_size += 1
                elif cu.mode != CU_mode.INTRA:
                    frame_size += math.ceil(random.gauss(self.inter_mean, self.inter_mean * 0.2) * p_qp_factor / 8)
                else:
                    assert cu.mode == CU_mode.INTRA
                    frame_size += math.ceil(random.gauss(self.intra_mean, self.intra_mean * 0.1) * i_qp_factor / 8)
        return int(frame_size * 8)


    def encode(self, refs:'RefPicList', i_qp:int=-1, p_qp:int=-1):
        """
        Iterates through the frame's CU map and set the CU's `reference` and `size` properties. w/ inter_cu_variance = 0.2, intra_cu_variance = 0.1
        """
        qp_adjustment = lambda qp, qp_ref : pow(2, (qp_ref - qp)/6)
        
        if i_qp > 0:
            i_qp_factor = qp_adjustment(i_qp, self.i_qp)
            i_y_psnr = model_pnsr_adjustment(i_qp, self.i_qp, self.i_y_psnr)
            i_yuv_psnr = model_pnsr_adjustment(i_qp, self.i_qp, self.i_yuv_psnr)
        else:
            i_qp = self.i_qp
            i_qp_factor = 1
            i_y_psnr = self.i_y_psnr
            i_yuv_psnr = self.i_yuv_psnr

        if p_qp > 0:
            p_qp_factor = qp_adjustment(p_qp, self.p_qp)
            p_y_psnr = model_pnsr_adjustment(p_qp, self.p_qp, self.p_y_psnr)
            p_yuv_psnr = model_pnsr_adjustment(p_qp, self.p_qp, self.p_yuv_psnr)
        else:
            p_qp = self.p_qp
            p_qp_factor = 1
            p_y_psnr = self.p_y_psnr
            p_yuv_psnr = self.p_yuv_psnr

        frame_size = 0

        for s in self.slices:
            rpl = refs.get_rpl(s)
            slice_size = 0
            for cu in self.cu_map.get_slice(s.cu_address, s.cu_count):
                if cu.mode == CU_mode.SKIP:
                    cu.reference = rpl[0]
                    cu.size = 1
                elif cu.mode != CU_mode.INTRA:
                    cu.qpnew = p_qp
                    cu.psnr_y = p_y_psnr
                    cu.psnr_yuv = p_yuv_psnr
                    cu.reference = rpl[0]
                    cu.size = math.ceil(random.gauss(self.inter_mean, self.inter_mean * 0.2) * p_qp_factor / 8)
                else:
                    assert cu.mode == CU_mode.INTRA
                    cu.qpnew = i_qp
                    cu.psnr_y = i_y_psnr
                    cu.psnr_yuv = i_yuv_psnr
                    cu.reference = None
                    cu.size = math.ceil(random.gauss(self.intra_mean, self.intra_mean * 0.1) * i_qp_factor / 8)
                slice_size += cu.size
            s.size = slice_size
            frame_size += slice_size 
        return frame_size

class RefPicList(ReferenceableList):
    
    def __init__(self):
        self.frames = []

    def is_empty(self) -> bool:
        return len(self.frames) == 0

    def push(self, pic:'Frame'):
        self.frames.append(pic)
        self.drop_unreferenced_frames()

    def drop_unreferenced_frames(self):
        """
        drop old frames that are beyond I slices (IDR):
        """
        if len(self.frames) == 0:
            return
        delta = self.get_largest_intra_delta() + 1
        self.frames = self.frames[-delta:]

    def get_previous_intra_frame(self, s:Slice) -> 'Frame':
        for frame in reversed(self.frames):
            if frame.slices[s.slice_idx].slice_type == SliceType.IDR:
                return frame
        return None

    def get_previous_intra_delta(self, s:Slice) -> int:
        """
        Assuming slice cu_address & cu_count does not vary, Feedback not taken into account.
        :Return: how many frames since prev intra slice, -1 if None were found
        """
        assert s.frame == self.frames[-1], 'slice s must be slice of the latest buffered frame'
        frame = self.get_previous_intra_frame(s)
        assert frame != None
        delta = s.frame_idx - frame.frame_idx
        assert delta >= 0
        return delta

    def get_largest_intra_delta(self):
        assert len(self.frames) > 0, 'referenceable frame buffer is empty'
        largest_intra_delta = -1
        for s in self.frames[-1].slices:
            slice_intra_delta = self.get_previous_intra_delta(s)
            assert slice_intra_delta >= 0, f'no intra frame found for {s.slice_idx}'
            largest_intra_delta = max(largest_intra_delta, slice_intra_delta)
        return largest_intra_delta

    def iter_referenceable_frames(self, s:Slice) -> Iterator[Tuple[int, int]]:
        """
        iter frames that can be referenced by this slice
        """
        for i, frame in enumerate(reversed(self.frames)):
            if frame.slices[s.slice_idx].referenceable:
                yield i, frame.frame_idx
            if frame.slices[s.slice_idx].slice_type == SliceType.IDR:
                return i, None

    def get_rpl(self, s:Slice) -> List[int]:
        return [frame_idx for _, frame_idx in self.iter_referenceable_frames(s)]

    def set_referenceable_status(self, frame_idx:int, slice_idx:int, status:bool):
        for frame in reversed(self.frames):
            if frame.frame_idx == frame_idx:
                frame.slices[slice_idx].referenceable = status
                return

    def reset(self, slice_idx:int=None):
        self.frames = []
