import os
import shutil
import csv
from typing import Any, Iterator, List, Dict, Callable
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

        if self.mode == self.GAUSSIANTRUNC:
            mean, std = self.parameter1, self.parameter2
            lower, upper = mean - self.parameter3, mean + self.parameter3
            a, b = (lower - mean) / std, (upper - mean) / std
            return stats.truncnorm( a, b, loc=mean, scale=std).rvs()

class EncodingDelay(Delay):
    
    def get_delay(self, no_slices:float, frame_interval_ms:float):
        if self.mode == self.GAUSSIANTRUNC:
            mean, std = self.parameter1 / no_slices, self.parameter2 / no_slices
            lower, upper = 0, frame_interval_ms / no_slices
            a, b = (lower - mean) / std, (upper - mean) / std
            r = stats.truncnorm(a, b, loc=mean, scale=std).rvs()
            return r
        else:
            return super().get_delay()

class EncoderConfig:
    
    def __init__(self):
        self.rc_mode = RC_mode.VBR
        self.rc_bitrate = None
        self.rc_window_size = None
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
        self.max_refs = 1
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
        return self._encoding_delay.get_delay(self.slices_per_frame, self.get_frame_duration(unit=1e3))
    
    def get_buffer_delay(self, buff_idx:int):
        """
        the buffer delay to apply when buffers are interleaved
        NOTE:
            * for len(self.buffers) > 2, it is frame_duration / 2
            * behavior has not been discussed for len(self.buffers) > 2
        """
        return 0 if buff_idx == 0 else self.get_frame_duration() / 2

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
            cfg.rc_window_size = int(rc.get("window_size", 1)) # frames
            cfg.rc_qp_min = int(rc.get("QPmin", -1)) # default: -1
            cfg.rc_qp_max = int(rc.get("QPmax", -1)) # optional
        elif rc["mode"] == "CBR":
            cfg.rc_mode = RC_mode.cVBR
            _requires("bitrate", rc, "CBR requires an explicit bitrate")
            cfg.rc_bitrate = int(rc["bitrate"])
            cfg.rc_window_size = int(rc.get("window_size", 1))
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


class XRTM(Enum):
    
    @property
    def name(self):
        return self.value.name

    @property
    def default(self):
        return self.value.default

    def parse(self, data) -> Any:
        return self.value.parse(data)

    def serialize(self, data) -> str:
        return self.value.serialize(data)

    # Session cfg
    USER_ID = CSV("user_id", int, None)
    
    # Generic fielnames
    SIZE = CSV("size", int, None)
    INDEX = CSV("index", int, None)
    EYE_BUFFER = CSV("eye_buffer", int, None)
    TIME_STAMP_IN_MICRO_S = CSV("time_stamp_in_micro_s", int, None, -1)
    
    # VTraceTx
    VIEW_IDX = CSV("view_idx", int, None, -1) # 0: LEFT, 1: RIGHT
    ENCODE_ORDER = CSV("encode_order", int)
    I_QP = CSV("i_qp", int )
    I_BITS = CSV("i_bits", int)
    I_TOTAL_FRAME_TIME_MS = CSV("i_total_frame_time_ms", int)
    I_PSNR_Y = CSV("i_y_psnr", float)
    I_PSNR_U = CSV("i_u_psnr", float)
    I_PSNR_V = CSV("i_v_psnr", float)
    I_PSNR_YUV = CSV("i_yuv_psnr", float)
    I_SSIM = CSV("i_ssim", float)
    I_SSIM_DB = CSV("i_ssim_db", float)
    P_SSIM = CSV("i_ssim", float)
    P_SSIM_DB = CSV("i_ssim_db", float)

    P_QP = CSV("p_qp", float)
    P_BITS = CSV("p_bits", int)
    P_TOTAL_FRAME_TIME_MS = CSV("p_total_frame_time_ms", int)
    P_PSNR_Y = CSV("p_y_psnr", float)
    P_PSNR_U = CSV("p_u_psnr", float)
    P_PSNR_V = CSV("p_v_psnr", float)
    P_PSNR_YUV = CSV("p_yuv_psnr", float)

    PSNR_Y = CSV("y_psnr", float)
    PSNR_U = CSV("u_psnr", float)
    PSNR_V = CSV("v_psnr", float)
    PSNR_YUV = CSV("yuv_psnr", float)

    INTRA = CSV("intra", float)
    INTER = CSV("inter", float)
    SKIP = CSV("skip", float)
    MERGE = CSV("merge", float)

    # PTraceTx
    NUMBER = CSV("number", int, None)
    NUMBER_IN_SLICE = CSV("number_in_slice", int, None)
    LAST_IN_SLICE = CSV("last_in_slice", bool, lambda b: 1 if b else 0)
    DELAY = CSV("delay", int, None)
    S_TRACE = CSV("s_trace", )

    # STraceTx
    FRAME_IDX = CSV("frame_idx", int, None)
    SLICE_IDX = CSV("slice_idx", int)
    TYPE = CSV("type", SliceType.parse, SliceType.serialize)
    INTRA_MEAN = CSV("intra_mean", float)
    INTER_MEAN = CSV("inter_mean", float)
    IMPORTANCE = CSV("importance", int, None, -1)

    RENDER_TIMING = CSV("render_timing", int)
    START_ADDRESS_CU = CSV('start_address_cu', int, None)

    REFS = CSV("refs", parse_list(int), serialize_list)
    NUMBER_CUS = CSV('number_cus', int, None)
    FRAME_FILE = CSV('frame_file', None, None)


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


class CU(CsvRecord):
    
    attributes = [
        CSV("address", int), # Address of CU in frame.
        CSV("mode", CU_mode.parse, CU_mode.serialize, None), # The mode of the CU 1=intra, 2=merge, 3=skip, 4=inter
        CSV("size", int), # Slice size in bytes.
        CSV("reference", int), # The reference frame of the CU 0 n/a, 1=previous, 2=2 in past, etc.
        CSV("qp", int), # the QP decided for the CU
        CSV("psnr_y", int), # the estimated Y-PSNR for the CU in db multiplied by 1000
        CSV("psnr_yuv", int) # the estimated weighted YUV-PSNR for the CU db multiplied by 1000
    ]

    def __init__(self, mode=CU_mode.UNDEFINED, size=-1, ref:List[int]=[], address=-1):
        self.mode = mode
        self.size = size
        self.reference = ref
        self.address = address



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
    def draw_ctus(cls, count:int, weights:List[float]) -> List[CU]:
        return random.choices(
            [CU(CU_mode.INTRA), CU(CU_mode.INTER), CU(CU_mode.SKIP), CU(CU_mode.MERGE)], 
            weights=weights, 
            k=count)

    @classmethod
    def draw_intra_ctus(cls, count:int) -> List[CU]:
        return [CU(CU_mode.INTRA)] * count

    def draw(self, *args, **kwargs):
        self._map = self.draw_ctus(*args, **kwargs)

    def get_slice(self, index:int, count:int) -> List[CU]:
        stop = index + count
        return self._map[index:stop]

    def update_slice(self, i:int, m:List[CU]):
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
            XRTM.TIME_STAMP_IN_MICRO_S,
            XRTM.ENCODE_ORDER,

            XRTM.I_QP,
            XRTM.I_BITS,
            XRTM.I_TOTAL_FRAME_TIME_MS,
            XRTM.I_PSNR_Y,
            XRTM.I_PSNR_U,
            XRTM.I_PSNR_V,
            XRTM.I_PSNR_YUV,

            XRTM.P_QP,
            XRTM.P_BITS,
            XRTM.P_TOTAL_FRAME_TIME_MS,
            XRTM.P_PSNR_Y,
            XRTM.P_PSNR_U,
            XRTM.P_PSNR_V,
            XRTM.P_PSNR_YUV,

            XRTM.INTRA,
            XRTM.INTER,
            XRTM.SKIP,
            XRTM.MERGE
    ]

    def __init__(self, data=None):

        super().__init__(data)
        
        # TODO: clarify expected data range 
        for k in [
            XRTM.I_QP,
            XRTM.P_QP,
            XRTM.INTRA,
            XRTM.INTER,
            XRTM.SKIP,
            XRTM.MERGE
        ]:
            v = getattr(self, k.name) / 100
            setattr(self, k.name, v)

        # TODO: clarify expected data range 
        for k in [
            XRTM.I_PSNR_Y,
            XRTM.I_PSNR_U,
            XRTM.I_PSNR_V,
            XRTM.I_PSNR_YUV,
            XRTM.P_PSNR_Y,
            XRTM.P_PSNR_U,
            XRTM.P_PSNR_V,
            XRTM.P_PSNR_YUV
        ]:
            v = getattr(self, k.name) / 1000
            setattr(self, k.name, v)

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
                XRTM.PSNR_Y.name: self.i_y_psnr,
                XRTM.PSNR_U.name: self.i_u_psnr,
                XRTM.PSNR_V.name: self.i_v_psnr,
                XRTM.PSNR_YUV.name: self.i_yuv_psnr
            }
        elif slice_type == SliceType.P:
            return {
                XRTM.PSNR_Y.name: self.p_y_psnr,
                XRTM.PSNR_U.name: self.p_u_psnr,
                XRTM.PSNR_V.name: self.p_v_psnr,
                XRTM.PSNR_YUV.name: self.p_yuv_psnr
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
        XRTM.INDEX,
        XRTM.TIME_STAMP_IN_MICRO_S,
        XRTM.RENDER_TIMING,
        XRTM.IMPORTANCE,
        XRTM.TYPE,
        XRTM.SIZE,
        XRTM.START_ADDRESS_CU,
        XRTM.NUMBER_CUS,
        XRTM.EYE_BUFFER,
        XRTM.FRAME_FILE,
        XRTM.FRAME_IDX
    ]

    @classmethod
    def from_slice(cls, s:'Slice') -> 'STraceTx':
        st = cls({})
        st.time_stamp_in_micro_s = s.time_stamp_in_micro_s
        st.frame_idx = s.frame_idx
        st.index = -1
        st.size = s.size
        st.eye_buffer = s.view_idx
        st.render_timing = s.render_timing
        st.type = s.slice_type
        st.importance = s.importance
        st.start_address_cu = s.cu_address
        st.number_cus = s.cu_count
        st.eye_buffer = s.view_idx
        st.frame_file = None
        return st

class STraceRx(STraceTx):
    pass


class PTraceTx(CsvRecord):
    
    attributes = [
            XRTM.USER_ID,
            XRTM.IMPORTANCE,
            XRTM.NUMBER,
            XRTM.NUMBER_IN_SLICE,
            XRTM.LAST_IN_SLICE,
            XRTM.SIZE,
            XRTM.TIME_STAMP_IN_MICRO_S,
            XRTM.DELAY,
            XRTM.INDEX,
            XRTM.EYE_BUFFER,
            XRTM.TYPE,
            XRTM.RENDER_TIMING,
            XRTM.S_TRACE
    ]
    
    def is_fragment(self) ->  bool:
        return not (self.number_in_slice == 0 and self.last_in_slice)

    @classmethod
    def from_strace(cls, s:STraceTx, **kwargs) -> 'PTraceTx':
        p = cls(kwargs)
        p.importance = s.importance
        p.render_timing = s.render_timing
        p.eye_buffer = s.eye_buffer
        p.type = s.type
        p.time_stamp_in_micro_s = s.time_stamp_in_micro_s
        return p


class PTraceRx(PTraceTx):
    """
    TODO: implement P'Trace
    """
    pass


#################################################################################################################################

class RC_mode(Enum):
    VBR = 0
    cVBR = 1
    CBR = 2
    fVBR = 3

class RateControl:

    def __init__(self, cfg:EncoderConfig):
        self.mode = cfg.rc_mode
        
        # for now, assuming bitrate is split equally between all buffers 
        self.bitrate = cfg.rc_bitrate
        self.frame_rate = cfg.frame_rate
        self.window_duration = cfg.rc_window_size * cfg.get_frame_duration()
        self._buffer_count = len(cfg.buffers)
        self._target_buffer_bitrate = self.bitrate / self._buffer_count
        self._window = []

        self.target_qp = cfg.rc_target_qp
        self.qp_min = cfg.rc_qp_min
        self.qp_max = cfg.rc_qp_max
    
    def get_frame_budget(self) -> int:
        return self.bitrate / self.frame_rate
    


def model_pnsr_adjustment(qp_new, qp_ref, psnr):
    qp_delta = qp_new - qp_ref
    return psnr + qp_delta

class Slice(Referenceable):
    
    def __init__(self, frame:'Frame', slice_idx:int):
        
        self._frame = frame
        self.view_idx = frame.view_idx
        self.frame_idx = frame.frame_idx
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

    def get_referenceable_status(self):
        return self._referenceable

    def set_referenceable_status(self, status:bool):
        self._referenceable = status
    
    referenceable = property(get_referenceable_status, set_referenceable_status)

class RefPicList(ReferenceableList):
    
    def __init__(self, max_size:int=16):
        self.max_size = max_size
        self.frames = []

    def is_empty(self) -> bool:
        return len(self.frames) == 0

    def push(self, pic:'Frame'):
        self.frames.append(pic)
        self.frames = self.frames[-self.max_size:]

    def get_rpl(self, slice_idx:int) -> List[int]:
        return [*self.iter(slice_idx)]

    def find_intra(self, slice_idx:int) -> int:
        for frame in reversed(self.frames):
            if frame.slices[slice_idx].slice_type == SliceType.IDR:
                return frame.frame_idx
        return None

    def iter(self, slice_idx:int) -> Iterator[int]:
        for frame in reversed(self.frames):
            if frame.slices[slice_idx].referenceable:
                yield frame.frame_idx
            if frame.slices[slice_idx].slice_type == SliceType.IDR:
                return

    def set_referenceable_status(self, frame_idx:int, slice_idx:int, status:bool):
        for frame in reversed(self.frames):
            if frame.frame_idx == frame_idx:
                frame.slices[slice_idx].referenceable = status
                return

    def reset(self, slice_idx:int=None):
        self.frames = []

class Frame:

    def __init__(self, 
            vtrace:VTraceTx,
            cfg:EncoderConfig,
            view_idx=0
        ):
        self.view_idx = view_idx
        self.slices_per_frame = cfg.slices_per_frame
        self.cu_per_slice = cfg.get_cu_per_slice()
        self.cu_count = cfg.get_cu_per_frame()
        self.cu_size = cfg.cu_size

        self.frame_idx = vtrace.encode_order
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

    def encode(self, qp:int, refs:RefPicList):
        """
        Iterates through the frame's CU map and set the CU's `reference` and `size` properties. w/ inter_cu_variance = 0.2, intra_cu_variance = 0.1
        """
        qp_adjustment = lambda qp, qp_ref : pow(2, (qp - qp_ref)/6)
        
        i_qp = self.i_qp if qp < 0 else qp
        i_qp_factor = 1 if qp < 0 else qp_adjustment(qp, self.i_qp)
        i_y_psnr = self.i_y_psnr if qp < 0 else model_pnsr_adjustment(qp, self.i_qp, self.i_y_psnr)
        i_yuv_psnr = self.i_yuv_psnr if qp < 0 else model_pnsr_adjustment(qp, self.i_qp, self.i_yuv_psnr)

        p_qp = self.p_qp if qp < 0 else qp
        p_qp_factor = 1 if qp < 0 else qp_adjustment(qp, self.p_qp)
        p_y_psnr = self.p_y_psnr if qp < 0 else model_pnsr_adjustment(qp, self.p_qp, self.p_y_psnr)
        p_yuv_psnr = self.p_yuv_psnr if qp < 0 else model_pnsr_adjustment(qp, self.p_qp, self.p_yuv_psnr)

        frame_size = 0

        for s in self.slices:
            rpl = refs.get_rpl(s.slice_idx)
            slice_size = 0
            for cu in self.cu_map.get_slice(s.cu_address, s.cu_count):
                if cu.mode == CU_mode.SKIP:
                    cu.reference = rpl[0]
                    cu.size = 1
                elif cu.mode != CU_mode.INTRA:
                    cu.qp = p_qp
                    cu.psnr_y = p_y_psnr
                    cu.psnr_yuv = p_yuv_psnr
                    cu.reference = rpl[0]
                    cu.size = math.ceil(random.gauss(self.inter_mean, self.inter_mean * 0.2) * p_qp_factor)
                else:
                    assert cu.mode == CU_mode.INTRA
                    cu.qp = i_qp
                    cu.psnr_y = i_y_psnr
                    cu.psnr_yuv = i_yuv_psnr
                    cu.reference = None
                    cu.size = math.ceil(random.gauss(self.intra_mean, self.intra_mean * 0.1) * i_qp_factor)
                slice_size += cu.size
            s.size = slice_size
            frame_size += slice_size
        return frame_size

