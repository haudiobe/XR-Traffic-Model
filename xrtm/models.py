import os
import csv
from typing import Any, Iterable, List, Dict, Callable
from enum import Enum, IntEnum
import json
import zlib
import math
from pathlib import Path
from collections import OrderedDict
import random

import logging as logger

from .exceptions import VTraceTxException
from .feedback import Referenceable
# from .cu_map import CtuMap

#################################################################################################################################

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
        
        # CU map config
        self.cu_size = data.get('cu_size', 64)
        self.cu_per_frame = int(( self.frame_width * self.frame_height ) / self.cu_size**2 )
        self.cu_per_slice = int(( self.frame_width / self.cu_size ) * self.frame_height / ( self.slices_per_frame * self.cu_size ))
        # slices/CU refs
        self.max_refs = data.get('max_refs', 16)
        self.frames_dir = Path('.')

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
            return cls(data)


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
    PTS = CSV("pts", int)
    ENCODE_ORDER = CSV("encode_order", int)
    CRF_REF = CSV("crf_ref", float)
    QP_REF = CSV("qp_ref", float)
    TYPE = CSV("type", SliceType.parse, SliceType.serialize)

    INTRA_QP_REF = CSV("intra_qp_ref", int )
    INTRA_TOTAL_BITS = CSV("intra_total_bits", int)
    INTRA_TOTAL_TIME = CSV("intra_total_time", int)
    INTRA_PSNR_Y = CSV("intra_y_psnr", float)
    INTRA_PSNR_U = CSV("intra_u_psnr", float)
    INTRA_PSNR_V = CSV("intra_v_psnr", float)
    INTRA_PSNR_YUV = CSV("intra_yuv_psnr", float)

    INTER_QP_REF = CSV("inter_qp_ref", float)
    INTER_TOTAL_BITS = CSV("inter_total_bits", int)
    INTER_TOTAL_TIME = CSV("inter_total_time", int)
    INTER_PSNR_Y = CSV("inter_y_psnr", float)
    INTER_PSNR_U = CSV("inter_u_psnr", float)
    INTER_PSNR_V = CSV("inter_v_psnr", float)
    INTER_PSNR_YUV = CSV("inter_yuv_psnr", float)

    PSNR_Y = CSV("y_psnr", float)
    PSNR_U = CSV("u_psnr", float)
    PSNR_V = CSV("v_psnr", float)
    PSNR_YUV = CSV("yuv_psnr", float)

    CU_INTRA = CSV("ctu_intra_pct", float)
    CU_INTER = CSV("ctu_inter_pct", float)
    CU_SKIP = CSV("ctu_skip_pct", float)
    CU_MERGE = CSV("ctu_merge_pct", float)

    # PTraceTx
    NUMBER = CSV("number", int, None)
    NUMBER_IN_SLICE = CSV("number_in_slice", int, None)
    LAST_IN_SLICE = CSV("last_in_slice", bool, int)
    DELAY = CSV("delay", int, None)
    S_TRACE = CSV("s_trace", )

    # STraceTx
    SLICE_IDX = CSV("slice_idx", int)
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
        CSV("reference", int) # The reference frame of the CU 0 n/a, 1=previous, 2=2 in past, etc.
        # CSV("qp", int) # the QP decided for the CU
        # CSV("psnr_y", int) # the estimated Y-PSNR for the CU in db multiplied by 1000
        # CSV("psnr_yuv", int) # the estimated weighted YUV-PSNR for the CU db multiplied by 1000
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
            # XRTM.VIEW_IDX
            XRTM.PTS,
            XRTM.ENCODE_ORDER,
            XRTM.CRF_REF,

            XRTM.INTRA_QP_REF,
            XRTM.INTRA_TOTAL_BITS,
            XRTM.INTRA_TOTAL_TIME,
            XRTM.INTRA_PSNR_Y,
            XRTM.INTRA_PSNR_U,
            XRTM.INTRA_PSNR_V,
            XRTM.INTRA_PSNR_YUV,

            XRTM.INTER_QP_REF,
            XRTM.INTER_TOTAL_BITS,
            XRTM.INTER_TOTAL_TIME,
            XRTM.INTER_PSNR_Y,
            XRTM.INTER_PSNR_U,
            XRTM.INTER_PSNR_V,
            XRTM.INTER_PSNR_YUV,

            XRTM.CU_INTRA,
            XRTM.CU_INTER,
            XRTM.CU_SKIP,
            XRTM.CU_MERGE
    ]

    def __init__(self, data=None):

        super().__init__(data)
        
        # TODO: clarify expected data range 
        for k in [
            XRTM.INTRA_QP_REF,
            XRTM.INTER_QP_REF,
            XRTM.CU_INTRA,
            XRTM.CU_INTER,
            XRTM.CU_SKIP,
            XRTM.CU_MERGE
        ]:
            v = getattr(self, k.name) / 100
            setattr(self, k.name, v)

        # TODO: clarify expected data range 
        for k in [
            XRTM.INTRA_PSNR_Y,
            XRTM.INTRA_PSNR_U,
            XRTM.INTRA_PSNR_V,
            XRTM.INTRA_PSNR_YUV,
            XRTM.INTER_PSNR_Y,
            XRTM.INTER_PSNR_U,
            XRTM.INTER_PSNR_V,
            XRTM.INTER_PSNR_YUV
        ]:
            v = getattr(self, k.name) / 1000
            setattr(self, k.name, v)

    def get_intra_mean(self, cu_count):
       return self.intra_total_bits / cu_count
    
    def get_inter_mean(self, cu_count):
        if self.ctu_intra_pct == 1.:
            return 0.
        intra_mean = self.intra_total_bits / cu_count
        inter_bits = self.inter_total_bits - (self.ctu_intra_pct * intra_mean) - (self.ctu_skip_pct * 8)
        inter_ctu_count = cu_count * (1 - self.ctu_intra_pct - self.ctu_skip_pct)
        return inter_bits / inter_ctu_count

    def get_psnr_ref(self, slice_type:SliceType) -> dict:
        if slice_type == SliceType.IDR:
            return {
                XRTM.PSNR_Y.name: self.intra_y_psnr,
                XRTM.PSNR_U.name: self.intra_u_psnr,
                XRTM.PSNR_V.name: self.intra_v_psnr,
                XRTM.PSNR_YUV.name: self.intra_yuv_psnr
            }
        elif slice_type == SliceType.P:
            return {
                XRTM.PSNR_Y.name: self.inter_y_psnr,
                XRTM.PSNR_U.name: self.inter_u_psnr,
                XRTM.PSNR_V.name: self.inter_v_psnr,
                XRTM.PSNR_YUV.name: self.inter_yuv_psnr
            }
        raise ValueError('invalid argument: slice_type')

    def get_encoding_time(self, slice_type:SliceType):
        if slice_type == SliceType.IDR:
            return self.intra_total_time
        elif slice_type == SliceType.P:
            return self.inter_total_time
        raise ValueError('invalid argument: slice_type')
    
    def get_qp_ref(self, slice_type:SliceType):
        if slice_type == SliceType.IDR:
            return self.intra_qp_ref
        elif slice_type == SliceType.P:
            return self.inter_qp_ref
        raise ValueError('invalid argument: slice_type')
    

    def get_cu_distribution(self):
        return self.ctu_intra_pct, self.ctu_inter_pct, self.ctu_skip_pct, self.ctu_merge_pct

    @property
    def is_full_intra(self):
        return self.ctu_intra_pct == 1.

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
        # TODO: move these to CU properties ?
        # XRTM.REFERENCE,
        # XRTM.QP,
        # XRTM.PSNR_Y,
        # XRTM.PSNR_U,
        # XRTM.PSNR_V,
        # XRTM.PSNR_YUV,
    ]

    @classmethod
    def from_slice(cls, s:'Slice') -> 'STraceTx':
        st = cls({})
        st.time_stamp_in_micro_s = s.time_stamp_in_micro_s
        st.index = s.slice_idx
        st.size = s.size
        st.eye_buffer = s.view_idx
        st.render_timing = s.render_timing
        st.type = s.slice_type
        st.importance = s.importance
        st.start_address_cu = s.cu_address
        st.number_cus = s.cu_count
        st.frame_file = s.frame_file
        st.eye_buffer = s.view_idx
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

    HEADER_SIZE = 40
    
    def is_fragment(self) ->  bool:
        return not (self.number_in_slice == 0 and self.last_in_slice)

    @classmethod
    def from_strace(cls, s:STraceTx, **kwargs) -> 'PTraceTx':
        p = cls(kwargs)
        p.importance = s.importance
        p.render_timing = s.render_timing
        p.eye_buffer = s.eye_buffer
        p.type = s.slice_type
        p.time_stamp_in_micro_s = s.time_stamp_in_micro_s + p.delay
        return p


class PTraceRx(PTraceTx):
    """
    TODO: implement P'Trace
    """
    pass


#################################################################################################################################

class RateControl:

    def __init__(self, rc_max_bits=-1, crf=-1):
        # from config
        self.max_bits = -1
        self.target_crf = crf
        # to be set on update
        self._crf_ref = -1
        self._i_qp_ref = -1
        self._p_qp_ref = -1

    def update(self, vtrace:VTraceTx):
        #TODO: maintain bitrate over configurable window
        self._i_qp_ref = vtrace.intra_qp_ref
        self._p_qp_ref = vtrace.inter_qp_ref
        self._crf_ref = vtrate.crf_ref

    def adjust(self, size:int, intra:bool):
        qp = self._i_qp_ref if intra else self._p_qp_ref
        if self.max_bits > 0:
            crf_new = self.target_crf if self.target_crf > 0 else self._crf_ref
            bits_new = size
            while bits_new > self.max_bits:
                crf_new += 1
                bits_new, qp_new = model_crf_adjustment(size, crf_new, self._crf_ref, qp)
                if crf_new == 51: 
                    break
            return bits_new, qp_new 

        elif self.target_crf > 0:
            return model_crf_adjustment(size, self.target_crf, self._crf_ref, qp)
            
        else:
            return size, qp

def model_crf_adjustment(bits, CRF, CRFref, qp):
    y = CRFref - CRF
    final_bits = bits * pow(2, y/6)
    QPnew = qp - y
    return final_bits, QPnew

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

    @property
    def frame_file(self):
        return self._frame.frame_file

    def get_referenceable_status(self):
        return self._referenceable

    def set_referenceable_status(self, status:bool):
        self._referenceable = status
    
    referenceable = property(get_referenceable_status, set_referenceable_status)


class Frame:

    def __init__(self, 
            vtrace:VTraceTx,
            cfg:EncoderConfig,
            view_idx=0
        ):
        self.view_idx = view_idx
        self.frame_idx = vtrace.encode_order
        
        self.intra_mean = vtrace.get_intra_mean(cfg.cu_per_frame)
        self.inter_mean = vtrace.get_inter_mean(cfg.cu_per_frame)
        
        self.slices_per_frame = cfg.slices_per_frame
        self.cu_per_slice = cfg.cu_per_slice
        self.cu_count = cfg.cu_per_frame
        self.cu_size = cfg.cu_size
        self.cu_distribution = vtrace.get_cu_distribution()

        self.cu_map = CuMap(self.cu_count)
        self.slices = [Slice(self, slice_idx) for slice_idx in range(self.slices_per_frame)]
    
    @property
    def frame_file(self):
        return f'{self.frame_idx}_{self.view_idx}.csv'

    def draw(self, address:int=0, count:int=-1, intra_refresh=False):
        """
        Draw into the frame's CU map
        follows vtrace's distribution if intra_refresh = False (default)
        """
        if count == -1:
            count = self.cu_count
        if intra_refresh:
            cu_list = CuMap.draw_intra_ctus(count)
        else:
            cu_list = CuMap.draw_ctus(count, self.cu_distribution)
        self.cu_map.update_slice(address, cu_list)

    def encode(self, address:int=0, count:int=-1, rpl:List[int]=[]):
        """
        Iterates through the frame's CU map and set the CU's `reference` and `size` properties. w/ inter_cu_variance = 0.2, intra_cu_variance = 0.1
        Returns:
            size (int) in bytes
        """
        size = 0
        for cu in self.cu_map.get_slice(address, count):
            if cu.mode == CU_mode.SKIP:
                cu.reference = rpl[0]
                cu.size = 1
            elif cu.mode != CU_mode.INTRA and len(rpl) > 0:
                cu.reference = rpl[0]
                cu.size = math.ceil(random.gauss(self.inter_mean, 0.2)/8)
            else:
                assert cu.mode == CU_mode.INTRA
                cu.reference = None
                cu.size = math.ceil(random.gauss(self.intra_mean, 0.1)/8)
            size += cu.size
        return size
