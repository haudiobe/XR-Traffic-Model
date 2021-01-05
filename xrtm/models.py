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

    @staticmethod
    def parse(t:str) -> Enum:
        for T in self.__members__:
            if t == str(T.value):
                return T
        raise ValueError(f'parser error - invalid data: "{t}"')

    @staticmethod
    def serialize(t:Enum) -> str:
        return str(t.value)


class SliceType(CsvEnum):
    IDR = 1
    P = 3


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
    PCKT_SEQNUM = CSV("pckt_seqnum", int, None)
    PCKT_AVAILABILITY = CSV("pckt_availability", int, None)
    PCKT_SIZE = CSV("pckt_size", int, None)
    PCKT_FRAGNUM = CSV("pckt_fragnum", int, None)
    PCKT_IS_LAST = CSV("pckt_is_last", bool, int)
    CN_JITTER_DELAY = CSV("cn_jitter_delay", int, None)

    # STraceTx
    TIME_STAMP_IN_MICRO_S = CSV("time_stamp_in_micro_s", int, None, -1)
    SLICE_IDX = CSV("slice_idx", int)
    INTRA_MEAN = CSV("intra_mean", float)
    INTER_MEAN = CSV("inter_mean", float)
    PRIORITY = CSV("priority", int, None, -1)

    RENDER_TIMING = CSV("render_timing", int)
    START_ADDRESS_CU = CSV('start_address_cu', int, None)

    REFS = CSV("refs", parse_list(int), serialize_list)
    NUMBER_CUS = CSV('number_cus', int, None)
    FRAME_FILE = CSV('frame_file', None, None)


class CTU_mode(CsvEnum):
    INTRA = 0
    INTER = 1
    MERGE = 2
    SKIP = 3

class CTU_status(CsvEnum):
    OK              = 0
    DAMAGED         = 1
    UNAVAILABLE     = 2

class CTU:
    def __init__(self, mode:CTU_mode, size:float, refs:List[int]=[]):
        self.mode = mode
        self.size = size
        self.refs = refs


class CuMap:
    
    def __init__(self, count:int, cu_size:int=64):
        self.count = count
        self.cu_size = cu_size
        self._map:List[CTU] = [None] * count

    @classmethod
    def draw_ctus(cls, count:int, intra:float, inter:float, skip:float, merge:float, intra_mean:int, inter_mean:int, refs:List[int]=[]) -> List[CTU]:
        return random.choices([
            CTU(CTU_mode.INTRA, intra_mean),
            CTU(CTU_mode.INTER, inter_mean, refs),
            CTU(CTU_mode.SKIP, inter_mean, refs),
            CTU(CTU_mode.MERGE, inter_mean, refs),
        ], weights=[intra, inter, skip, merge], k=count)

    def draw(self, *args, **kwargs):
        self._map = self.draw_ctus(*args, **kwargs)

    def draw_intra(self, intra_mean:int, idx:int=0, count:int=None) -> List[CTU]:
        end = None
        if count:
            end = idx + count
        self._map[idx:end] = [CTU(CTU_mode.INTRA, intra_mean)] * count

    def get_slice(self, index:int, count:int) -> List[CTU]:
        stop = index + count
        return self._map[index:stop]

    def update_slice(self, i:int, m:List[CTU]):
        stop = i + len(m)
        assert stop <= self.count
        self._map[i:stop] = m

    def dump(self, csv_out:str):
        with open(csv_out, 'w') as f:
            writer = CTU.get_csv_writer(f)
            for ctu in self._map:
                writer.writerow(ctu)

    def load(self, csv_in:str) -> 'CtuMap':
        data = []
        with open(csv_in, 'r') as f:
            data = [ctu for ctu in CTU.iter_csv_file(csv_in)]
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
        # XRTM.USER_IDX,
        XRTM.TIME_STAMP_IN_MICRO_S,
        XRTM.RENDER_TIMING,
        XRTM.PRIORITY,
        XRTM.TYPE,
        XRTM.SIZE,
        XRTM.START_ADDRESS_CU,
        XRTM.NUMBER_CUS,
        XRTM.EYE_BUFFER,
        XRTM.INDEX,
        XRTM.FRAME_FILE
        # XRTM.BITS_REF,
        # XRTM.BITS_NEW,
        # XRTM.QP_REF,
        # XRTM.QP_NEW,
        # XRTM.REFS,
        # XRTM.PSNR_Y,
        # XRTM.PSNR_U,
        # XRTM.PSNR_V,
        # XRTM.PSNR_YUV,
        # XRTM.INTRA_MEAN,
        # XRTM.INTER_MEAN,
        # *SliceStats.attributes
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
        st.priority = s.priority
        st.start_address_cu = s.cu_address
        st.number_cus = s.cu_count
        st.frame_file = s.frame_file
        st.eye_buffer = s.view_idx
        return st

class STraceRx(STraceTx):
    """
    TODO: implement V'Trace
    """
    pass


class PTraceTx(CsvRecord):
    
    attributes = [
            XRTM.PRIORITY,
            XRTM.USER_ID,
            XRTM.PCKT_SEQNUM,
            XRTM.CN_JITTER_DELAY,
            XRTM.PCKT_AVAILABILITY,
            XRTM.PCKT_SIZE,
            XRTM.PCKT_FRAGNUM,
            XRTM.PCKT_IS_LAST
    ]

    HEADER_SIZE = 40

    def __init__(self, s:STraceTx, payload_size:int, seqnum:int, fragnum:int, is_last=False):

        self._slice = s
        self.pckt_size = payload_size 
        self.pckt_seqnum = seqnum
        self.pckt_fragnum = fragnum
        self.pckt_is_last = is_last
        self.cn_jitter_delay = 0
        self.user_id = 0
        self.lost = False


    @property
    def pckt_availability(self):
        return 0 if self.lost else self._slice.time_stamp_in_micro_s + self.cn_jitter_delay
    
    @property
    def priority(self):
        return self._slice.priority

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

    def adjust(self, slice_type:SliceType, size:int):
        qp = self._i_qp_ref if slice_type == SliceType.IDR else self._p_qp_ref

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
        self.refs = []

        self.priority = -1
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

    def __init__(self, frame_idx:int, intra_mean:float, inter_mean:float, slices_per_frame:int, cu_per_slice:int, cu_count:int, cu_size:int, view_idx=0):
        self.view_idx = view_idx
        self.frame_idx = frame_idx
        
        self.intra_mean = intra_mean
        self.inter_mean = inter_mean
        
        self.slices_per_frame = slices_per_frame
        self.cu_per_slice = cu_per_slice
        self.cu_count = cu_count
        self.cu_size = cu_size

        self.cu_map = CuMap(cu_size)
        self.slices = []
        
        for slice_idx in range(self.slices_per_frame):
            self.slices.append(Slice(self, slice_idx))

    def draw(self, *cu_distribution):
        self.cu_map.draw(self.cu_count, *cu_distribution, self.intra_mean, self.inter_mean)
    
    def encode_slice(self, s:Slice, rc:RateControl, refs:List[int]):
        bit_size = 0
        for cu in self.cu_map.get_slice(s.cu_address, s.cu_count):
            if cu.mode == CTU_mode.INTRA:
                cu.size = self.intra_mean
            elif cu.mode in ( CTU_mode.INTER, CTU_mode.MERGE ):
                cu.refs = refs
                cu.size = self.inter_mean
            elif cu.mode == CTU_mode.SKIP:
                cu.refs = refs
                cu.size = 8
            else:
                raise ValueError('Invalid CTU type')
            bit_size += cu.size
        return rc.adjust(s.slice_type, bit_size)

    @property
    def frame_file(self):
        return f'{self.view_idx}_{self.frame_idx}.csv'
