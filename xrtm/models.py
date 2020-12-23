import os
import csv
from typing import Any, Iterable, List, Dict, Callable
from enum import Enum, IntEnum
import json
import zlib
import math
from pathlib import Path
from collections import OrderedDict

import logging as logger

from .exceptions import VTraceTxException
from .feedback import Referenceable


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

class SliceType(IntEnum):
    IDR = 1
    P = 3

    @staticmethod
    def parse(t:str) -> str:
        for T in [ SliceType.IDR, SliceType.P ]:
            if t == str(T.value):
                return T
        raise ValueError(f'parser error - invalid data: "{t}"')

    @staticmethod
    def serialize(t:'SliceType') -> str:
        return str(t.value)


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
    
    # VTraceTx
    VIEW_IDX = CSV("view_idx", int, None, -1) # 0: LEFT, 1: RIGHT
    PTS = CSV("pts", int)
    POC = CSV("poc", int)
    CRF_REF = CSV("crf_ref", float)
    QP_REF = CSV("qp_ref", float)
    SLICE_TYPE = CSV("slice_type", SliceType.parse, SliceType.serialize)

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
    QP_NEW = CSV("qp_new", float)
    INTRA_MEAN = CSV("intra_mean", float)
    INTER_MEAN = CSV("inter_mean", float)
    BITS_REF = CSV("bits_ref", int)
    BITS_NEW = CSV("bits_new", int)
    PRIORITY = CSV("priority", int, None, -1)
    SLICE_AVAILABILITY = CSV("slice_availability", int, None, -1)
    SLICE_IDX = CSV("slice_idx", int)

    # @TODO: store actual CTU map
    REFS = CSV("refs", parse_list(int), serialize_list)
    INTRA_CTU_COUNT = CSV("intra_ctu_count", int, None, -1)
    INTRA_CTU_BITS = CSV("intra_ctu_bits", int, None, -1)
    INTER_CTU_COUNT = CSV("inter_ctu_count", int, None, -1)
    INTER_CTU_BITS = CSV("inter_ctu_bits", int, None, -1)
    SKIP_CTU_COUNT = CSV("skip_ctu_count", int, None, -1)
    SKIP_CTU_BITS = CSV("skip_ctu_bits", int, None, -1)
    MERGE_CTU_COUNT = CSV("merge_ctu_count", int, None, -1)
    MERGE_CTU_BITS = CSV("merge_ctu_bits", int, None, -1)


#################################################################################################################################
# @TODO: CTU Map / currently, the CTU map is 

class CTUEncodingMode(IntEnum):
    INTRA = 1
    INTER = 2
    MERGE = 3
    SKIP = 4

class CTU:
    _mode:CTUEncodingMode

    def __init__(self, mode:CTUEncodingMode, *args):
        self._mode = mode
    
    @property
    def mode(self):
        return self._mode



class SliceStats(CsvRecord):

    attributes = [
        XRTM.INTRA_CTU_COUNT,
        XRTM.INTRA_CTU_BITS,
        XRTM.INTER_CTU_COUNT,
        XRTM.INTER_CTU_BITS,
        XRTM.SKIP_CTU_COUNT,
        XRTM.SKIP_CTU_BITS,
        XRTM.MERGE_CTU_COUNT,
        XRTM.MERGE_CTU_BITS
    ]

    def __init__(self, data={}):
        for h in self.attributes:
            setattr(self, h.name, data.get(h.name, 0))

    def add_intra_ctu(self, bits):
        self.intra_ctu_count += 1
        self.intra_ctu_bits += math.ceil(bits)

    def add_inter_ctu(self, bits):
        self.inter_ctu_count += 1
        self.inter_ctu_bits += math.ceil(bits)

    def add_skip_ctu(self):
        self.skip_ctu_count += 1
        self.skip_ctu_bits += math.ceil(8)

    def add_merge_ctu(self, bits):
        self.merge_ctu_count += 1
        self.merge_ctu_bits += math.ceil(bits)
    
    @property
    def total_size(self) -> int:
        """
        total bit size
        """
        return int(self.intra_ctu_bits + self.inter_ctu_bits + self.skip_ctu_bits + self.merge_ctu_bits)

    @property
    def ctu_count(self) -> int:
        return int(self.intra_ctu_count + self.inter_ctu_count + self.skip_ctu_count + self.merge_ctu_count)


#################################################################################################################################
# models

class VTraceTx(CsvRecord):

    attributes = [
            # XRTM.VIEW_IDX
            XRTM.PTS,
            XRTM.POC,
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
        XRTM.PTS,
        XRTM.POC,
        XRTM.VIEW_IDX,
        # XRTM.USER_IDX,
        XRTM.SLICE_IDX,
        XRTM.SLICE_TYPE,
        XRTM.BITS_REF,
        XRTM.BITS_NEW,
        XRTM.QP_REF,
        XRTM.QP_NEW,
        XRTM.REFS,
        XRTM.PSNR_Y,
        XRTM.PSNR_U,
        XRTM.PSNR_V,
        XRTM.PSNR_YUV,
        XRTM.PRIORITY,
        XRTM.INTRA_MEAN,
        XRTM.INTER_MEAN,
        XRTM.SLICE_AVAILABILITY,
        *SliceStats.attributes
    ]

    def __init__(self, data={}):
        ctu_map = {}
        for h in self.attributes:
            if h in SliceStats.attributes:
                ctu_map[h.name] = data.get(h.name, h.default)
            else:
                setattr(self, h.name, data.get(h.name, h.default))
        self.ctu_map = SliceStats(ctu_map)

    @property
    def intra_ctu_count(self):
        return self.ctu_map.intra_ctu_count

    @property
    def intra_ctu_bits(self):
        return self.ctu_map.intra_ctu_bits
    
    @property
    def inter_ctu_count(self):
        return self.ctu_map.inter_ctu_count
    
    @property
    def inter_ctu_bits(self):
        return self.ctu_map.inter_ctu_bits
    
    @property
    def skip_ctu_count(self):
        return self.ctu_map.skip_ctu_count
    
    @property
    def skip_ctu_bits(self):
        return self.ctu_map.skip_ctu_bits
    
    @property
    def merge_ctu_count(self):
        return self.ctu_map.merge_ctu_count
    
    @property
    def merge_ctu_bits(self):
        return self.ctu_map.merge_ctu_bits

    @classmethod
    def from_row(cls, row) -> 'STraceTx':
        for k, v in row:
            print(k, v)
        
    @classmethod
    def from_slice(cls, s:'Slice') -> 'STraceTx':
        st = cls({ 
                'bits_ref': s.bits_ref, 
                **s.__dict__, 
                **s.stats.__dict__ 
            })
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
        return 0 if self.lost else self._slice.slice_availability + self.cn_jitter_delay
    
    @property
    def priority(self):
        return self._slice.priority

class PTraceRx(PTraceTx):
    """
    TODO: implement P'Trace
    """
    pass


#################################################################################################################################
# TODO: Move these to encoder module
#################################################################################################################################

class Slice(Referenceable):

    def __init__(self, 
        pts:int, 
        poc:int, 
        slice_type:SliceType = None, 
        slice_idx:int=0, 
        intra_mean:float=0, 
        inter_mean:float=0, 
        referenceable:bool = True, 
        view_idx:int = 0, 
        anchor_time:int = 0,
        render_jitter:int = 0,
        slice_delay:int = 0,
        eye_buffer_delay:int = 0,
        **noop
    ):
        self.pts = pts
        self.poc = poc
        self.slice_type = slice_type
        self.slice_idx = slice_idx
        self.intra_mean = intra_mean
        self.inter_mean = inter_mean
        self.qp_ref = -1
        self.bits_new = -1
        self.qp_new = -1
        self.y_psnr = -1
        self.u_psnr = -1
        self.v_psnr = -1
        self.yuv_psnr = -1
        self.stats = None
        self.refs = []
        self.view_idx = view_idx
        self._referenceable = referenceable
        
        self.render_timing = 0 
        self.time_stamp_in_micro_s = 0
        
    @property
    def bits(self):
        assert self.bits_new > 0, 'new size not set'
        return self.bits_new
    
    @property
    def bits_ref(self) -> int:
        if self.stats == None:
            assert self.bits_new == -1
            return -1
        else:
           return self.stats.total_size

    @property
    def priority(self):
        return -1

    def get_referenceable_status(self):
        return self._referenceable

    def set_referenceable_status(self, status:bool):
        self._referenceable = status
    
    referenceable = property(get_referenceable_status, set_referenceable_status)

class Frame:

    poc:int
    slices:List[Slice]
    is_idr_frame:bool

    def __init__(self, poc:int, idr:bool = False, view_idx:int = 0):
        self.poc = poc
        self.slices = []
        self.is_idr_frame = idr
        self.view_idx = view_idx

def validates_ctu_distribution(vt:VTraceTx, raise_exception=True) -> bool:
    total = vt.ctu_intra_pct + vt.ctu_inter_pct + vt.ctu_skip_pct + vt.ctu_merge_pct
    valid = round(total * 100) == 100
    if not valid:
        e_msg = f'Invalid CTU distribution on frame {vt.poc} - sum of all CTUs is {total}%'
        if raise_exception:
            raise VTraceTxException(e_msg)
        logger.critical(e_msg)
    return valid

