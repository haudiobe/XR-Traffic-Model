import os
import csv
from typing import List
from enum import Enum, IntEnum

import logging as logger

from .exceptions import VTraceException
from abc import ABC, abstractstaticmethod

XRTM_CSV_PTS = 'pts'
XRTM_CSV_POC = 'poc'
XRTM_CSV_CRF = 'crf_ref'
XRTM_CSV_SLICE_TYPE = 'slice_type'

XRTM_CSV_INTRA_QP = 'intra_qp_ref'
XRTM_CSV_INTRA_BITS = 'intra_total_bits'
XRTM_CSV_INTER_QP = 'inter_qp_ref'
XRTM_CSV_INTER_BITS = 'inter_total_bits'
XRTM_CSV_CU_INTRA = 'ctu_intra_pct'
XRTM_CSV_CU_INTER = 'ctu_inter_pct'
XRTM_CSV_CU_SKIP = 'ctu_skip_pct'
XRTM_CSV_CU_MERGE = 'ctu_merge_pct'
XRTM_CSV_STRACE_BITS = 'bits'
XRTM_CSV_STRACE_QP_REF = 'qp_ref'
XRTM_CSV_STRACE_QP_NEW = 'qp_new'
XRTM_CSV_STRACE_BITS_REF = 'bits_ref'
XRTM_CSV_INTRA_CTU_COUNT = 'intra_ctu_count'
XRTM_CSV_INTRA_CTU_BITS = 'intra_ctu_bits'
XRTM_CSV_INTER_CTU_COUNT = 'inter_ctu_count'
XRTM_CSV_INTER_CTU_BITS = 'inter_ctu_bits'
XRTM_CSV_SKIP_CTU_COUNT = 'skip_ctu_count'
XRTM_CSV_SKIP_CTU_BITS = 'skip_ctu_bits'
XRTM_CSV_MERGE_CTU_COUNT = 'merge_ctu_count'
XRTM_CSV_MERGE_CTU_BITS = 'merge_ctu_bits'
XRTM_CSV_INTRA_MEAN = 'intra_mean'
XRTM_CSV_INTER_MEAN = 'inter_mean'
XRTM_CSV_REF0 = 'ref0'

class SliceType(Enum):
    IDR = 1
    P = 3

class VTrace:

    @staticmethod
    def csv_fields():
        return [
            XRTM_CSV_PTS,
            XRTM_CSV_POC,
            XRTM_CSV_CRF,
            XRTM_CSV_INTRA_QP,
            XRTM_CSV_INTRA_BITS,
            XRTM_CSV_INTER_QP,
            XRTM_CSV_INTER_BITS,
            XRTM_CSV_CU_INTRA,
            XRTM_CSV_CU_INTER,
            XRTM_CSV_CU_SKIP,
            XRTM_CSV_CU_MERGE
        ]

    def __init__(self, data):
        self.pts = int(data[XRTM_CSV_PTS])
        self.poc = int(data[XRTM_CSV_POC])
        self.crf_ref = float(data[XRTM_CSV_CRF])
        self.intra_total_bits = int(data[XRTM_CSV_INTRA_BITS])
        self.intra_qp_ref = float(data[XRTM_CSV_INTRA_QP])/100
        self.inter_total_bits = int(data[XRTM_CSV_INTER_BITS])
        self.inter_qp_ref = float(data[XRTM_CSV_INTER_QP])/100
        self.ctu_intra_pct = float(data[XRTM_CSV_CU_INTRA])/100
        self.ctu_inter_pct = float(data[XRTM_CSV_CU_INTER])/100
        self.ctu_skip_pct = float(data[XRTM_CSV_CU_SKIP])/100
        self.ctu_merge_pct = float(data[XRTM_CSV_CU_MERGE])/100


class SliceStats:

    intra_ctu_count = 0
    intra_ctu_bits = 0
    inter_ctu_count = 0
    inter_ctu_bits = 0
    skip_ctu_count = 0
    skip_ctu_bits = 0
    merge_ctu_count = 0
    merge_ctu_bits = 0

    def add_intra_ctu(self, bits):
        self.intra_ctu_count += 1
        self.intra_ctu_bits += bits

    def add_inter_ctu(self, bits):
        self.inter_ctu_count += 1
        self.inter_ctu_bits += bits

    def add_skip_ctu(self):
        self.skip_ctu_count += 1
        self.skip_ctu_bits += 8

    def add_merge_ctu(self, bits):
        self.merge_ctu_count += 1
        self.merge_ctu_bits += bits
    
    @property
    def total_size(self):
        return self.intra_ctu_bits + self.inter_ctu_bits + self.skip_ctu_bits + self.merge_ctu_bits

    @property
    def ctu_count(self):
        return self.intra_ctu_count + self.inter_ctu_count + self.skip_ctu_count + self.merge_ctu_count


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

class Slice:

    slice_type:SliceType
    refs = List[int]
    stats:SliceStats
    
    def __init__(self, pts:int, poc:int, intra_mean:int, inter_mean:int, referenceable:bool=True):
        self.pts = pts
        self.poc = poc
        self.slice_type = None
        self.qp_ref = -1
        self.bits_new = -1
        self.qp_new = -1
        self.refs = []
        self.stats = None
        self.intra_mean = intra_mean
        self.inter_mean = inter_mean
        self.referenceable = referenceable


    @property
    def bits(self):
        if self.bits_new < 0:
            return self.bits_ref
        else:
            return self.bits_new
    
    @property
    def bits_ref(self):
        if self.stats == None:
            assert self.bits_new == -1
            return -1
        else:
           return self.stats.total_size 
    
    @property
    def qp(self):
        raise NotImplementedError()
    

class STrace:

    @staticmethod
    def csv_headers():
        return [
            XRTM_CSV_PTS,
            XRTM_CSV_POC,
            XRTM_CSV_SLICE_TYPE,
            XRTM_CSV_STRACE_BITS,
            XRTM_CSV_STRACE_BITS_REF,
            XRTM_CSV_STRACE_QP_REF,
            XRTM_CSV_STRACE_QP_NEW,
            XRTM_CSV_INTRA_CTU_COUNT,
            XRTM_CSV_INTRA_CTU_BITS,
            XRTM_CSV_INTER_CTU_COUNT,
            XRTM_CSV_INTER_CTU_BITS,
            XRTM_CSV_SKIP_CTU_COUNT,
            XRTM_CSV_SKIP_CTU_BITS,
            XRTM_CSV_MERGE_CTU_COUNT,
            XRTM_CSV_MERGE_CTU_BITS,
            XRTM_CSV_INTRA_MEAN,
            XRTM_CSV_INTER_MEAN,
            XRTM_CSV_REF0
        ]
    

    @staticmethod
    def to_csv_dict(s:Slice):
        assert s.stats != None

        ctu_stats = {
            XRTM_CSV_INTRA_CTU_COUNT: s.stats.intra_ctu_count,
            XRTM_CSV_INTRA_CTU_BITS: s.stats.intra_ctu_bits,
            XRTM_CSV_INTER_CTU_COUNT: s.stats.inter_ctu_count,
            XRTM_CSV_INTER_CTU_BITS: s.stats.inter_ctu_bits,
            XRTM_CSV_SKIP_CTU_COUNT: s.stats.skip_ctu_count,
            XRTM_CSV_SKIP_CTU_BITS: s.stats.skip_ctu_bits,
            XRTM_CSV_MERGE_CTU_COUNT: s.stats.merge_ctu_count,
            XRTM_CSV_MERGE_CTU_BITS: s.stats.merge_ctu_bits,
        }
        
        return {
            XRTM_CSV_PTS: s.pts,
            XRTM_CSV_POC: s.poc,
            XRTM_CSV_SLICE_TYPE: s.slice_type,
            XRTM_CSV_STRACE_QP_NEW: s.qp_new,
            XRTM_CSV_STRACE_QP_REF: s.qp_ref,
            XRTM_CSV_STRACE_BITS: s.bits,
            XRTM_CSV_STRACE_BITS_REF: s.bits,
            XRTM_CSV_INTRA_MEAN: s.intra_mean,
            XRTM_CSV_INTER_MEAN: s.inter_mean,
            XRTM_CSV_REF0: s.refs,
            **ctu_stats
        }



class Frame:

    poc:int
    slices:List[Slice]

    def __init__(self, poc:int):
        self.poc = poc
        self.slices = []


class RefPicList:
    max_size:int
    pics:List[Frame]
    
    def __init__(self, max_size):
        self.max_size = max_size
        self.pics = []

    def add_frame(self, pic:Frame):
        self.pics.append(pic)
        self.pics = self.pics[-self.max_size:]

    def reset(self):
        self.pics = []
    
    def slice_refs(self, slice_idx): 
        for frame in reversed(self.pics):
            if frame.slices[slice_idx].referenceable:
                yield frame.poc
            if frame.slices[slice_idx].slice_type == SliceType.IDR:
                return

    def set_non_referenceable(self, frame_idx, slice_idx):
        for frame in reversed(self.pics):
            if frame.poc >= frame_idx:
                frame.slices[slice_idx].referenceable = False
            else:
                return


def validates_ctu_distribution(vt:VTrace, raise_exception=True) -> bool:
    total = vt.ctu_intra_pct + vt.ctu_inter_pct + vt.ctu_skip_pct + vt.ctu_merge_pct
    valid = round(total * 100) == 100
    if not valid:
        e_msg = f'Invalid CTU distribution on frame {vt.poc} - sum of all CTUs is {total}%'
        if raise_exception:
            raise VTraceException(e_msg)
        logger.critical(e_msg)
    return valid
