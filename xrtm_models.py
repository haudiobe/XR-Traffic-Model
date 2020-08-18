import os
import csv
from typing import List

class VTraceException(Exception):
    pass

XRTM_CSV_PTS = 'pts'
XRTM_CSV_POC = 'poc'
XRTM_CSV_CRF = 'crf_ref'

XRTM_CSV_INTRA_QP = 'intra_qp_ref'
XRTM_CSV_INTRA_BITS = 'intra_total_bits'

XRTM_CSV_INTER_QP = 'inter_qp_ref'
XRTM_CSV_INTER_BITS = 'inter_total_bits'

XRTM_CSV_CU_INTRA = 'ctu_intra_pct'
XRTM_CSV_CU_INTER = 'ctu_inter_pct'
XRTM_CSV_CU_SKIP = 'ctu_skip_pct'
XRTM_CSV_CU_MERGE = 'ctu_merge_pct'

def plot(vtraces, straces, nslices=1):

    from matplotlib import pyplot as plt

    NPLOTS = 5

    if nslices > 1:
        NPLOTS = 4
        slices_by_poc = {}
        for s in straces:
            if s.poc in slices_by_poc:
                slices_by_poc[s.poc].bits += s.bits
                slices_by_poc[s.poc].intra_ctu_bits += s.intra_ctu_bits
                slices_by_poc[s.poc].inter_ctu_bits += s.inter_ctu_bits
                slices_by_poc[s.poc].skip_ctu_bits += s.skip_ctu_bits
                slices_by_poc[s.poc].merge_ctu_bits += s.merge_ctu_bits
                slices_by_poc[s.poc].intra_ctu_count += s.intra_ctu_count
                slices_by_poc[s.poc].inter_ctu_count += s.inter_ctu_count
                slices_by_poc[s.poc].skip_ctu_count += s.skip_ctu_count
                slices_by_poc[s.poc].merge_ctu_count += s.merge_ctu_count
            else:
                slices_by_poc[s.poc] = s
        agg = []
        for i in range(len(slices_by_poc.keys())):
            agg.append(slices_by_poc[i])
        straces = agg
        assert len(straces) == len(vtraces)

    fig, axs = plt.subplots(NPLOTS, sharex=True)

    def filter(traces, key):
        return [t.__dict__[key] for t in traces]

    vtx = filter(vtraces, 'poc')
    vtIbits = filter(vtraces, 'intra_total_bits')
    vtPbits = filter(vtraces, 'inter_total_bits')

    stx = filter(straces, 'poc')
    sty = filter(straces, 'bits')

    axs[0].plot( vtx, vtIbits, label='intra total ref', color='green')
    axs[0].plot( vtx, vtPbits, label='inter total ref', color='yellowgreen')
    axs[0].plot( stx, sty, label='S-traces', color = 'red')
    axs[0].set_ylabel('bit size')
    axs[0].set_xlabel('poc')
    axs[0].legend()

    intra_bits = filter(straces, 'intra_ctu_bits')
    inter_bits = filter(straces, 'inter_ctu_bits')
    merge_bits = filter(straces, 'merge_ctu_bits')
    skip_bits = filter(straces, 'skip_ctu_bits')

    axs[1].plot( stx, intra_bits, label='intra', color='yellowgreen')
    axs[1].plot( stx, inter_bits, label='inter', color='darkorange')
    axs[1].plot( stx, merge_bits, label='merge', color='chocolate')
    axs[1].plot( stx, skip_bits, label='skip', color='steelblue')
    axs[1].set_ylabel('total CTU bits')
    axs[1].set_xlabel('poc')
    axs[1].legend()

    intra_count = filter(straces, 'intra_ctu_count')
    inter_count = filter(straces, 'inter_ctu_count')
    merge_count = filter(straces, 'merge_ctu_count')
    skip_count = filter(straces, 'skip_ctu_count')

    axs[2].plot( stx, intra_count, label='intra', color='yellowgreen')
    axs[2].plot( stx, inter_count, label='inter', color='darkorange')
    axs[2].plot( stx, merge_count, label='merge', color='chocolate')
    axs[2].plot( stx, skip_count, label='skip', color='steelblue')
    axs[2].set_ylabel('CTU count')
    axs[2].set_xlabel('poc')
    axs[2].legend()

    intra_mean = filter(straces, 'intra_mean')
    inter_mean = filter(straces, 'inter_mean')
    axs[3].plot( stx, intra_mean, label='intra medium value', color='green')
    axs[3].plot( stx, inter_mean, label='inter medium value', color='yellowgreen')
    axs[3].set_ylabel('bits')
    axs[3].set_xlabel('poc')
    axs[3].legend()
    
    if nslices == 1:
        qp = filter(straces, 'qp')
        qp_new = filter(straces, 'qp_new')
        axs[4].plot( stx, qp, label='qp', color='grey')
        axs[4].plot( stx, qp_new, label='qp_new', color='teal')
        axs[3].set_ylabel('QP')
        axs[4].set_xlabel('poc')
        axs[4].legend()

    plt.show()
    

class VTrace:
    pts:int
    poc:int
    crf_ref:float
    
    intra_qp_ref:float
    intra_total_bits:int
    
    inter_qp_ref:float
    inter_total_bits:int
    
    ctu_intra_pct:float
    ctu_inter_pct:float
    ctu_skip_pct:float
    ctu_merge_pct:float

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
        self.pts = int(data['pts'])
        self.poc = int(data['poc'])
        self.crf_ref = float(data['crf_ref'])

        self.intra_total_bits = int(data['intra_total_bits'])
        self.intra_qp_ref = float(data['intra_qp_ref'])/100

        self.inter_total_bits = int(data['inter_total_bits'])
        self.inter_qp_ref = float(data['inter_qp_ref'])/100

        self.ctu_intra_pct = float(data['ctu_intra_pct'])/100
        self.ctu_inter_pct = float(data['ctu_inter_pct'])/100
        self.ctu_skip_pct = float(data['ctu_skip_pct'])/100
        self.ctu_merge_pct = float(data['ctu_merge_pct'])/100

def read_csv_vtraces(csv_filename):
    with open(csv_filename, newline='') as csvfile:
        vtrace_reader = csv.DictReader(csvfile)
        for row in vtrace_reader:
            yield VTrace(row)



class STrace:

    pts:int
    poc:int
    bits:int
    qp:float
    qp_new:float
    slice_type:int
    
    intra_ctu_count:int
    intra_ctu_bits:int
    inter_ctu_count:int
    inter_ctu_bits:int
    skip_ctu_count:int
    skip_ctu_bits:int
    merge_ctu_count:int
    skip_ctu_bits:int
    
    intra_mean:int
    inter_mean:int

    feedback:str

    @staticmethod
    def csv_fields():
        return [
            'pts',
            'poc',
            'slice_type',
            'bits',
            'qp',
            'qp_new',
            'intra_ctu_count',
            'intra_ctu_bits',
            'inter_ctu_count',
            'inter_ctu_bits',
            'skip_ctu_count',
            'skip_ctu_bits',
            'merge_ctu_count',
            'merge_ctu_bits',
            'intra_mean',
            'inter_mean',
            'feedback'
        ]

    def to_csv_dict(self) -> dict:
        # custom formating here as needed
        return self.__dict__
