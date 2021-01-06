from typing import List, Iterator

from matplotlib import pyplot as plt
from .models import (
    VTraceTx, 
    STraceTx,
    XRTM
)
import csv

def read_csv_vtraces(csv_filename:str) -> Iterator[VTraceTx]:
    with open(csv_filename, newline='') as csvfile:
        vtrace_reader = csv.DictReader(csvfile)
        for row in vtrace_reader:
            yield VTraceTx.from_csv_row(row)

def plot(vtraces:List[VTraceTx], straces:List[STraceTx], nslices=1):

    NPLOTS = 5
    keys = [
        XRTM.BITS_NEW,
        XRTM.INTRA_CU_BITS,
        XRTM.INTER_CU_BITS,
        XRTM.SKIP_CU_BITS,
        XRTM.MERGE_CU_BITS,
        XRTM.INTRA_CU_COUNT,
        XRTM.INTER_CU_COUNT,
        XRTM.SKIP_CU_COUNT,
        XRTM.MERGE_CU_COUNT,
    ]

    if nslices > 1:
        NPLOTS = 4
        slices_by_poc = {}
        for s in straces:
            if s[XRTM.FRAME_IDX] in slices_by_poc:
                for k in keys:
                    slices_by_poc[s[XRTM.FRAME_IDX]][k] += s[k]
            else:
                slices_by_poc[s[XRTM.FRAME_IDX]] = s
        agg = []
        for i in range(len(slices_by_poc.keys())):
            agg.append(slices_by_poc[i])
        straces = agg
        assert len(straces) == len(vtraces)

    fig, axs = plt.subplots(NPLOTS, sharex=True)

    def filter(traces, key):
        if type(traces[0]) == dict:
            return [t[key] for t in traces]
        return [t.__dict__[key] for t in traces]
        
    vtx = filter(vtraces, XRTM.FRAME_IDX)
    vtIbits = filter(vtraces, 'intra_total_bits')
    vtPbits = filter(vtraces, 'inter_total_bits')

    stx = filter(straces, XRTM.FRAME_IDX)
    sty = filter(straces, XRTM.BITS_NEW.name)

    axs[0].plot( vtx, vtIbits, label='intra total ref', color='green')
    axs[0].plot( vtx, vtPbits, label='inter total ref', color='yellowgreen')
    axs[0].plot( stx, sty, label='S-traces bits', color = 'red')
    # axs[0].plot( stx, sty_ref, label='S-traces bits_ref', color = 'orange')
    axs[0].set_ylabel('bit size')
    axs[0].set_xlabel(XRTM.FRAME_IDX)
    axs[0].legend()

    intra_bits = filter(straces, XRTM.INTRA_CU_BITS.name)
    inter_bits = filter(straces, XRTM.INTER_CU_BITS.name)
    merge_bits = filter(straces, XRTM.MERGE_CU_BITS.name)
    skip_bits = filter(straces, XRTM.SKIP_CU_BITS.name)

    axs[1].plot( stx, intra_bits, label='intra', color='yellowgreen')
    axs[1].plot( stx, inter_bits, label='inter', color='darkorange')
    axs[1].plot( stx, merge_bits, label='merge', color='chocolate')
    axs[1].plot( stx, skip_bits, label='skip', color='steelblue')
    axs[1].set_ylabel('total CU bits')
    axs[1].set_xlabel(XRTM.FRAME_IDX)
    axs[1].legend()

    intra_count = filter(straces, XRTM.INTRA_CU_COUNT.name)
    inter_count = filter(straces, XRTM.INTER_CU_COUNT.name)
    merge_count = filter(straces, XRTM.MERGE_CU_COUNT.name)
    skip_count = filter(straces, XRTM.SKIP_CU_COUNT.name)

    axs[2].plot( stx, intra_count, label='intra', color='yellowgreen')
    axs[2].plot( stx, inter_count, label='inter', color='darkorange')
    axs[2].plot( stx, merge_count, label='merge', color='chocolate')
    axs[2].plot( stx, skip_count, label='skip', color='steelblue')
    axs[2].set_ylabel('CU count')
    axs[2].set_xlabel(XRTM.FRAME_IDX)
    axs[2].legend()

    intra_mean = filter(straces, XRTM.INTRA_MEAN.name)
    inter_mean = filter(straces, XRTM.INTER_MEAN.name)
    axs[3].plot( stx, intra_mean, label='intra medium value', color='green')
    axs[3].plot( stx, inter_mean, label='inter medium value', color='yellowgreen')
    axs[3].set_ylabel('bits')
    axs[3].set_xlabel(XRTM.FRAME_IDX)
    axs[3].legend()
    
    if nslices == 1:
        qp_new = filter(straces, XRTM.QP_NEW.name)
        qp_ref = filter(straces, XRTM.QP_REF.name)
        axs[4].plot( stx, qp_new, label='qp_new', color='grey')
        axs[4].plot( stx, qp_ref, label='qp_ref', color='teal')
        axs[4].set_ylabel('QP')
        axs[4].set_xlabel(XRTM.FRAME_IDX)
        axs[4].legend()

    plt.show()