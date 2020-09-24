from typing import List, Iterator

from matplotlib import pyplot as plt
from .models import (
    VTrace, 
    STrace
)
import csv

def read_csv_vtraces(csv_filename:str) -> Iterator[VTrace]:
    with open(csv_filename, newline='') as csvfile:
        vtrace_reader = csv.DictReader(csvfile)
        for row in vtrace_reader:
            yield VTrace(row)

def plot(vtraces:List[VTrace], straces:List[STrace], nslices=1):

    NPLOTS = 5

    if nslices > 1:
        NPLOTS = 4
        slices_by_poc = {}
        for s in straces:
            if s['poc'] in slices_by_poc:
                slices_by_poc[s['poc']]['bits'] += s['bits']
                slices_by_poc[s['poc']]['intra_ctu_bits'] += s['intra_ctu_bits']
                slices_by_poc[s['poc']]['inter_ctu_bits'] += s['inter_ctu_bits']
                slices_by_poc[s['poc']]['skip_ctu_bits'] += s['skip_ctu_bits']
                slices_by_poc[s['poc']]['merge_ctu_bits'] += s['merge_ctu_bits']
                slices_by_poc[s['poc']]['intra_ctu_count'] += s['intra_ctu_count']
                slices_by_poc[s['poc']]['inter_ctu_count'] += s['inter_ctu_count']
                slices_by_poc[s['poc']]['skip_ctu_count'] += s['skip_ctu_count']
                slices_by_poc[s['poc']]['merge_ctu_count'] += s['merge_ctu_count']
            else:
                slices_by_poc[s['poc']] = s
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
        
    vtx = filter(vtraces, 'poc')
    vtIbits = filter(vtraces, 'intra_total_bits')
    vtPbits = filter(vtraces, 'inter_total_bits')

    stx = filter(straces, 'poc')
    sty = filter(straces, 'bits')

    axs[0].plot( vtx, vtIbits, label='intra total ref', color='green')
    axs[0].plot( vtx, vtPbits, label='inter total ref', color='yellowgreen')
    axs[0].plot( stx, sty, label='S-traces bits', color = 'red')
    # axs[0].plot( stx, sty_ref, label='S-traces bits_ref', color = 'orange')
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
        qp_new = filter(straces, 'qp_new')
        qp_ref = filter(straces, 'qp_ref')
        axs[4].plot( stx, qp_new, label='qp_new', color='grey')
        axs[4].plot( stx, qp_ref, label='qp_ref', color='teal')
        axs[4].set_ylabel('QP')
        axs[4].set_xlabel('poc')
        axs[4].legend()

    plt.show()