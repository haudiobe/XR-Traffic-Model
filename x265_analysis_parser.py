import sys
import os
import csv
import argparse
from pathlib import Path

from enum import Enum, IntEnum
from typing import List

from xrtm_encoder import (
    XRTM_CSV_CU_INTRA,
    XRTM_CSV_CU_INTER,
    XRTM_CSV_CU_SKIP,
    XRTM_CSV_CU_MERGE,
    XRTM_CSV_REFS,
    XRTM_CSV_PTS,

    I_SLICE,
    P_SLICE,
    
    VTrace,
    write_csv_vtraces
)

import logging
logger = logging.getLogger(__name__)

# Note: no PTS
X265_CSV_SLICETYPE = (1, ' Type')
X265_CSV_POC = (2, ' POC')
X265_CSV_QP = (3, ' QP')
X265_CSV_BITS = (4, ' Bits')
X265_CSV_RF = (7, ' RateFactor')
X265_CSV_PSNR = (11, ' YUV PSNR')
X265_CSV_SSIM = (13, ' SSIM(dB)')
X265_CSV_REF0 = (15, ' List 0')
X265_CSV_REF1 = (16, ' List 1')
X265_CSV_TOTAL_FRAME_TIME = (61, ' Total frame time (ms)')

X265_CSV_CU_INTRA = [
    (17, ' Intra 64x64 DC', 64*64),
    (18, ' Intra 64x64 Planar', 64*64),
    (19, ' Intra 64x64 Ang', 64*64),
    (20, ' Intra 32x32 DC', 32*32),
    (21, ' Intra 32x32 Planar', 32*32),
    (22, ' Intra 32x32 Ang', 32*32),
    (23, ' Intra 16x16 DC', 16*16),
    (24, ' Intra 16x16 Planar', 16*16),
    (25, ' Intra 16x16 Ang', 16*16),
    (26, ' Intra 8x8 DC', 8*8),
    (27, ' Intra 8x8 Planar', 8*8),
    (28, ' Intra 8x8 Ang', 8*8),
    (29, ' 4x4', 4*4)
]

X265_CSV_CU_INTER = [
    (30, ' Inter 64x64', 64*64),
    (31, ' Inter 32x32', 32*32),
    (32, ' Inter 16x16', 16*16),
    (33, ' Inter 8x8', 8*8)
]

X265_CSV_CU_SKIP = [
    (34, ' Skip 64x64', 64*64),
    (35, ' Skip 32x32', 32*32),
    (36, ' Skip 16x16', 16*16),
    (37, ' Skip 8x8', 8*8)
]

X265_CSV_CU_MERGE = [
    (38, ' Merge 64x64', 64*64),
    (39, ' Merge 32x32', 32*32),
    (40, ' Merge 16x16', 16*16),
    (41, ' Merge 8x8', 8*8)
]

cu_count_min = 0
cu_count_max = 0

def aggregate_ctu_distribution(row:list) -> dict:
    
    cu_px = {
        XRTM_CSV_CU_INTRA: 0,
        XRTM_CSV_CU_INTER: 0,
        XRTM_CSV_CU_SKIP: 0,
        XRTM_CSV_CU_MERGE: 0,
    }
    
    cu_pct = {
        XRTM_CSV_CU_INTRA: 0,
        XRTM_CSV_CU_INTER: 0,
        XRTM_CSV_CU_SKIP: 0,
        XRTM_CSV_CU_MERGE: 0,
    }

    def cu_weight(field):
        # return field[2] * float(row[field[0]][:-1]) / 100
        pct = float(row[field[0]][:-1]) / 100
        return field[2] * pct, pct
    
    cu_px_sum = 0

    for field in X265_CSV_CU_INTRA:
        w, pct = cu_weight(field)
        cu_px_sum += w
        cu_px[XRTM_CSV_CU_INTRA] += w
        cu_pct[XRTM_CSV_CU_INTRA] += pct

    for field in X265_CSV_CU_INTER:
        w, pct = cu_weight(field)
        cu_px_sum += w
        cu_px[XRTM_CSV_CU_INTER] += w
        cu_pct[XRTM_CSV_CU_INTER] += pct

    for field in X265_CSV_CU_SKIP:
        w, pct = cu_weight(field)
        cu_px_sum += w
        cu_px[XRTM_CSV_CU_SKIP] += w
        cu_pct[XRTM_CSV_CU_SKIP] += pct

    for field in X265_CSV_CU_MERGE:
        w, pct = cu_weight(field)
        cu_px_sum += w
        cu_px[XRTM_CSV_CU_MERGE] += w
        cu_pct[XRTM_CSV_CU_MERGE] += pct
    
    RES = 2048*2048
    cu_count = RES / cu_px_sum

    for (cu_mode, px) in cu_px.items():
        cu_px[cu_mode] = px / cu_px_sum

    total = cu_px[XRTM_CSV_CU_INTRA] + cu_px[XRTM_CSV_CU_INTER] + cu_px[XRTM_CSV_CU_SKIP] + cu_px[XRTM_CSV_CU_MERGE]
    assert round(total) == 1.

    total = cu_pct[XRTM_CSV_CU_INTRA] + cu_pct[XRTM_CSV_CU_INTER] + cu_pct[XRTM_CSV_CU_SKIP] + cu_pct[XRTM_CSV_CU_MERGE]
    assert round(total) == 1.

    return cu_px, cu_pct

def parse_x265_row(row:List, fps=None, w=2048, h=2048, cu_size=False) -> VTrace:

    vt = VTrace()

    def col(field, C=str):
        return C(row[field[0]])

    vt.slice_type = col(X265_CSV_SLICETYPE).strip()

    if not vt.slice_type in [I_SLICE, P_SLICE]:
        raise Exception(f'Invalid "{X265_CSV_SLICETYPE[1]}" : "{vt.slice_type}"')

    vt.poc = col(X265_CSV_POC, int)
    vt.pts = -1 if fps == None else round(vt.poc * fps[0] / fps[1] * 1e+6)
    vt.bits = col(X265_CSV_BITS, int)
    vt.qp = col(X265_CSV_QP, float)
    vt.psnr = col(X265_CSV_PSNR, float)
    vt.ssim = col(X265_CSV_SSIM, float)
    vt.total_frame_time = col(X265_CSV_TOTAL_FRAME_TIME, int)
    vt.rate_factor = col(X265_CSV_RF, float)
    
    (pct, pct_raw) = aggregate_ctu_distribution(row)
    
    if cu_size == True: 
        for (k, v) in pct_raw.items():
            vt.__setattr__(k, v)
    else:
        for (k, v) in pct.items():
            vt.__setattr__(k, v)

    def refs(l:str):
        return list(int(poc) for poc in filter( lambda x:x not in('','-'), l.split(' ')))
    vt.refs = [refs(col(X265_CSV_REF0)), refs(col(X265_CSV_REF1))]

    return vt

def x265_to_xrtm(filename_in, filename_out, fps=None, w=2048, h=2048, cu_size=False):
    vtraces:List[VTrace] = []

    with open(filename_in, 'r', newline='') as X265_csv_in:
        try:
            x265_csv_reader = csv.reader(X265_csv_in)
            headers_row = 0
            for row in x265_csv_reader:
                if headers_row > 0:
                    vtraces.append(parse_x265_row(row, fps=fps, w=2048, h=2048, cu_size=False))
                else:
                    headers_row += 1

        except BaseException as e:
            # x265's csv ends with non-csv data
            logger.error(e)
            raise

    write_csv_vtraces(vtraces, filename_out)


def print_csv_headers(csv_in):
    """
    print csv headers to stdout
    """
    with open(csv_in, 'r', newline='') as X265_csv_in:
        x265_csv_reader = csv.reader(X265_csv_in, delimiter=',')
        for row in x265_csv_reader:
            for h in enumerate(row):
                print(h)
            return


def plot(csv_in):
    with open(csv_in, 'r', newline='') as X265_csv_in:
        x265_csv_reader = csv.reader(X265_csv_in)
        headers_row = 0
        
        poc = []
        raw = {}
        adjusted = {}

        def aggs(r, acc):
            for (k, v) in r.items():
                if not k in acc:
                    acc[k] = [v]
                else:
                    acc[k].append(v)

        for row in x265_csv_reader:
            if headers_row > 0:
                (pct, pct_raw) = aggregate_ctu_distribution(row)
                aggs(pct, adjusted)
                aggs(pct_raw, raw)
                poc.append(int(row[X265_CSV_POC[0]]))
            else:
                headers_row += 1

        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(4, sharex=True)
        
        def add_plot(i, k):
            axs[i].plot(poc, raw[k], label='count based', color = 'green')
            axs[i].plot(poc, adjusted[k], label='adjusted to pixel coverage', color = 'red')
            axs[i].set_ylabel(f'{k} CU %')
            axs[i].legend()

        i=0
        for (k, _) in raw.items():
            add_plot(i, k)
            i += 1

        plt.show()


def main():
    
    parser = argparse.ArgumentParser(description='attempts converting x265 analysis to V-trace csv')    
    parser.add_argument('--fps', metavar='num/den', type=str,
                         help='constant fps in num/den notation', default='1/60', required=False )
    parser.add_argument('-s', '--cu_size', help='CTU percentage based on CU size instead of CU count', action='store_true')
    
    parser.add_argument('-W', type=int, help='frame width', default=2048, required=False )
    parser.add_argument('-H', type=int, help='frame height', default=2048, required=False )

    parser.add_argument('csv', nargs=1)

    parser.add_argument('--plot', help='compare the CTU percentage w/out --cu_size', action='store_true')
    # parser.add_argument('--headers', help='print csv headers', action='store_true')

    args = parser.parse_args()

    p = Path(args.csv[0])
    if p.suffix != '.csv':
        print('not a *.csv file')
        return

    if not p.exists():
        print(f'not found: {p.absolute()}')
        return

    csv_in = str(p.absolute())
    if args.plot:
        plot(csv_in)
        return

    csv_out = f'{csv_in[:-4]}.vtrace.csv'
    
    (num, den) = (int(i) for i in args.fps.split('/'))
    x265_to_xrtm(csv_in, csv_out, fps=(num, den), w=args.W, h=args.H, cu_size=args.cu_size)

if __name__ == '__main__':
    main()