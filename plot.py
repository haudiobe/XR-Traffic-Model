#!/usr/bin/python3

import argparse
import numpy
from xrtm.models import STraceTx, PTraceTx
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='XRTM Packetizer')
    parser.add_argument('-s', '--s_trace', type=str, help='S-trace file', required=False)
    parser.add_argument('-p', '--p_trace', type=str, help='P-trace file', required=False)

    args = parser.parse_args()
    if args.s_trace != None:
        ts = []
        size = []
        for s in STraceTx.iter_csv_file(args.s_trace):
            ts.append(s.time_stamp_in_micro_s * 1e-6)
            size.append(s.size)
        plt.plot(ts, size)
        plt.xlabel('seconds')
        plt.ylabel('bytes')
        plt.show()

    elif args.p_trace != None:
        ts = []
        size = []
        for p in PTraceTx.iter_csv_file(args.p_trace):
            ts.append(p.time_stamp_in_micro_s * 1e-6)
            size.append(p.size)
        plt.plot(ts, size)
        plt.xlabel('seconds')
        plt.ylabel('bytes')
        plt.show()
