#!/usr/bin/python3

import argparse
import numpy
from xrtm.models import STraceTx, PTraceTx
import matplotlib.pyplot as plt
from matplotlib import ticker
from pathlib import Path
from itertools import groupby

def bitrate(data, interval):
    data = sorted(data, key=lambda x: x.time_stamp_in_micro_s)
    for ts, samples in groupby(data, key=lambda sample:int(sample.time_stamp_in_micro_s / interval)):
        yield ts * interval / 1e6, (sum([s.size for s in samples]) * 8 * 1e6 / interval)

def latency(data):
    tmax = 0
    for sample in data:
        t = (sample.time_stamp_in_micro_s - sample.render_timing)*1e-3
        tmax = max(tmax, t)
        yield t
    print(f'max latency: {tmax} ms.')

def plot_bitrate(axs, data, interval):
    ts = []
    bps = []
    for xy in bitrate(data, interval):
        ts.append(xy[0])
        bps.append(xy[1] * 1e-6)
    axs.plot(ts, bps)
    axs.grid(alpha=0.5, linestyle=':')
    axs.set_xlabel('time (s)')
    axs.set_ylabel('Mbps')
    fmt = ticker.ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    axs.yaxis.set_major_formatter(fmt)
    axs.set_title('bitrate over time')

def plot_latency(axs, data):
    axs.hist([*latency(data)], density=True, bins=50, alpha=0.5, histtype='bar', ec='black')
    axs.grid(alpha=0.5, linestyle=':')
    axs.set_xlabel('latency (ms)')
    axs.set_ylabel('density')
    axs.set_title('latency distribution')

def plot_strace(fp):
    fig, axs = plt.subplots()
    fig.suptitle(f'{fp.parent}/{fp.stem}{fp.suffix}')
    plot_bitrate(axs, data=STraceTx.iter_csv_file(fp), interval=1e5)
    return fig

def plot_ptrace(fp):
    fig, axs = plt.subplots(2, constrained_layout=True)
    fig.suptitle(f'{fp.parent}/{fp.stem}{fp.suffix}')
    plot_bitrate(axs[0], data=PTraceTx.iter_csv_file(fp), interval=1e5)
    plot_latency(axs[1], data=PTraceTx.iter_csv_file(fp))
    return fig

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='XRTM Packetizer')
    parser.add_argument('-s', '--s_trace', type=str, help='S-trace file', required=False)
    parser.add_argument('-p', '--p_trace', type=str, help='P-trace file', required=False)

    args = parser.parse_args()
    
    if args.s_trace != None:
        p = Path(args.s_trace)
        fig = plot_strace(Path(p))
        png = p.resolve().parent / f'{p.name}.png'
        fig.savefig(png)

    elif args.p_trace != None:
        p = Path(args.p_trace)
        fig = plot_ptrace(Path(p))
        png = p.resolve().parent / f'{p.name}.png'
        fig.savefig(png)

    plt.show()
