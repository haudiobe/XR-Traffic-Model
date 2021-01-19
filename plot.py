#!/usr/bin/python3

import argparse
import numpy
from xrtm.models import STraceTx, PTraceTx, CU
import matplotlib.pyplot as plt
from matplotlib import ticker
from pathlib import Path
from itertools import groupby

hstyle = {'alpha':0.5, 'histtype':'bar', 'ec':'black'}
gstyle = {'alpha':0.5, 'linestyle':':'}

def bitrate(data, interval):
    data = sorted(data, key=lambda x: x.time_stamp_in_micro_s)
    for ts, samples in groupby(data, key=lambda sample:int(sample.time_stamp_in_micro_s / interval)):
        yield ts * interval / 1e6, (sum([s.size for s in samples]) * 8 * 1e6 / interval)

def latency(data):
    tmax = 0
    for sample in data:
        t = (sample.time_stamp_in_micro_s - sample.render_timing)*1e-3
        tmax = max(tmax, t)
        yield sample.time_stamp_in_micro_s * 1e-6, t
    print(f'max latency: {tmax} ms.')

def frames(data):
    frames = {}
    for s in data:
        k = f'{s.frame_index}{s.buffer}'
        if not k in frames:
            frames[k] = [s.size]
        else:
            frames[k].append(s.size)
    for v in frames.values():
        yield sum(v), v

def plot_bitrate(axs, data, interval):
    ts = []
    bps = []
    for xy in bitrate(data, interval):
        ts.append(xy[0])
        bps.append(xy[1] * 1e-6)
    axs.plot(ts, bps)
    axs.grid(**gstyle)
    axs.set_xlabel('time (s)')
    axs.set_ylabel('Mbps')
    fmt = ticker.ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    axs.yaxis.set_major_formatter(fmt)
    axs.set_title('bitrate over time')

def plot_latency(axs, data):
    axs.hist([tl[1] for tl in latency(data)], density=True, bins=50, **hstyle)
    axs.grid(**gstyle)
    axs.set_xlabel('latency (ms)')
    axs.set_ylabel('density')
    axs.set_title('latency distribution')

def plot_latency_over_time(axs, data):
    ts, pckt_latency = zip(*latency(data))
    axs.plot(ts, pckt_latency)
    axs.grid(**gstyle)
    axs.set_xlabel('time (s)')
    axs.set_ylabel('latency (ms)')
    axs.set_title('latency over time')

def plot_size_distribution(axf, axs , data):
    f, s = zip(*frames(data))
    axf.hist(f, density=True, bins='auto', **hstyle)
    axf.grid(**gstyle)
    axf.set_xlabel('frame size (bytes)')
    axf.set_ylabel('density')
    axf.set_title('frame size distribution')
    r = []
    for l in s:
        r += l
    axs.hist(r, density=True, bins='auto', **hstyle)
    axs.grid(**gstyle)
    axs.set_xlabel('slice size (bytes)')
    axs.set_ylabel('density')
    axs.set_title('slice size distribution')

def plot_qp(axs, data):
    axs.hist([cu.qpnew for cu in data], density=True, bins='auto', **hstyle)
    axs.grid(**gstyle)
    axs.set_xlabel('QP')
    axs.set_ylabel('density')
    axs.set_title('QP distribution')

def plot_frame(fp):
    fig, axs = plt.subplots()
    fig.suptitle(f'{fp.parent}/{fp.stem}{fp.suffix}')
    plot_qp(axs, data=CU.iter_csv_file(fp))
    return fig

def plot_strace(fp):
    fig, axs = plt.subplots(3, constrained_layout=True)
    fig.suptitle(f'{fp.parent}/{fp.stem}{fp.suffix}')
    plot_bitrate(axs[0], data=STraceTx.iter_csv_file(fp), interval=1e5)
    plot_size_distribution(axs[1], axs[2], data=STraceTx.iter_csv_file(fp))
    return fig

def plot_ptrace(fp):
    fig, axs = plt.subplots(3, constrained_layout=True)
    fig.suptitle(f'{fp.parent}/{fp.stem}{fp.suffix}')
    plot_bitrate(axs[0], data=PTraceTx.iter_csv_file(fp), interval=1e5)
    plot_latency_over_time(axs[1], data=PTraceTx.iter_csv_file(fp))
    plot_latency(axs[2], data=PTraceTx.iter_csv_file(fp))
    return fig

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='XRTM Packetizer')
    parser.add_argument('-s', '--s_trace', type=str, help='S-trace file', required=False)
    parser.add_argument('-p', '--p_trace', type=str, help='P-trace file', required=False)
    parser.add_argument('-f', '--frame_file', type=str, help='plot frame file', required=False)
    parser.add_argument('-n', '--no_show', action='store_true', help='don\'t show figure, save png', required=False)

    args = parser.parse_args()
    
    if args.s_trace != None:
        p = Path(args.s_trace)
        fig = plot_strace(p)
        png = p.resolve().parent / f'{p.name}.png'
        fig.savefig(png)

    elif args.p_trace != None:
        p = Path(args.p_trace)
        fig = plot_ptrace(p)
        png = p.resolve().parent / f'{p.name}.png'
        fig.savefig(png)

    elif args.frame_file != None:
        p = Path(args.frame_file)
        fig = plot_frame(p)
        png = p.resolve().parent / f'{p.name}.png'
        fig.savefig(png)

    if not args.no_show:
        plt.show()
