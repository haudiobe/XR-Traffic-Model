#!/usr/bin/python3

import argparse
import numpy as np
from xrtm.models import STraceTx, PTraceTx, CU, PTraceRx
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

def inter_arrival(data):
    prev = None
    for _, p in enumerate(data):
        if prev is None:
            yield None, p
        else:
            dt =  p.time_stamp_in_micro_s - prev.time_stamp_in_micro_s
            yield dt * 1e-3, p
        prev = p

def latency(data):
    tmax = 0
    tmin = 1e9
    for sample in data:
        t = (sample.time_stamp_in_micro_s - sample.render_timing)*1e-3
        tmax = max(tmax, t)
        tmin = min(tmin, t)
        yield sample.time_stamp_in_micro_s * 1e-6, t
    print(f'pckt latency (ms) | min: {tmin} | max: {tmax}')

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
    axs.set_title('bitrate')

def plot_latency(axs, data):
    axs.hist([tl[1] for tl in latency(data)], density=True, bins=50, **hstyle)
    axs.grid(**gstyle)
    axs.set_xlabel('latency (ms)')
    axs.set_ylabel('density')
    axs.set_title('pckt latency')

def plot_latency_over_time(axs, data):
    ts, pckt_latency = zip(*latency(data))
    axs.plot(ts, pckt_latency)
    axs.grid(**gstyle)
    axs.set_xlabel('time (s)')
    axs.set_ylabel('latency (ms)')
    # axs.set_title('latency over time')

def plot_size_distribution(axf, axs , data):
    f, s = zip(*frames(data))
    axf.hist(f, density=True, bins='auto', **hstyle)
    axf.grid(**gstyle)
    axf.set_xlabel('frame size (bytes)')
    axf.set_ylabel('density')
    r = []
    for l in s:
        r += l
    axs.hist(r, density=True, bins='auto', **hstyle)
    axs.grid(**gstyle)
    axs.set_xlabel('slice size (bytes)')
    axs.set_ylabel('density')
    # axs.set_title('slice size distribution')

def plot_pckt_size(*ax, data=[]):
    ps_min = 1e9
    ps_max = 0
    ps = []
    for p in data:
        ps_min = min(p.size, ps_min)
        ps_max = max(p.size, ps_max)
        ps.append(p.size)
    print(f'pckt size (bytes) | min: {ps_min} | max: {ps_max}')
    sor = np.sort(np.array(ps))
    dis = 1 - 1. * np.arange(len(sor)) / (len(sor) - 1)
    ax[0].plot(sor, dis)
    ax[0].grid(**gstyle)
    ax[0].set_ylabel('distribution')
    ax[0].set_yscale('linear')
    ax[0].set_xlabel('size (bytes)')
    ax[0].set_title('pckt size distribution')

def dump_pckt_inter_arrival(fp, ptraces):
    with open(fp, 'w') as f:
        writer = PTraceRx.get_csv_writer(f)
        for p in ptraces:
            writer.writerow(p.get_csv_dict())


def plot_pckt_inter_arrival(ax, data, csv=None):
    dt_max = 0
    dt_min = 1e9
    y = []
    x = []
    prx = []

    for dt, p in inter_arrival(data):
        if dt == None:
            dt = 0
        else:
            dt_min = min(dt_min, dt)
        y.append(dt)
        x.append(p.time_stamp_in_micro_s)
        dt_max = max(dt_max, dt)
        if csv:
            rx = PTraceRx(p.__dict__)
            rx.inter_arrival_micro_s = dt * 1e3
            prx.append(rx)
    if csv:
        dump_pckt_inter_arrival(csv, prx)
    
    sor = np.sort(np.array(y))
    dis = 1 - 1. * np.arange(len(sor)) / (len(sor) - 1)
    print(f'pckt inter arrival (ms) | min: {dt_min} | max: {dt_max}')
    ax.plot(sor, dis)
    ax.grid(**gstyle)
    ax.set_ylabel('distribution')
    ax.set_yscale('log')
    ax.set_xlabel('inter arrival (ms)')
    ax.set_title('inter arrival distribution')
    
def plot_qp(axs, data, frames_dir):
    frames = {}
    for s in data:
        if s.frame_file in frames:
            continue
        frames[s.frame_file] = s.time_stamp_in_micro_s
    tq = sorted(
        [(timestamp * 1e-6, avg_frame_qp(Path(frame_file))) for frame_file, timestamp in frames.items()],
        key=lambda xy: xy[0]
    )
    ts, qp = zip(*tq)
    axs.plot(ts, qp)
    axs.grid(**gstyle)
    axs.set_xlabel('time (s)')
    axs.set_ylabel('QP')
    # axs.set_title('QP over time')

def avg_frame_qp(fp:Path):
    qp = []
    for cu in CU.iter_csv_file(fp):
        if cu.qpnew != None:
            qp.append(cu.qpnew)
    return sum(qp)/len(qp)

def plot_strace(fp):
    fig, axs = plt.subplots(3, constrained_layout=True)
    fig.suptitle(f'{fp.parent}/{fp.stem}{fp.suffix}')
    plot_bitrate(axs[0], data=STraceTx.iter_csv_file(fp), interval=1e5)
    plot_size_distribution(axs[1], axs[2], data=STraceTx.iter_csv_file(fp))
    # frames_dir = fp.resolve().parent / f'{fp.stem}.frames'
    # plot_qp(axs[3], STraceTx.iter_csv_file(fp), frames_dir)
    return fig

def plot_ptrace(fp):
    fig, axs = plt.subplots(4, constrained_layout=True, figsize=(8, 6), dpi=80)
    fig.suptitle(f'{fp.parent}/{fp.stem}{fp.suffix}')
    plot_pckt_inter_arrival(axs[0], data=PTraceTx.iter_csv_file(fp), csv=f'{fp.parent}/{fp.stem}.inter_arrival{fp.suffix}')
    plot_pckt_size(axs[1], data=PTraceTx.iter_csv_file(fp))
    plot_latency(axs[2], data=PTraceTx.iter_csv_file(fp))
    plot_bitrate(axs[3], data=PTraceTx.iter_csv_file(fp), interval=1e5)
    return fig


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='XRTM Packetizer')
    parser.add_argument('-s', '--s_trace', type=str, help='S-trace file', required=False)
    parser.add_argument('-p', '--p_trace', type=str, help='P-trace file', required=False)
    parser.add_argument('-n', '--no_show', action='store_true', help='don\'t show figure, save png', required=False)
    
    args = parser.parse_args()
    noop = True

    if args.s_trace != None:
        p = Path(args.s_trace)
        fig = plot_strace(p)
        png = p.resolve().parent / f'{p.name}.png'
        fig.savefig(png, dpi=600)
        noop = False

    elif args.p_trace != None:
        p = Path(args.p_trace)
        fig = plot_ptrace(p)
        png = p.resolve().parent / f'{p.name}.png'
        fig.savefig(png, dpi=600)
        noop = False

    if noop:
        parser.print_help()
    elif not args.no_show:
        plt.show()

