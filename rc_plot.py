#!/usr/bin/python3

import argparse
import numpy
from xrtm.models import RCtrace, VTraceTx
import matplotlib.pyplot as plt
from matplotlib import ticker
from pathlib import Path
from itertools import groupby
import math
import random

hstyle = {'alpha':0.5, 'histtype':'bar', 'ec':'black'}
gstyle = {'alpha':0.5, 'linestyle':':'}

TARGET_RATE = 15e6
FPS = 60.
CU_COUNT = pow(2048/64, 2)

RC_INTRA_ADJUST = True

########################################

def estimate_size(vt:VTraceTx, qp:int, cu_count:int):
    i_size = vt.intra * vt.get_intra_mean(cu_count)
    i_size = qp_adjust(vt.i_qp, qp, i_size)
    p_size = (vt.inter + vt.merge) * vt.get_inter_mean(cu_count)
    p_size = qp_adjust(vt.p_qp, qp, p_size)
    skip = vt.skip
    size_new = i_size + p_size + skip
    return size_new * cu_count

def qp_clamp(qp, qp_min=0, qp_max=51):
    return max(min(qp, qp_max), qp_min)

def target_qp(qp_ref, size_ref, wanted_bits):
    return qp_ref - 6 * math.log2(wanted_bits/size_ref)

def qp_adjust(qp_ref, qp_new, size_ref):
    return size_ref * pow(2, (qp_ref - qp_new)/6)

def get_bits_ref(vt):
    i = vt.intra * vt.i_bits
    p = (vt.inter + vt.merge) * vt.p_bits
    return i+p+(CU_COUNT*vt.inter*8)

VBR = 0
CVBR = 3
CBR = 2

class RC:
    """
    Total average bit budget per frame: Tbits
    Window size is W
    Allocated bits to frame i: Bits[i]
    Allocated bits in model encoding (from V-Trace): A[i]
    """
    def __init__(self, fps, target_rate, cu_count=1024, factor=2, w_max=12, intra_period=-1, _id="", intra_compensation_factor=3):
        self.w = []
        self.w_max = w_max
        self.factor = factor
        self.fps = fps
        self.target_rate = target_rate
        self.ncu = cu_count
        self.intra_period = intra_period
        self._id = _id
        self.intra_compensation_factor = intra_compensation_factor
    
    @staticmethod
    def estimate_frame_size(vt, cu_count, qp=-1) -> int:
        i_size = vt.intra * vt.get_intra_mean(cu_count)
        if qp >= 0:
            i_size = qp_adjust(vt.i_qp, qp, i_size)
        p_size = (vt.inter + vt.merge) * vt.get_inter_mean(cu_count)
        if qp >= 0:
            p_size = qp_adjust(vt.p_qp, qp, p_size)
        skip = vt.skip * 8
        size_new = i_size + p_size + skip
        return size_new * cu_count

    @property
    def frame_bits_ref(self):
        return self.target_rate / self.fps

    def is_in_range(self, qp):
        return (qp >= 0) and (qp <= 50) 

    @property
    def W(self):
        return min(len(self.w)+1, self.w_max)
    
    def get_available_bits(self):
        return self.frame_bits_ref * self.W - sum(self.w)
        
    def get_offset(self, intra=False):
        available = self.get_available_bits()
        if RC_INTRA_ADJUST and intra:
            return available - self.frame_bits_ref
            """
            elif RC_INTRA_ADJUST:
                offset = (available - self.frame_bits_ref) / (self.W / self.factor)
                print(intra_compensation, offset, intra_compensation * offset)
                return intra_compensation * offset
            """
        else:
            return (available - self.frame_bits_ref) / (self.W / self.factor)
    
    def get_budget_max(self):
        return self.w_max * self.frame_bits_ref - sum(self.w[1:])

    def get_frame_budget(self, custom=False, intra=False):
        overflow = self.get_budget_max()
        assert not custom
        if custom:
            if len(self.w) == 0:
                return self.frame_bits_ref
            available = (self.frame_bits_ref * len(self.w) - sum(self.w)) # / len(self.w)
            if intra:
                return available
            offset = available * self.factor * len(self.w) / self.w_max
            bits = self.frame_bits_ref + offset
            return min(bits, overflow)
        else:
            # Available Bits B[i] = Tbits*W – sum (j=i-W+1) (i-1) Bits[j]
            # Bits[i] <= Tbits + (B[i] – Tbits)/(W/factor) (only take a fair portion of the excess bitrate such that you do not overshoot too much)
            """
            if RC_INTRA_ADJUST and intra:
                return self.get_available_bits()
            else:
            """
            bits = self.frame_bits_ref + self.get_offset(intra=intra)
            if RC_INTRA_ADJUST and not intra:
                intra_compensation = self.intra_period / ((self.intra_period-1) + self.intra_compensation_factor)
                bits *= intra_compensation
            assert overflow > 0, 'invalid buffer state'
            return min(bits, overflow)

    def get_fullness_prediction(self, bits):
        # Wm = min(len(self.w)+1, self.w_max)
        return (sum(self.w[1:]) + bits) / (self.w_max * self.frame_bits_ref)

    def no_overflow(self, bits):
        """
        1 > (sum(self.w[1:]) + bits) / (self.w_max * self.frame_bits_ref)
        self.w_max * self.frame_bits_ref - sum(self.w[1:] > bits
        bits = self.w_max * self.frame_bits_ref - sum(self.w[1:]
        """
        return self.get_fullness_prediction(bits) <= 1

    @property
    def current_rate(self):
        if len(self.w) == 0:
            return 0
        return sum(self.w) / (self.w_max/self.fps)

    def rc_start(self, vt):
        qp = weighted(vt, vt.i_qp, vt.p_qp)
        bits = self.estimate_frame_size(vt, self.ncu, qp)
        budget = self.get_frame_budget(intra=vt.is_full_intra)
        if bits <= budget:
            return qp, bits
        else:
            while self.is_in_range(qp+1):
                qp += 1
                bits = self.estimate_frame_size(vt, self.ncu, qp)
                if bits <= budget:
                    break
            return qp, bits

    def rc_end(self, bits):
        self._it = 0
        self.w.append(bits)
        self.w = self.w[-self.w_max:]


def low_pass(decay, h):
    return sum([pow(decay,(i+1))*x for i, x in enumerate(h)])

def to_qscale(qp):
    return math.pow(2, (qp - 12.) / 6. ) * 0.85

def to_qp(qscale):
    return 12.0 + 6.0 * math.log2(qscale/0.85)

def to_complexity(bits, qp):
    return to_qscale(qp) * bits / pow((0.04 * FPS), 0.4)

def intra_frame(vt, rand=False):
    if rand and random.randint(1,360) > 1:
        return vt.is_full_intra
    vt.intra = 1.0
    vt.inter = 0
    vt.merge = 0
    vt.skip = 0
    return vt.is_full_intra


########################################################################################


def compute_size_new(vt:VTraceTx, qp_new:int, cu_count:int) -> int:

    i_size = vt.intra * vt.get_intra_mean(cu_count)
    size_ref = i_size
    i_size = qp_adjust(vt.i_qp, qp_new, i_size)

    p_size = (vt.inter + vt.merge) * vt.get_inter_mean(cu_count)
    size_ref += p_size
    p_size = qp_adjust(vt.p_qp, qp_new, p_size)

    skip = vt.skip
    size_ref += skip

    size_new = i_size + p_size + skip
    return size_ref * cu_count, size_new * cu_count


def weighted(vt, intra, inter):
    return (intra*vt.intra + inter*(1-vt.intra))


def debug():
    qp_ref = 25
    x = []
    y = []
    for q in range(0,50):
        factor = qp_adjust(qp_ref, q, 1e5)
        x.append(q)
        y.append(factor)
    figs = (
        ('size', y),
    )
    return x, y, figs


def run_simulation(fp, fps=60., bitrate=25e6, cu_count=1024, start_frame=0, frames=-1, factor=2, w_max=12, intra_period=-1, intra_compensation_factor=3):

    it = VTraceTx.iter_csv_file(fp)

    rctl = RC(fps, bitrate, cu_count, factor=factor, w_max=w_max, intra_period=intra_period, _id="rctl", intra_compensation_factor=intra_compensation_factor)
    rctl_ref = RC(fps, bitrate, cu_count, factor=factor, w_max=w_max, intra_period=intra_period, _id="ref", intra_compensation_factor=intra_compensation_factor)
    
    x = []

    qp_ref = []
    qp_new = []
    
    frame_ref = []
    frame_new = []
    frame_budget = []
    frame_budget_max = []

    offset = []

    rate_new = []
    rate_ref = []
    rate_target = []
    
    intra_refresh = []

    for idx, vt in enumerate(it):
        if idx < start_frame:
            continue
        if idx == start_frame:
            intra_frame(vt)
            # print('1st frame ', idx)
        elif intra_period == -1:
            if intra_frame(vt, rand=True):
                # print('random intra ', idx)
                pass
        elif ((idx-start_frame) % intra_period) == 0:
            intra_frame(vt)
            # print('periodic ', idx)
        
        intra_refresh.append(vt.is_full_intra)

        x.append(idx-start_frame)
        # ref
        wqp = weighted(vt, vt.i_qp, vt.p_qp)
        bits_ref = rctl.estimate_frame_size(vt, cu_count, wqp)
        qp_ref.append(wqp)
        frame_ref.append(bits_ref)
        rctl_ref.rc_end(bits_ref)
        rate_ref.append(rctl_ref.current_rate)
        # new
        qp, bits_new = rctl.rc_start(vt)
        qp_new.append(qp)
        frame_new.append(bits_new)
        budget = rctl.get_frame_budget(intra=vt.is_full_intra)
        budget_max = rctl.get_budget_max()
        frame_budget.append(budget)
        frame_budget_max.append(budget_max)

        if budget > budget_max:
            print(f'\n[{idx}] invalid frame budget | intra: {vt.is_full_intra} - QP: {qp} - overflow: {int(budget-budget_max)} | max budget: {int(budget_max)}')
        if bits_new > budget_max:
            print(f'[{idx}] buffer overflow | excess bits: {bits_new-budget_max}')
        offset.append(rctl.get_offset(intra=vt.is_full_intra))
        rate_target.append(rctl.target_rate)
        # end
        rctl.rc_end(bits_new)
        rate_new.append(rctl.current_rate)
        if vt.is_full_intra:
            # print(qp, bits_new)
            pass
        # assert rctl.current_rate <= rctl.target_rate
        if frames > 0 and len(x) >= frames:
            break

    frame_target = [rctl.frame_bits_ref] * len(x)
    figs = (
        ('qp', ('VBR', 'C0', qp_ref), ('cVBR', 'C1', qp_new)),
        ('window rate (bps)', ('VBR', 'C0', rate_ref), ('cVBR', 'C1', rate_new), ('target', 'C2', rate_target)),
        ('frame size (bits)', ('VBR', 'C0', frame_ref), ('cVBR', 'C1', frame_new), ('target', 'C2', frame_target)), 
        ('frame (bits)', ('overflow threshold', 'grey', frame_budget_max), ('budget', 'C2', frame_budget), ('cVBR frame', 'C1', frame_new)),
        ('intra refresh', ('I', 'C0', intra_refresh))
    )

    return x, figs


def plot_simulation(fp, fps, bitrate, cu_count):
    
    x, figs1 = run_simulation(fp, fps, bitrate, cu_count, factor=2, w_max=12, start_frame=1000, frames=500, intra_period=8)
    figs = [*figs1]

    def plot(ax, y_label, *y_args):
        for _, y in enumerate(y_args):
            if y == None:
                continue
            if len(y) == 1:
                ax.plot(x, y[0])
            elif len(y) == 2:
                ax.plot(x, y[1], label=y[0])
            elif len(y) == 3:
                ax.plot(x, y[2], label=y[0], color=y[1])
        ax.grid(**gstyle)
        ax.set_ylabel(y_label)
        ax.legend()

    _, axs = plt.subplots(len(figs), constrained_layout=True)
    
    for i,f in enumerate(figs):
        plot(axs[i], *f)
    
    axs[0].set_title(f'[{int(bitrate*1e-3)}Kbps] intra compensation: {RC_INTRA_ADJUST}')
    
    plt.show()


def plot_simulation_multi(fp, fps, bitrate, cu_count):
    ifactor = lambda x: x
    x, figs1 = run_simulation(fp, fps, bitrate, cu_count, factor=2, w_max=12, start_frame=1000, frames=500, intra_period=8, intra_compensation_factor=ifactor(1))
    _, figs2 = run_simulation(fp, fps, bitrate, cu_count, factor=2, w_max=12, start_frame=1000, frames=500, intra_period=8, intra_compensation_factor=ifactor(2))
    _, figs3 = run_simulation(fp, fps, bitrate, cu_count, factor=2, w_max=12, start_frame=1000, frames=500, intra_period=8, intra_compensation_factor=ifactor(3))
    _, figs4 = run_simulation(fp, fps, bitrate, cu_count, factor=2, w_max=12, start_frame=1000, frames=500, intra_period=8, intra_compensation_factor=ifactor(4))
    _, figs5 = run_simulation(fp, fps, bitrate, cu_count, factor=2, w_max=12, start_frame=1000, frames=500, intra_period=8, intra_compensation_factor=ifactor(5))
    _, figs6 = run_simulation(fp, fps, bitrate, cu_count, factor=2, w_max=12, start_frame=1000, frames=500, intra_period=8, intra_compensation_factor=ifactor(6))

    all_figs = [ figs1, figs2, figs3, figs4, figs5, figs6 ]
    figs = [f[4] for f in all_figs]

    _, axs = plt.subplots(len(figs), constrained_layout=True)

    imulti = lambda x: 8/(7+x)
    
    def plot(i, y_label, *y_args):
        ax = axs[i]
        for _, y in enumerate(y_args):
            if y == None:
                continue
            if len(y) == 1:
                ax.plot(x, y[0])
            elif len(y) == 2:
                ax.plot(x, y[1], label=y[0])
            elif len(y) == 3:
                ax.plot(x, y[2], label=y[0], color=y[1])
        ax.grid(**gstyle)
        ax.set_ylabel(f'{ifactor(i+1)}')
        ax.legend()

    for i,f in enumerate(figs):
        plot(i, *f)
    
    axs[0].set_title(f'[{int(bitrate*1e-3)}Kbps] intra compensation: {RC_INTRA_ADJUST}')
    
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XRTM Packetizer')
    parser.add_argument('-v', '--vtrace', type=str, help='vtrace', required=True)
    parser.add_argument('-c', '--cu_count', type=int, help='cu count', required=False, default=CU_COUNT)
    parser.add_argument('-r', '--mbps', type=float, help='target rate mbps', required=False, default=15)
    args = parser.parse_args()

    fp = Path(args.vtrace)
    assert fp.exists(), 'file not found'
    fps=60.
    bitrate = args.mbps * 1e6
    cu_count=1024
    # plot_simulation(fp, fps, bitrate, args.cu_count)
    plot_simulation_multi(fp, fps, bitrate, args.cu_count)