
import sys
import argparse
from xrtm.models import PTraceTx, PTraceRx, Delay
from typing import Iterable
import random
from scipy.stats import bernoulli
import functools
import json


"""
Random loss according to loss rate
X = Delay equally distributed between 0 and max delay
(Packets throttled to bitrate depending on size) => less important
"""

def compose(*fn):
    def comp(f, g):
        return lambda x: f(g(x))
    return functools.reduce(comp, fn, lambda x: x)


class TestChanCfg():

    def __init__(self, p_trace:str, pp_trace:str, loss_rate=0.01, max_delay_ms=50, max_bitrate=45e6, **noop):
        self.loss_rate = loss_rate
        self.max_delay_ms = max_delay_ms
        self.max_bitrate = max_bitrate
        self.p_trace = p_trace
        self.pp_trace = pp_trace
    
    @classmethod
    def load(cls, fp):
        with open(fp, 'r') as f:
            cfg = json.load(f)
            p = cfg["P-Trace"]
            pp = cfg["Pp-Trace"]
            loss_rate = cfg["loss_rate"]
            max_delay_ms = cfg["max_delay_ms"]
            max_bitrate = cfg["max_bitrate"]
            return cls(p, pp, loss_rate, max_delay_ms, max_bitrate)
    

class TestChan():

    def __init__(self, cfg):
        self.cfg = cfg
        self._delay = Delay(mode=Delay.EQUALLY, parameter1=0, parameter2=self.cfg.max_delay_ms)

    def apply_delay(self, it) -> Iterable[PTraceTx]:
        for p in it:
            p.delay = round(self._delay.get_delay())
            p.time_stamp_in_micro_s += p.delay
            yield p

    def apply_rate(self, it) -> Iterable[PTraceTx]:
        prev_pckt_timestamp = 0
        for p in it:
            rate_delay = 1e6 * (8 * p.size) / self.cfg.max_bitrate
            p.time_stamp_in_micro_s = round(max(p.time_stamp_in_micro_s, prev_pckt_timestamp) + rate_delay)
            prev_pckt_timestamp = p.time_stamp_in_micro_s
            yield p

    def apply_loss(self, it) -> Iterable[PTraceTx]:
        for p in it:
            if bernoulli.rvs(p=self.cfg.loss_rate):
                p.time_stamp_in_micro_s = 0
                p.delay = -1
            yield p

    def process(self, it) -> Iterable[PTraceTx]:
        return compose(
            self.apply_loss,
            self.apply_rate,
            self.apply_delay,
        )(it) # <= functions get called in reverse order


def main(cfg):
    tc = TestChan(cfg)
    with open(cfg.pp_trace, 'w') as of:
        writer = PTraceRx.get_csv_writer(of)
        for op in tc.process(PTraceTx.iter_csv_file(cfg.p_trace)):
            writer.writerow(op.get_csv_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XRTM simple test channel')
    parser.add_argument('-c', '--config', type=str, help='config path', required=True)
    args = parser.parse_args()
    cfg = TestChanCfg.load(args.config)
    main(cfg)
