import sys
import argparse
import csv
import re
from pathlib import Path
from typing import Iterator, List
from xrtm.packets import DePacketizer, PTraceTx, DePacketizerCfg, DePacketizer
from xrtm.models import STraceTx, STraceRx

from xrtm.utils import _requires, ConfigException, gen_timesteps

from time import perf_counter

class PacketStream:

    def __init__(self, ptraces_csv:Path):
        self._stream = sorted(PTraceTx.iter_csv_file(ptraces_csv), key=lambda x: x.time_stamp_in_micro_s)
    
    def is_empty(self) -> bool:
        return len(self._stream) == 0
    
    def peak(self) -> PTraceTx:
        return self._stream[0]

    def pop(self, timestamp:int) -> List[PTraceTx]:
        bucket = []
        while self._stream[0].time_stamp_in_micro_s <= timestamp:
            p = self._stream.pop(0)
            bucket.append(p)
            if self.is_empty():
                break
        return bucket

def probe_size(pp_input):
    return len([*PTraceTx.iter_csv_file(pp_input)])


_start_ts = 0 

def display_progress(pp_processed, pp_total):
    elapsed = perf_counter() - _start_ts
    rmn = int(elapsed * (pp_total - pp_processed) / pp_processed) if (pp_processed > 0) else 0

    pct = 100 * pp_processed / pp_total
    sys.stdout.write(f'{pct:.2f} % - {int(elapsed)} s. {rmn} s.\r')
    sys.stdout.flush()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Decoder configuration')
    parser.add_argument('-c', '--config', type=str, help='config file', required=True)
    
    args = parser.parse_args()
    
    cfg_path = Path(args.config)
    assert cfg_path.exists(), "config file not found"

    cfg = DePacketizerCfg.load(cfg_path)
    unpack = DePacketizer(cfg)
    
    writers = {}
    for buffer_idx, ofp in cfg.buffers.items():
        of = open(ofp,'w')
        writers[buffer_idx] = (of, STraceRx.get_csv_writer(of))

    pp_total = probe_size(cfg.pp_input)
    pp_processed = 0

    pp_stream = PacketStream(cfg.pp_input)
    step = 1e6

    slices = 0
    _start_ts = perf_counter() 
    
    for ts in gen_timesteps(step, t=0):
        if pp_stream.is_empty():
            break
        t_bucket = pp_stream.pop(ts)
        pp_processed += len(t_bucket)

        for s in unpack.process(t_bucket):
            writers[s.buffer][1].writerow(s.get_csv_dict())
            slices += 1

        display_progress(pp_processed, pp_total)


    for w in writers.values():
        w[0].close()