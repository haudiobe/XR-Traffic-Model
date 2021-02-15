import sys
import argparse
import csv
import re
from pathlib import Path
from typing import Iterator, List
from xrtm.packets import DePacketizer, PTraceTx, DePacketizerCfg, DePacketizer
from xrtm.models import STraceTx, STraceRx

from xrtm.utils import _requires, ConfigException, gen_timesteps

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

def peak(pp_input):
    return len([*PTraceTx.iter_csv_file(pp_input)])

def display_progress(pp_processed, pp_total):
    pct = 100 * pp_processed / pp_total
    sys.stdout.write(f'{pct:.2f} %\r')
    sys.stdout.flush()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Decoder configuration')
    parser.add_argument('-c', '--config', type=str, help='config file', required=True)
    
    args = parser.parse_args()
    
    cfg_path = Path(args.config)
    assert cfg_path.exists(), "config file not found"

    cfg = DePacketizerCfg.load(cfg_path)
    unpack = DePacketizer(cfg)
    
    writers = { buffer_idx: STraceRx.get_csv_writer(open(ofp,'w')) for buffer_idx, ofp in cfg.buffers.items() }
    pp_stream = PacketStream(cfg.pp_input)
    
    step = 1e6
    lines = 0

    for ts in gen_timesteps(step, t=0):
        if pp_stream.is_empty():
            break
        t_bucket = pp_stream.pop(ts)
        for s in unpack.process(t_bucket):
            print(s)
            writers[s.buffer].writerow(s.get_csv_dict())
            lines += 1
        if lines > 10:
            break

    for w in writers.values():
        w.close()