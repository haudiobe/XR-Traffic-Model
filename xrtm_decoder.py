import sys
import argparse
import csv
import re
from pathlib import Path
from typing import Iterator, List
from xrtm.packets import DePacketizer, PTraceTx
from xrtm.models import XRTM, STraceTx, Frame, Slice, CuMap, CU_mode, CU_status, CU


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

def gen_timesteps(step:float, t=0.) -> Iterator[int]:
    while True:
        t += step
        yield round(t)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Decoder configuration')
    parser.add_argument('-p', '--p_trace', type=str, help='p-trace csv input', required=True)
    parser.add_argument('-s', '--s_trace', type=str, help='s-trace csv input', required=True)
    
    args = parser.parse_args()
    
    ptrace_in = Path(args.p_trace)
    assert ptrace_in.exists()
    
    strace_in = Path(args.s_trace)
    assert strace_in.exists()
    
    strace_out = strace_in.parent / f'{strace_in.stem}-rx.csv'
    
    # all timestamps use microsecond accuracy
    cfg = {
        'user_id': 0,
        'start_time': 0,
        'delay_budget_ms': 50,
        'strace_file_check': True,
        'strace_file': str(strace_in)
    }
    step = 1e6/60
    unpack = DePacketizer(strace_in, cfg)
    assert unpack.delay_budget > step, 'timestep must be higher than delay budget'
    packets = PacketStream(ptrace_in)
    t0 = 0 # packets.peak().time_stamp_in_micro_s

    with open(strace_out, 'w') as f:
        writer = STraceTx.get_csv_writer(f)
        for ts in gen_timesteps(step, t=t0):
            if packets.is_empty():
                break
            t_bucket = packets.pop(ts)
            for s in unpack.process(ts, t_bucket):
                # decode with incoming straces ...
                writer.writerow(s.get_csv_dict())
