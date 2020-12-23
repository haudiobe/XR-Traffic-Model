import sys
import argparse
import csv
from pathlib import Path
from typing import Iterable
from xrtm.packetizer import Packetizer, PTraceTx
from xrtm.models import STraceTx

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Model encoder configuration')
    parser.add_argument('-i', '--s_trace', type=str, help='s-trace csv input', required=True)
    parser.add_argument('-o', '--p_trace', type=str, help='p-trace csv output', required=True)
    args = parser.parse_args()
    
    strace_in = Path(args.s_trace)
    assert strace_in.exists()
    
    ptrace_out = Path(args.p_trace)
    
    pack = Packetizer(constant_delay=5, jitter_min=0, jitter_max=5, user_id=0)
    straces = STraceTx.iter_csv_file(strace_in)
    with open(ptrace_out, 'w') as f:
        writer = PTraceTx.get_csv_writer(f)
        for p in pack.process(straces):
            writer.writerow(p.get_csv_dict())
    
