import sys
import argparse
import csv
import re
import json
from pathlib import Path
from typing import Iterable

from xrtm.packets import Packetizer, PTraceTx, PacketizerCfg
from xrtm.models import STraceTx
from xrtm.utils import _requires, ConfigException


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='XRTM Packetizer')
    parser.add_argument('-c', '--config', type=str, help='packetizer config', required=True)
    parser.add_argument('-u', '--user_id', type=int, help='user id', required=False, default=-1)
    
    args = parser.parse_args()
    
    cfg_path = Path(args.config)
    assert cfg_path.exists(), 'config file not found'

    cfg = PacketizerCfg.load(cfg_path)
    pack = Packetizer(cfg, user_idx=args.user_id)

    with open(cfg.get_ptrace_output(args.user_id), 'w') as f:
        writer = PTraceTx.get_csv_writer(f)
        source = cfg.get_strace_source(args.user_id)
        seqnum = 0
        prev_pckt_timestamp = 0
        for s_trace in STraceTx.iter_csv_file(source):
            for p in pack.process([s_trace], seqnum):
                p.s_trace = str(source)
                rate_delay = 8e6 * p.size / cfg.bitrate
                p.time_stamp_in_micro_s = round(max(s_trace.time_stamp_in_micro_s, prev_pckt_timestamp) + rate_delay)
                prev_pckt_timestamp = p.time_stamp_in_micro_s
                writer.writerow(p.get_csv_dict())
                seqnum += 1
