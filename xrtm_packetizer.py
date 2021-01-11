import sys
import argparse
import csv
import re
import json
from pathlib import Path
from typing import Iterable

from xrtm.packets import Packetizer, PTraceTx
from xrtm.models import STraceTx
from xrtm.utils import _requires, ConfigException

class PacketizerConfig:

    def __init__(self):
        self.source = None
        self.start_time = None
        self.pckt_max_size = None
        self.pckt_overhead = None
        self.bitrate = None
        self.output = None

    @classmethod
    def parse_config(cls, data:dict):
        cfg = cls()
        _requires("S-Trace", data, "invalid config. no source S-Trace definition")
        _requires("source", data["S-Trace"], "invalid config. no source S-Trace definition")
        cfg.source = Path(data["S-Trace"].get("source"))
        cfg.start_time = int(data["S-Trace"].get("startTime", 0))

        _requires("Packet", data, "invalid config. no Packet definition")
        packet = data["Packet"]
        cfg.pckt_max_size = int(packet.get("maxSize", 1500)) # bytes
        cfg.pckt_overhead = int(packet.get("overhead", 40))

        cfg.bitrate = int(data.get("Bitrate", 45000000)) # bits/s
        _requires("P-Trace", data, "invalid config. no P-Trace output definition")
        cfg.output = Path(data["P-Trace"])
        return cfg

    def get_strace_source(self, user_idx=-1):
        p = Path(self.source)
        if user_idx < 0:
            return p
        else:
            return p.parent / f'{p.stem}[{str(user_idx)}]{p.suffix}'

    def get_ptrace_output(self, user_idx=-1):
        p = Path(self.output)
        if user_idx < 0:
            return p
        else:
            return p.parent / f'{p.stem}[{str(user_idx)}]{p.suffix}'

    @classmethod
    def load(cls, p:Path):
        with open(p, 'r') as f:
            cfg = json.load(f)
            return cls.parse_config(cfg)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Model encoder configuration')
    parser.add_argument('-c', '--config', type=str, help='packetizer config', required=True)
    parser.add_argument('-u', '--user_id', type=int, help='user id', required=False, default=-1)
    
    args = parser.parse_args()
    
    cfg_path = Path(args.config)
    assert cfg_path.exists(), 'config file not found'

    cfg = PacketizerConfig.load(cfg_path)
    pack = Packetizer(cfg, user_idx=args.user_id)

    with open(cfg.get_ptrace_output(args.user_id), 'w') as f:
        writer = PTraceTx.get_csv_writer(f)

        source = cfg.get_strace_source(args.user_id)
        seqnum = 0
        for trace in STraceTx.iter_csv_file(source):
            rate_delay = 0
            for p in pack.process([trace], seqnum):

                p.s_trace = str(source)
                p.time_stamp_in_micro_s += round(p.time_stamp_in_micro_s + rate_delay)
                writer.writerow(p.get_csv_dict())

                rate_delay += cfg.bitrate / (p.size * 8)
                seqnum += 1