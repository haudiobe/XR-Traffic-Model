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
from xrtm.quality import QTrace, QualEvalCfg, QualEvalInput, q_eval

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='XRTM Quality evaluation tool')
    parser.add_argument('-c', '--config', type=str, help='quality eval config', required=True)    
    args = parser.parse_args()
    
    cfg_path = Path(args.config)
    assert cfg_path.exists(), 'config file not found'

    cfg = QualEvalCfg.load(cfg_path)
    pack = Packetizer(cfg, user_idx=args.user_id)

    q_all = QTrace()

    with open(cfg.output, 'w') as of:
        writer = QTrace.get_csv_writer(of)
        for qi in cfg.iter_inputs():
            q_user = q_eval(60, qi)
            writer.writerow(q_user.get_csv_dict())

            q_all.total_packets += q_user.total_packets
            q_all.lost_packets += q_user.lost_packets
            q_all.late_packets += q_user.late_packets
            q_all.total_slices += q_user.total_slices
            q_all.lost_slices += q_user.lost_slices
            q_all.buffer = -1
            q_all.user = -1
            q_all.duration = max(q_user, q_all.duration)
        
        writer.writerow(q_all.get_csv_dict())
        