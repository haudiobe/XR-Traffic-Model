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
    
    cfg = None
    with open(cfg_path, 'r') as cf:
       cfg = QualEvalCfg.load(cf)

    q_all = QTrace({ "buffer": -1, "user": -1 })
    qi_count = 0

    q_buff_sum = {}
    q_buff_count = {}

    with open(cfg.output, 'w') as of:
        writer = QTrace.get_csv_writer(of)

        for qi in cfg.iter_inputs():

            q_user = q_eval(80e3, qi)
            writer.writerow(q_user.get_csv_dict())

            ####################################################################

            if q_user.buffer not in q_buff_sum:
                q_buff_sum[q_user.buffer] = QTrace({ "buffer": q_user.buffer, "user": -1 })
                q_buff_count[q_user.buffer] = 0

            q_buff_count[q_user.buffer] += 1
            q_buff = q_buff_sum[q_user.buffer]

            q_buff.total_packets += q_user.total_packets
            q_buff.lost_packets += q_user.lost_packets
            q_buff.late_packets += q_user.late_packets
            q_buff.total_slices += q_user.total_slices
            q_buff.lost_slices += q_user.lost_slices
            q_buff.slice_recovery += q_user.slice_recovery

            q_buff.correct_CUs += q_user.correct_CUs
            q_buff.damaged_CUs += q_user.damaged_CUs
            q_buff.lost_CUs += q_user.lost_CUs
            q_buff.total_CUs += q_user.total_CUs

            q_buff.PSNR += q_user.PSNR
            q_buff.PSNRyuv += q_user.PSNRyuv
            q_buff.RPSNR += q_user.RPSNR
            q_buff.RPSNRyuv += q_user.RPSNRyuv

            ####################################################################

            q_all.total_packets += q_user.total_packets
            q_all.lost_packets += q_user.lost_packets
            q_all.late_packets += q_user.late_packets
            q_all.total_slices += q_user.total_slices
            q_all.lost_slices += q_user.lost_slices
            q_all.slice_recovery += q_user.slice_recovery

            q_all.correct_CUs += q_user.correct_CUs
            q_all.damaged_CUs += q_user.damaged_CUs
            q_all.lost_CUs += q_user.lost_CUs
            q_all.total_CUs += q_user.total_CUs

            q_all.PSNR += q_user.PSNR
            q_all.PSNRyuv += q_user.PSNRyuv
            q_all.RPSNR += q_user.RPSNR
            q_all.RPSNRyuv += q_user.RPSNRyuv
            
            qi_count += 1

        ###################################################################

        for idx, q_buff in q_buff_sum.items():

            q_buff.PLoR = q_buff.lost_packets / q_buff.total_packets
            q_buff.PLaR = q_buff.late_packets / q_buff.total_packets
            q_buff.SLR = q_buff.lost_slices / q_buff.total_slices
    
            q_buff.CAR = q_buff.correct_CUs / q_buff.total_CUs
            q_buff.DAR = q_buff.damaged_CUs /q_buff.total_CUs
            q_buff.ALR = q_buff.lost_CUs / q_buff.total_CUs
            q_buff.LDR = q_buff.ALR + q_buff.DAR

            q_buff.PSNR /= q_buff_count[idx]
            q_buff.PSNRyuv /= q_buff_count[idx]
            q_buff.RPSNR /= q_buff_count[idx]
            q_buff.RPSNRyuv /= q_buff_count[idx]

        ###################################################################
        q_all.slice_recovery /= qi_count
        q_all.PLoR = q_all.lost_packets / q_all.total_packets
        q_all.PLaR = q_all.late_packets / q_all.total_packets
        q_all.SLR = q_all.lost_slices / q_all.total_slices
    
        q_all.CAR = q_all.correct_CUs / q_all.total_CUs
        q_all.DAR = q_all.damaged_CUs /q_all.total_CUs
        q_all.ALR = q_all.lost_CUs / q_all.total_CUs
        q_all.LDR = q_all.ALR + q_all.DAR

        q_all.PSNR /= qi_count
        q_all.PSNRyuv /= qi_count
        q_all.RPSNR /= qi_count
        q_all.RPSNRyuv /= qi_count

        writer.writerow(q_all.get_csv_dict())
        