#!/usr/bin/python3
import subprocess
import random
from pathlib import Path
import shutil
import json
import argparse
from xrtm_test_channel import TestChan, TestChanCfg
from xrtm.utils import user_buffer_from_path
from copy import deepcopy

from typing import Iterable

root_dir = './VR'
use_cases = [f'vr2-{i}' for i in range(1,9)]
USERS = [0,1,2,3] # [*range(16)]

def run_process(*args, dry_run=False, **kwargs):
    print('python3', *args)
    if dry_run:
        return
    subprocess.run(['python3', *args])

def gen_testchan_configs(template:dict, users_idx:Iterable[int]):
    for user_idx in users_idx:
        cfg = deepcopy(template)
        op = Path(template["P-Trace"])
        cfg["P-Trace"] = str(op.with_name(f'{op.stem}[{user_idx}].csv'))
        opp = Path(template["Pp-Trace"])
        cfg["Pp-Trace"] = str(opp.with_name(f'{opp.stem}[{user_idx}].csv'))
        yield user_idx, cfg

def gen_unpack_configs(template:dict, testchan_cfgs:dict):
    for user_idx, testchan_cfg in testchan_cfgs.items():
        cfg = deepcopy(template)
        cfg["Input"]["Pp-Trace"] = str(testchan_cfg["Pp-Trace"])
        ifp = Path(testchan_cfg["Pp-Trace"])
        for o in cfg["Output"]:
            o["Sp-Trace"] = str(ifp.parent / f'Sp-Trace[{user_idx}]-{o["buffer"]}.csv')
        yield user_idx, cfg

def gen_qeval_configs(unpack_cfgs:dict, output='./Quality.csv'):
    cfg = {
        "Users": [],
        "Q-Trace": str(output)
    }
    for user_idx, unpack_cfg in unpack_cfgs.items():
        u = {
            "Id": user_idx,
            "Inputs": []
        }
        for buff in unpack_cfg["Output"]:
            b = deepcopy(buff)
            b["Pp-Trace"] = unpack_cfg["Input"]["Pp-Trace"]
            u["Inputs"].append(b)
        cfg["Users"].append(u)
    return cfg

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='P-Trace generator')
    parser.add_argument('-i', '--input_dir', help='input dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', help='output dir', type=str, required=True)
    parser.add_argument('-y', '--overwrite', help='overwrite output dir', action='store_true', required=False)
    parser.add_argument('-d', '--dry_run', help='generate configs, do not run the scripts', action='store_true', required=False)
    parser.add_argument("-u", '--users', help='optional list of user IDs', nargs="+", type=int, default=[], required=False)
    
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
            output_dir.mkdir()
        else:
            raise ValueError('output dir already exists. use -y to overwritte')
    else:
        output_dir.mkdir()
    
    input_dir = Path(args.input_dir)
    assert input_dir.exists()
    p_path = input_dir / 'P-Trace_VR2-1.csv'
    s_path = input_dir / 'S-Trace_VR2-1.csv'
    pp_path = output_dir / 'Pp-Trace_VR2-1.csv'
    sp_path = output_dir / 'Sp-Trace_VR2-1.csv'

    if len(args.users) > 0:
        USERS = args.users
    
    # TEST CHANNEL ###################################################################
    testchan_template = {
        "loss_rate" : 0.01,
        "max_delay_ms" : 50,
        "max_bitrate" : 60000000,
        "P-Trace": p_path,
        "Pp-Trace": pp_path
    }
    testchans = { user_idx: cfg for user_idx, cfg in gen_testchan_configs(testchan_template, USERS) }
    for user_idx, cfg in testchans.items():
        cfg_path = output_dir / f'testchan_cfg[{user_idx}].json'
        with open(cfg_path, 'w') as fp:
            json.dump(cfg, fp, indent=4)
        run_process('./xrtm_test_channel.py', '-c', str(cfg_path), dry_run=args.dry_run)
        run_process('./plot.py', '-p', str(cfg["Pp-Trace"]), '-n', dry_run=args.dry_run)

    # UNPACK ###################################################################
    unpack_template = {
        "Input": { 
            "Pp-Trace": pp_path,
        },
        "maxDelay": 80,
        "sliceLossMode": "partial",
        "Output": [
            {
                "buffer": 0
            },
            {
                "buffer": 1
            }
        ]
    }
    unpacks = {}
    for user_idx, cfg in gen_unpack_configs(unpack_template, testchans):
        unpacks[user_idx] = cfg
    for user_idx, cfg in unpacks.items():
        cfg_path = output_dir / f'depacketizer_cfg[{user_idx}].json'
        with open(cfg_path, 'w') as fp:
            json.dump(cfg, fp, indent=4)
        run_process('./xrtm_depacketizer.py', '-c', str(cfg_path), dry_run=args.dry_run)

    # QUALITY ###################################################################
    cfg = gen_qeval_configs(unpacks, output=str(output_dir/'Q-Trace.csv'))
    cfg_path = output_dir / f'quality_cfg.json'
    with open(cfg_path, 'w') as fp:
        json.dump(cfg, fp, indent=4)
    run_process('./xrtm_quality.py', '-c', str(cfg_path), dry_run=args.dry_run)
    


