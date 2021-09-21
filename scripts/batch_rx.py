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

from typing import Iterable, List

use_cases = [f'vr2-{i}' for i in range(1,9)]
USERS = [*range(16)]

batch = {}

def run_process(*args, dry_run=False, **kwargs):
    if dry_run:
        p = args[0]
        if p not in batch:
            batch[p] = []
        batch[p].append(args)
        return
    print('python3', *args)
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

def gen_decoder_configs(template:dict, unpack_cfgs:dict, output:Path):
    for user_idx, unpack_cfg in unpack_cfgs.items():
        for o in unpack_cfg["Output"]:
            cfg = deepcopy(template)
            cfg["Input"]["Sp-Trace"] = o["Sp-Trace"]
            s = Path(cfg["Input"]["S-Trace"])
            buffer_idx = o["buffer"]
            cfg["Input"]["S-Trace"] = str(s.parent / f'{s.stem}[{user_idx}]-{buffer_idx}.csv')
            cfg["maxDelay"] = unpack_cfg["maxDelay"]
            cfg["Vp-Trace"] = str(output.parent / f'{output.stem}[{user_idx}]-{buffer_idx}.csv')
            yield user_idx, buffer_idx, cfg
        
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


def gen_scenario(key:str, input_dir:Path, output_dir:Path,  users:List[int], dry_run=True, overwrite=False, max_delay_ms=50, **kwargs):
    input_dir = input_dir / key
    # assert input_dir.exists()
    output_dir = output_dir / key
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
            output_dir.mkdir()
        else:
            print('output dir already exists. use -y to overwritte')
    else:
        output_dir.mkdir(parents=True)

    p_path = input_dir / f'P-Trace_{key}.csv'
    s_path = input_dir / f'S-Trace_{key}.csv'

    pp_path = Path(kwargs.get("pp_dir", output_dir)) / f'Pp-Trace_{key}.csv'
    sp_path = output_dir / f'Sp-Trace_{key}.csv'

    
    # TEST CHANNEL ###################################################################
    testchan_template = {
        "loss_rate" : 0.01,
        "max_delay_ms" : 50,
        "max_bitrate" : 60000000,
        "P-Trace": p_path,
        "Pp-Trace": pp_path
    }
    testchans = { user_idx: cfg for user_idx, cfg in gen_testchan_configs(testchan_template, USERS) }
    if kwargs.get('testchan', False):
        for user_idx, cfg in testchans.items():
            cfg_path = output_dir / f'testchan_cfg[{user_idx}].json'
            with open(cfg_path, 'w') as fp:
                json.dump(cfg, fp, indent=4)
            run_process('./xrtm_test_channel.py', '-c', str(cfg_path), dry_run=dry_run)
            run_process('./plot.py', '-p', str(cfg["Pp-Trace"]), '-n', dry_run=dry_run)

    # UNPACK ###################################################################
    unpack_template = {
        "Input": {
            "Pp-Trace": pp_path,
        },
        "maxDelay": max_delay_ms,
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

    if kwargs.get('unpack', False):
        for user_idx, cfg in unpacks.items():
            cfg_path = output_dir / f'depacketizer_cfg[{user_idx}].json'
            with open(cfg_path, 'w') as fp:
                json.dump(cfg, fp, indent=4)
            run_process('./xrtm_depacketizer.py', '-c', str(cfg_path), f'--strace-dir={input_dir}', dry_run=dry_run)


    # DECODER ###################################################################
    decoder_template = {
        "frame_width": 2048,
        "frame_height": 2048,
        "frame_rate": 60,
        "Input": {
            "S-Trace": s_path,
            "Sp-Trace": sp_path,
        }
    }
    if kwargs.get('decode', False):
        for user_idx, buffer_idx, decoder_cfg in gen_decoder_configs(decoder_template, unpacks, output=output_dir/'Vp-Trace.csv'):
            cfg_path = output_dir / f'decoder_cfg[{user_idx}]-{buffer_idx}.json'
            with open(cfg_path, 'w') as fp:
                json.dump(decoder_cfg, fp, indent=4)
            run_process('./xrtm_decoder.py', '-c', str(cfg_path), dry_run=dry_run)

    # QUALITY ###################################################################
    if kwargs.get('quality', False):
        cfg = gen_qeval_configs(unpacks, output=str(output_dir/'Q-Trace.csv'))
        cfg_path = output_dir / f'quality_cfg.json'
        with open(cfg_path, 'w') as fp:
            json.dump(cfg, fp, indent=4)
        run_process('./xrtm_quality.py', '-c', str(cfg_path), dry_run=dry_run)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='P-Trace generator')
    parser.add_argument("-k", '--key', help='scenario key', type=str, required=True)
    parser.add_argument("-u", '--users', help='how many users to generate', type=int, required=True)
    parser.add_argument('-m', '--max_delay_ms', help='max delay', type=int, default=80, required=True)
    
    parser.add_argument('-i', '--input_dir', help='input dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', help='output dir', type=str, required=True)
    parser.add_argument('-y', '--overwrite', help='overwrite output dir', action='store_true', required=False)
    # parser.add_argument("-u", '--users', help='optional list of user IDs', nargs="+", type=int, default=[], required=False)
    
    parser.add_argument("-T", "--testchan", help='',  action='store_true', required=False)
    parser.add_argument("--pp_dir", help='', type=str, default=None, required=False)

    parser.add_argument("-U", "--unpack", help='',  action='store_true', required=False)
    # parser.add_argument("--sp_dir", help='', type=str, default=None, required=False)

    parser.add_argument("-D", "--decode", help='',  action='store_true', required=False)
    # parser.add_argument("--vp_dir", help='', type=str, default=None, required=False)

    parser.add_argument("-Q", "--quality", help='',  action='store_true', required=False)
    # parser.add_argument("--qt_dir", help='', type=str, default=None, required=False)

    parser.add_argument('-x', '--run', help='generate config and run the scripts immediately', action='store_true', required=False)

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir)

    USERS = [*range(args.users)]
    
    kwargs = {
        kw: getattr(args, kw) for kw in [
            "testchan", 
            "pp_dir", 
            "unpack", 
            "decode", 
            "quality",
            "max_delay_ms"
        ] if hasattr(args, kw)
    }
    
    dry_run = not args.run
    if args.key != None:
        use_cases = args.key.split(',')
    for key in use_cases:
        gen_scenario(key, input_dir, output_dir, USERS, dry_run, args.overwrite, **kwargs)
    
    if dry_run:
        for p, cmds in batch.items():
            print("\n"+("#"*128),"\n")
            for u, args in enumerate(cmds):
                if not (u % len(USERS)):
                    print("\n")
                print('python3', *args)
