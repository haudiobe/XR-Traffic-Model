#!/usr/bin/python3
import subprocess
import random
from pathlib import Path
import shutil
import json
import argparse

users = {
    'start_frame': [ 1 ] # , 1801, 901, 2701, 451, 2251, 1351, 3251, 226, 2026, 1126, 2926, 676, 2476, 1576, 3476 ]
}

def gen_user_configs(base_cfg:dict):

    for idx, start_frame in enumerate(users['start_frame']):
        cfg = base_cfg.copy()
        cfg['start_frame'] = start_frame
        yield idx, cfg


class Project(object):

    def __init__(self, encode_cfg:str, packet_cfg:str, output_dir:str):
        self.encode_cfg_path = Path(encode_cfg)
        assert Path(self.encode_cfg_path).exists()
        with open(encode_cfg, 'r') as fp:
            self.encode_cfg = json.load(fp)
        
        self.packet_cfg_path = Path(packet_cfg)
        assert Path(self.packet_cfg_path).exists()
        with open(packet_cfg, 'r') as fp:
            self.packet_cfg = json.load(fp)
        
        self.output_dir = Path(output_dir)
    
    
    def get_strace_path(self, user_id:int=None):
        p = Path(self.encode_cfg['S-Trace'])
        if user_id != None:
            return self.output_dir / f'{p.stem}[{user_id}]{p.suffix}'
        return self.output_dir / f'{p.stem}{p.suffix}'

    def get_ptrace_path(self, user_id:int=None):
        p = Path(self.packet_cfg['P-Trace'])
        if user_id != None:
            return self.output_dir / f'{p.stem}[{user_id}]{p.suffix}'
        return self.output_dir / f'{p.stem}{p.suffix}'

    def save_strace_cfg(self, user_id:int, cfg:dict):
        cfg['S-Trace'] = str(self.get_strace_path())
        p = self.output_dir / f'{self.encode_cfg_path.stem}[{user_id}]{self.encode_cfg_path.suffix}'
        with open(p, 'w') as fp:
            json.dump(cfg, fp, indent=4)
        return p

    def save_ptrace_cfg(self):
        cfg = P.packet_cfg.copy()
        cfg['S-Trace']['source'] = str(self.get_strace_path())
        cfg['P-Trace'] = str(self.get_ptrace_path())
        p = self.output_dir / self.packet_cfg_path.name
        with open(p, 'w') as fp:
            json.dump(cfg, fp, indent=4)
        return p


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='P-Trace generator')
    parser.add_argument('-s', '--encode_cfg', help='base encoder config', type=str, required=True)
    parser.add_argument('-p', '--packet_cfg', help='base packetizer config', type=str, required=True)
    parser.add_argument('-o', '--output_dir', help='output dir', type=str, required=True)
    parser.add_argument('-y', '--overwrite', help='overwrite output dir', action='store_true', required=False)
    parser.add_argument('-S', '--plot-only-strace', help='plot only strace',  action='store_true', required=False)
    parser.add_argument('-P', '--plot-only-ptrace', help='plot only ptrace',  action='store_true', required=False)
    
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
            output_dir.mkdir()
        # raise ValueError('output dir already exists. use -y to overwritte')
    else:
        output_dir.mkdir()

    P = Project(args.encode_cfg, args.packet_cfg, args.output_dir)
    packet_cfg = P.save_ptrace_cfg()

    plot_only = (args.plot_only_strace or args.plot_only_ptrace)

    for idx, encoder_cfg in gen_user_configs(P.encode_cfg):
        encoder_cfg_path = P.save_strace_cfg(idx, encoder_cfg)
        if not plot_only:
            cmd = ['python3', './xrtm_encoder.py', '-c', str(encoder_cfg_path), '-u', str(idx)]
            print(cmd)
            subprocess.run(['python3', './xrtm_encoder.py', '-c', str(encoder_cfg_path), '-u', str(idx)])
            subprocess.run(['python3', './xrtm_packetizer.py', '-c', str(packet_cfg), '-u', str(idx)])
        if args.plot_only_strace or (not plot_only):
            subprocess.run(['python3', './plot.py', '-s', P.get_strace_path(idx), '-n'])
        if args.plot_only_ptrace or (not plot_only):
            subprocess.run(['python3', './plot.py', '-p', P.get_ptrace_path(idx), '-n'])
        

        