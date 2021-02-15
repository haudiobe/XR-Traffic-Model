from random import randint
import functools
from typing import Iterable, List, Iterator, Tuple
from math import ceil
from enum import Enum
from pathlib import Path
import json

from xrtm.utils import _requires, ConfigException

from .models import STraceTx, PTraceTx, PTraceRx, STraceRx, CSV, CsvRecord

def pack(i:int, s:STraceTx, mtu=1500, header_size=40) -> Iterable[PTraceTx]:
    seqnum = i
    if mtu <= 0:
        bytes_to_pack = header_size + s.size
        yield PTraceTx.from_strace(s, size=bytes_to_pack, number=i, number_in_unit=0, last_in_unit=True)
        return
    fragnum = 0
    max_payload_size = mtu - header_size
    bytes_to_pack = s.size
    while bytes_to_pack > max_payload_size:
        yield PTraceTx.from_strace(s, size=mtu, number=seqnum, number_in_unit=fragnum, last_in_unit=False)
        bytes_to_pack -= max_payload_size
        seqnum += 1
        fragnum += 1
    bytes_to_pack += header_size
    yield PTraceTx.from_strace(s, size=bytes_to_pack, number=seqnum, number_in_unit=fragnum, last_in_unit=True)


########################################################################################################################

class PacketizerCfg:

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
        cfg.pckt_max_size = int(packet.get("maxSize", 1500)) # bytes, negative value means unrestricted
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


class Packetizer:

    def __init__(self, cfg, user_idx=-1):
        self.cfg = cfg
        self.user_id = max(user_idx, 0)

    def process(self, slices:Iterable[STraceTx], seqnum = 0) -> Iterable[PTraceTx]:
        seqnum = seqnum
        for s in slices:
            for p in pack(seqnum, s, mtu=self.cfg.pckt_max_size, header_size=self.cfg.pckt_overhead):
                seqnum += 1
                p.user_id = self.user_id
                p.delay = 0
                p.index = s.index
                p.render_timing = s.render_timing
                p.time_stamp_in_micro_s = s.time_stamp_in_micro_s
                p.type = s.type
                p.buffer = s.buffer
                p.s_trace = None
                yield p

########################################################################################################################

class SliceLossMode(Enum):
    PARTIAL_RECOVERY = "partial"
    NO_RECOVERY = "drop"

    @classmethod
    def parse(cls, value:str):
        for e in cls.__members__.values():
            if value == e.value:
                return e
    

class DePacketizerCfg:

    __slice_loss_modes = [
        SliceLossMode.PARTIAL_RECOVERY,
        SliceLossMode.NO_RECOVERY
    ]

    def __init__(self, pp_input:Path, buffers:dict, delay_budget_ms=60, slice_loss_mode="partial"): # user_id:int=0, strace_file=None):
        self.delay_budget = delay_budget_ms * 1e3
        self.slice_loss_mode = slice_loss_mode
        self.pp_input = pp_input
        p = next(PTraceTx.iter_csv_file(pp_input))
        self.strace_file = Path(p.s_trace)
        self.buffers = buffers
        assert self.slice_loss_mode in self.__slice_loss_modes, 'invalid slice loss/recovery mode'

    @classmethod
    def load(cls, fp:str):
        with open(fp, 'r') as f:
            cfg = json.load(f)
            pp_input = cfg.get('Input')
            assert (pp_input != None) and ("Pp-Trace" in pp_input)
            pp_trace = Path(pp_input.get("Pp-Trace"))
            assert pp_trace.exists(), f'Pp-Trace file not found: {pp_trace}'

            slice_loss_mode = SliceLossMode.parse(cfg.get('sliceLossMode', None))
            assert (slice_loss_mode != None) and (slice_loss_mode in cls.__slice_loss_modes), f'"sliceLossMode" must be one of ["partial", "drop"]. invalid value: {slice_loss_mode}'
            
            delay_budget_ms = cfg.get('maxDelay')
            assert delay_budget_ms != None, "maxDelay is required (eg. 60)"

            out = cfg.get("Output", [])
            assert len(out) > 0

            buffers = {}
            for buff in out:
                assert "buffer" in buff, "missing buffer id"
                assert "Sp-Trace" in buff,  "missing buffer output path"
                buffer_idx = buff["buffer"]
                buffers[buffer_idx] = buff["Sp-Trace"]
            
            return cls(
                delay_budget_ms = delay_budget_ms,
                slice_loss_mode = slice_loss_mode, 
                pp_input = pp_trace,
                buffers = buffers
            )

def assert_sorted(l):
    pre = None
    for p in l:
        if pre != None:
            assert p.number_in_unit == (pre.number_in_unit+1)
        pre = p
    return l


class DePacketizer:

    def __init__(self, cfg:DePacketizerCfg):
        """ a class to process timestamped buckets of incoming PTrace into decodable Strace """
        self.cfg = cfg
        self.buffer = StereoJitterBuffer(cfg)
        self.strace_data = [*STraceTx.iter_csv_file(cfg.strace_file)]

    def process(self, packets:List[PTraceTx]) -> List[STraceRx]:
        for p in packets:
            self.buffer.append(p)
            for slice_idx, pp_sorted in self.buffer.iter_slices():
                # pp_sorted = assert_sorted(pp_slice)
                # if self.is_complete_slice(pp_sorted):
                s = self.strace_data[slice_idx]
                pp_recovered = [pp for pp in pp_sorted if ((pp.time_stamp_in_micro_s - pp.render_timing) <= self.cfg.delay_budget)]
                if (len(pp_recovered) < len(pp_sorted)) and (self.cfg.slice_loss_mode == SliceLossMode.NO_RECOVERY):
                    s.recovery_position = 0
                    s.time_stamp_in_micro_s = 0
                elif len(pp_recovered) == 0:
                    s.recovery_position = 0
                    s.time_stamp_in_micro_s = 0
                else:
                    s.recovery_position = self.get_recovered_bytes(pp_recovered, pp_overhead=40)
                    s.time_stamp_in_micro_s = self.get_recovered_timestamp(pp_recovered)
                self.buffer.delete(slice_idx)
                yield s
            
    def is_complete_slice(self, pp_sorted):
        for num_in_unit, p in enumerate(pp_sorted):
            if p.number_in_unit != num_in_unit:
                return False
        return p.last_in_unit == 1
    
    def get_recovered_bytes(self, pp_recovered, pp_overhead) -> int:
        return functools.reduce(lambda x,y: x+(y.size-pp_overhead), pp_recovered, 0 )

    def get_recovered_timestamp(self, pp_recovered) -> int:
        return max(map(lambda x: x.time_stamp_in_micro_s, pp_recovered))


class JitterBuffer:

    def __init__(self, cfg:DePacketizerCfg):
        self._buffer = {}
        self._complete = {}
        self.max_delay = cfg.delay_budget # micro_s
        self.slice_loss_mode = cfg.slice_loss_mode
    
    def append(self, p:PTraceTx):
        if p.index not in self._buffer:
            self._buffer[p.index] = [p]
        elif p.number_in_unit > self._buffer[p.index][-1].number_in_unit:
            self._buffer[p.index].append(p)
        else:
            idx = 0
            while (p.number_in_unit > self._buffer[p.index][idx]) and (idx < len(self._buffer[p.index])):
                idx += 1
            self._buffer[p.index].insert(idx, p)
        p_last = self._buffer[p.index][-1]
        if p_last.last_in_unit and p_last.number_in_unit == len(self._buffer[p.index]):
            self._complete[p.index] = self._buffer.pop(p.index)

    def delete(self, idx):
        if idx in self._complete:
            self._complete.pop(idx)
        if idx in self._buffer:
            self._buffer.pop(idx)

    def iter_slices(self) -> Iterator[Tuple[int, List[PTraceRx]]]:
        for slice_idx, slice_packets in self._complete.copy().items():
            yield slice_idx, slice_packets

    def has(self, slice_idx:int):
        return slice_idx in self._buffer


class StereoJitterBuffer:

    def __init__(self, *args, **kwargs):
        self.buffers = {
            0: JitterBuffer(*args, **kwargs),
            1: JitterBuffer(*args, **kwargs)
        }

    def append(self, p:PTraceTx):
        buff = self.buffers[p.buffer]
        buff.append(p)

    def delete(self, slice_idx:int):
        for buff in self.buffers.values():
            try:
                buff.delete(slice_idx)
                return
            except AssertionError:
                pass
        raise ValueError(f'{slice_idx} is not buffered')
    
    def iter_slices(self) -> Iterator[Tuple[int, List[PTraceRx]]]:
        for buff in self.buffers.values():
            for slice_idx, slice_packets in buff.iter_slices():
                yield slice_idx, slice_packets
