from random import randint
from typing import Iterable, List, Iterator
from math import ceil
from enum import Enum
from pathlib import Path

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
        yield PTraceTx.from_strace(s, size=max_payload_size, number=seqnum, number_in_unit=fragnum, last_in_unit=False)
        bytes_to_pack -= max_payload_size
        seqnum += 1
        fragnum += 1
    yield PTraceTx.from_strace(s, size=bytes_to_pack, number=seqnum, number_in_unit=fragnum, last_in_unit=True)

def unpack(*packets:List[PTraceTx]) -> int:
    """
    asserts a list of packets can be decoded into a slice
    """
    p = packets[0]
    if len(packets) == 1:
        return p.index
    user_id = p.user_id
    view_idx = p.buffer
    for _, p in enumerate(sorted(packets, key=lambda x: x.fragnum)):
        if p.is_last:
            assert p.fragnum == len(packets)-1
        assert user_id == p.user_id
        assert view_idx == p.view_idx


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


class JitterBuffer:

    def __init__(self):
        self._buffer = {}
    
    # TODO: should receive PTraceRx
    def append(self, p:PTraceTx):
        if p.index not in self._buffer:
            self._buffer[p.index] = [p]
        else:
            self._buffer[p.index].append(p)

    def delete(self, idx):
        assert idx in self._buffer
        self._buffer.pop(idx)

    def iter_late_slices(self, deadline:int) -> Iterator[int]:
        for idx, slice_packets in self._buffer.items():
            for p in slice_packets:
                if p.time_stamp_in_micro_s <= deadline:
                    yield idx
                    continue
    
    def iter_complete_slices(self) -> Iterator[List[int]]:
        for idx, slice_packets in self._buffer.items():
            if len(slice_packets) == 1 and not slice_packets[0].is_fragment():
                yield idx, slice_packets[0].delay
            else:
                packets = sorted(slice_packets, key=lambda x: x.number_in_unit)
                p = packets[-1]
                if p.last_in_unit and p.number_in_unit == len(packets)-1:
                    yield idx, p.delay


class StereoJitterBuffer:

    def __init__(self):
        self.left_buffer = JitterBuffer()
        self.right_buffer = JitterBuffer()

    # TODO: should receive PTraceRx
    def append(self, p:PTraceTx):
        if p.buffer == 1:
            self.left_buffer.append(p)
        elif p.buffer == 2:
            self.right_buffer.append(p)
        else:
            raise ValueError(f'invalid eye buffer {p.buffer}')

    def delete(self, slice_idx:int):
        # TODO: needs improvements as deleting from both buffers raises a KeyError
        try:
            self.left_buffer.delete(slice_idx)
        except AssertionError:
            pass
        try:
            self.right_buffer.delete(slice_idx)
        except AssertionError:
            pass

    def iter_late_slices(self, deadline:int) -> Iterator[int]:
        for s in self.left_buffer.iter_late_slices(deadline):
            yield s
        for s in self.right_buffer.iter_late_slices(deadline):
            yield s

    def iter_complete_slices(self) -> Iterator[List[int]]:
        for s, delay in self.left_buffer.iter_complete_slices():
            yield s, delay
        for s, delay in self.right_buffer.iter_complete_slices():
            yield s, delay


class DePacketizer:

    def __init__(self, strace_file:Path, cfg={}):
        """
        a class to process timestamped buckets of incoming PTrace into decodable Strace
        """
        self.user_id = getattr(cfg, 'user_id', 0)
        self.time = getattr(cfg, 'start_time', 0)
        self.delay_budget = 1e3 * getattr(cfg, 'delay_budget', 50)
        self.buffer = StereoJitterBuffer()
        self.strace_file_check = getattr(cfg, 'strace_file_check', True)
        self.strace_file = getattr(cfg, 'strace_file', str(strace_file)) 
        self.strace_data = [*STraceTx.iter_csv_file(strace_file)]
    
    def process(self, timestamp:int, packets:List[PTraceTx]) -> List[STraceTx]:
        for p in packets:
            assert p.user_id == self.user_id
            if self.strace_file_check:
                assert p.s_trace == self.strace_file
            self.buffer.append(p)

        # remove late packets from the buffer
        deadline = timestamp - self.delay_budget
        for idx in [*self.buffer.iter_late_slices(deadline)]:
            self.buffer.delete(idx)

        # yield index of slices that can be reconstructed from buffered packets
        for idx, delay in [*self.buffer.iter_complete_slices()]:
            self.buffer.delete(idx)
            s = self.strace_data[idx]
            s.time_stamp_in_micro_s += delay
            yield s

