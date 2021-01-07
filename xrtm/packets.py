from random import randint
from typing import Iterable, List, Iterator
from math import ceil
from enum import Enum
from pathlib import Path

from .models import STraceTx, PTraceTx, PTraceRx, STraceRx, XRTM, CSV, CsvRecord

def pack(i:int, s:STraceTx, mtu=1500, header_size=PTraceTx.HEADER_SIZE) -> Iterable[PTraceTx]:
    seqnum = i
    fragnum = 0
    max_payload_size = mtu - header_size
    bytes_to_pack = s.size
    while bytes_to_pack > max_payload_size:
        yield PTraceTx.from_strace(s, size=bytes_to_pack, number=seqnum, number_in_slice=fragnum, is_last=False)
        bytes_to_pack -= max_payload_size
        seqnum += 1
        fragnum += 1
    yield PTraceTx.from_strace(s, size=bytes_to_pack, number=seqnum, number_in_slice=fragnum, is_last=True)

# TODO: should receive PTraceRx
def unpack(*packets:List[PTraceTx]) -> int:
    """
    asserts a list of packets can be decoded into a slice
    """
    p = packets[0]
    if len(packets) == 1:
        return p.index
    user_id = p.user_id
    view_idx = p.eye_buffer
    index = p.index
    for _, p in enumerate(sorted(packets, key=lambda x: x.fragnum)):
        if p.is_last:
            assert p.fragnum == len(packets)-1
        assert user_id == p.user_id
        assert view_idx == p.view_idx
        assert slice_idx == p.slice_idx


class Packetizer:

    def __init__(self, *args, **kwargs):
        # TODO: read this from config file
        self.constant_delay = kwargs.get('constant_delay', 5)
        self._jitter_min = kwargs.get('jitter_min', 0)
        self._jitter_max = kwargs.get('jitter_max', 5)
        self.user_id = kwargs.get('user_id', 0)
        # print('\n\t> Jitter model - const:', self.constant_delay, '- jitter: ', self._jitter_min, '~', self._jitter_max )

    def jitter(self):
        return self.constant_delay + randint(self._jitter_min, self._jitter_max)

    def process(self, slices:Iterable[STraceTx], seqnum = 0) -> Iterable[PTraceTx]:
        seqnum = 0
        p_per_slice = []
        for s in slices:
            p_in_slice = 0
            for p in pack(seqnum, s):
                seqnum += 1
                p_in_slice += 1
                p.user_id = self.user_id
                p.delay = self.jitter()
                p.index = s.index
                p.render_timing = s.render_timing
                p.type = s.type
                p.eye_buffer = s.eye_buffer
                p.s_trace = None
                yield p
            p_per_slice.append(p_in_slice)
        # print('\t> Fragments per slice - avg:', sum(p_per_slice) / len(p_per_slice), '- min:', min(*p_per_slice), '- max:', max(*p_per_slice) )


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
                packets = sorted(slice_packets, key=lambda x: x.number_in_slice)
                p = packets[-1]
                if p.last_in_slice and p.number_in_slice == len(packets)-1:
                    yield idx, p.delay


class StereoJitterBuffer:

    def __init__(self):
        self.left_buffer = JitterBuffer()
        self.right_buffer = JitterBuffer()

    # TODO: should receive PTraceRx
    def append(self, p:PTraceTx):
        if p.eye_buffer == 1:
            self.left_buffer.append(p)
        elif p.eye_buffer == 2:
            self.right_buffer.append(p)
        else:
            raise ValueError(f'invalid eye buffer {p.eye_buffer}')

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
