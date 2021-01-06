from random import randint
from typing import Iterable, List
from math import ceil
from enum import Enum

from .models import STraceTx, PTraceTx, PTraceRx, STraceRx, XRTM, CSV, CsvRecord

def pack(i:int, s:STraceTx, mtu=1500, header_size=PTraceTx.HEADER_SIZE) -> Iterable[PTraceTx]:
    seqnum = i
    fragnum = 0
    max_payload_size = mtu - header_size
    bytes_to_pack = s.size
    while bytes_to_pack > max_payload_size:
        yield PTraceTx(s, max_payload_size, seqnum, fragnum, is_last=False)
        bytes_to_pack -= max_payload_size
        seqnum += 1
        fragnum += 1
    yield PTraceTx(s, bytes_to_pack, seqnum, fragnum, is_last=True)

# TODO: should receive PTraceRx
def unpack(*packets:List[PTraceTx]) -> STraceTx:
    p = packets[0]
    if len(packets) == 1:
        return p.slice
    user_id = p.user_id
    view_idx = p.view_idx
    slice_idx = p.slice_idx
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
        if p.slice_idx not in self._buffer:
            self._buffer[p.slice_idx] = [p]
        else:
            self._buffer[p.slice_idx].append(p)

    def delete(self, slice_idx):
        self._buffer.pop(slice_idx)

    def iter_late_slices(self, timestamp:int):
        for slice_idx, packets in self._buffer.items():
            for p in packets:
                if p.packet_availability <= timestamp:
                    yield slice_idx
                    continue
    
    def iter_complete_slices(self) -> List[PTraceTx]:
        for _, packets in self._buffer.items():
            last = len(packets)-1
            for i, p in enumerate(sorted(packets, key=lambda x: x.number_in_slice)):
                if p != i:
                    continue
                if i == last and p.last_in_slice:
                    yield packets


class StereoJitterBuffer:

    def __init__(self):
        self.left_buffer = JitterBuffer()
        self.right_buffer = JitterBuffer()

    # TODO: should receive PTraceRx
    def append(self, p:PTraceTx):
        if p.slice.is_left:
            self.left_buffer.append(p)
        elif p.slice.is_right:
            self.right_buffer.append(p)
        else:
            raise ValueError('invalid packet')

    def delete(self, slice_idx):
        self.left_buffer.delete(slice_idx)
        self.right_buffer.delete(slice_idx)
    
    def iter_late_slices(self, timestamp:int):
        for s in self.left_buffer.iter_late_slices(timestamp):
            yield s
        for s in self.right_buffer.iter_late_slices(timestamp):
            yield s

    def iter_complete_slices(self) -> List[PTraceTx]:
        for s in self.left_buffer.iter_complete_slices():
            yield s
        for s in self.right_buffer.iter_complete_slices():
            yield s


class DePacketizer:

    def __init__(self, cfg):
        self.user_id:int = None
        self.time = 0
        self.delay_budget = 0
        self.buffer = StereoJitterBuffer()
    
    def availability_deadline(self):
        return self.time + self.delay_budget

    # TODO: should receive PTraceRx
    def process(self, timestep:int, packets:List[PTraceTx]):
        """
        decode packets received over the given timestep
        """
        self.time += timestep
        # append incoming packets to the jitterbuffer
        for p in packets:
            assert p.user_id == self.user_id
            if p.fragnum == 0 and p.is_last: # no fragmentation
                yield unpack(p)
            else: # fragmented
                self.buffer.append(p)

        # drop late packets / slices
        for slice_idx in self.buffer.iter_late_slices(self.availability_deadline):
            self.buffer.delete(slice_idx)

        # yield slices for which we have all packets
        for packets in self.buffer.iter_complete_slices():
            yield unpack(packets)

