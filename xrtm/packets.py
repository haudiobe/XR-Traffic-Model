# from xrtm_encoder import STrace
from abc import ABC

import time
import random
import struct
import math

import pickle
import zlib

from enum import Enum
from typing import Iterator, List

from .models import SliceType, Slice


def timestamp():
    # @TODO: 90Khz clock + random seed
    return int(time.monotonic() * 1000)


class MediaPacketType(Enum):
    SINGLE_UNIT = 1
    FRAGMENTED_UNIT = 2


class MediaPacketHeader(object):
    
    HEADER_SIZE = 10
    
    def __init__(self, seqnum:int, timestamp:int, packet_type:MediaPacketType, slice_type:SliceType, first:bool=False, last:bool=False):
        # TODO: use actual RTP timestamps
        self._seqnum = seqnum
        self._timestamp = timestamp
        # codec specific
        self._packet_type = packet_type
        self._slice_type = slice_type
        self._first = first
        self._last = last
        
    @property
    def size(self):
        return MediaPacketHeader.HEADER_SIZE

    @property
    def seqnum(self):
        return self._seqnum

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def packet_type(self):
        return self._packet_type

    @property
    def first(self):
        return self._first

    @property
    def last(self):
        return self._last

    @property
    def slice_type(self):
        return self._slice_type
    
    @staticmethod
    def from_slice(seqnum:int, s:Slice, packet_type:MediaPacketType, first:bool=False, last:bool=False):
        return MediaPacketHeader(
            seqnum=seqnum,
            timestamp=s.pts,
            packet_type=packet_type,
            slice_type=s.slice_type,
            first=first,
            last=last
        )

    @staticmethod
    def encode(header)->bytes:
        data = struct.pack('!H', header.seqnum) # 2 bytes
        data += struct.pack('!I', header.timestamp) # 4 bytes
        data += struct.pack('!B', header.packet_type.value) # 1 byte
        data += struct.pack('!B', header.slice_type.value) # )1 byte
        data += struct.pack('!?', header.first) # 1 byte
        data += struct.pack('!?', header.last) # 1 byte
        return data

    @staticmethod
    def decode(data:bytes):
        seqnum = struct.unpack('!H', data[:2])
        timestamp = struct.unpack('!I', data[2:6])
        packet_type = struct.unpack('!B', data[6:7])
        slice_type = struct.unpack('!B', data[7:8])
        first = struct.unpack('!?', data[8:9])
        last = struct.unpack('!?', data[9:10])
        return MediaPacketHeader(
            seqnum=seqnum[0],
            timestamp=timestamp[0],
            packet_type=MediaPacketType(packet_type[0]),
            slice_type=SliceType(slice_type[0]),
            first=first[0],
            last=last[0],
        )

class MediaPacket:

    def __init__(self, header:MediaPacketHeader, payload:bytes):
        self._header = header
        self.payload = payload

    def __lt__(self, mp):
        return self._header.seqnum < mp.header.seqnum

    @property
    def header(self):
        return self._header

    @property
    def packet_type(self):
        return self._header.packet_type

    @property
    def first(self):
        return self._header.first

    @property
    def last(self):
        return self._header.last

    @property
    def seqnum(self):
        return self._header.seqnum

    @property
    def payload_size(self):
        return len(self.payload)

    @property
    def size(self) -> int:
        return self._header.size + len(self.payload)

    @staticmethod
    def encode(mp) -> bytes:
        return MediaPacketHeader.encode(mp.header) + mp.payload

    @staticmethod
    def decode(data:bytes):
        header_bytes = data[:len(MediaPacketHeader.HEADER_SIZE)]
        header = MediaPacketHeader.decode(header_bytes)
        payload = data[len(MediaPacketHeader.HEADER_SIZE):]
        return MediaPacket(header, payload)


def pack_slice(seqnum:int, s:Slice, mtu:int) -> Iterator[MediaPacket]:

    if mtu <= 0:
        raise ValueError('Invalid MTU')

    bytes_to_pack = Slice.encode(s)
    max_payload_size = mtu - MediaPacketHeader.HEADER_SIZE

    if len(bytes_to_pack) <= max_payload_size:
            h = MediaPacketHeader.from_slice(seqnum, s, MediaPacketType.SINGLE_UNIT)
            p = MediaPacket(h, bytes_to_pack)
            yield p
            return

    mpt = MediaPacketType.FRAGMENTED_UNIT
    is_first = True
    while len(bytes_to_pack) > 0:
        chunksize = min(max_payload_size, len(bytes_to_pack))
        payload = bytes_to_pack[:chunksize]
        bytes_to_pack = bytes_to_pack[chunksize:]
        is_last = len(bytes_to_pack) == 0
        h = MediaPacketHeader.from_slice(seqnum, s, packet_type=mpt, first=is_first, last=is_last)
        p = MediaPacket(h, payload)
        seqnum += 1
        is_first = False
        yield p


def defrag(packets:List[MediaPacket]) -> Slice:
    """
    asserts t
    """
    try:
        seqnum = -1
        data = bytes()
        for i, p in enumerate(packets):
            assert p.packet_type == MediaPacketType.FRAGMENTED_UNIT
            if i == 0:
                assert p.first
                assert p.first != p.last
                seqnum = p.seqnum
            elif i == (len(packets)-1):
                assert p.last
            else:
                assert not p.first
                assert not p.last
            assert p.seqnum == (seqnum + i)
            data += p.payload
        return Slice.decode(data)
    except AssertionError:
        raise ValueError('Invalid media packet sequence')

def unpack_slice(*packets:Iterator[MediaPacket]) -> Slice:
    if len(packets) == 1:
        p = packets[0]
        if p.packet_type != MediaPacketType.SINGLE_UNIT:
            raise ValueError('Invalid packet type')
        return Slice.decode(p.payload)
    else:
        return defrag(packets)
