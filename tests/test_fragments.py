import unittest
import random
from xrtm.models import Slice, SliceType
from xrtm.packets import pack_slice, unpack_slice, MediaPacketHeader, MediaPacketType

def _slice(byte_size, pts=0):
    s = Slice(poc=0, pts=pts)
    s.slice_type = SliceType.IDR
    s.bits_new = byte_size * 8
    return s

header_types = [
    (MediaPacketType.SINGLE_UNIT, False, False),
    (MediaPacketType.FRAGMENTED_UNIT, True, False),
    (MediaPacketType.FRAGMENTED_UNIT, False, False),
    (MediaPacketType.FRAGMENTED_UNIT, False, True),
]

class TestMediaPacketHeader(unittest.TestCase):

    def test_header_from_slice(self):
        for (t, f, l) in header_types:
            with self.subTest():
                s = _slice(12345)
                s.poc = random.randint(3, 12345)
                s.pts = random.randint(12345, 67890)
                s.slice_idx = 8
                s.slice_type = SliceType.P
                seqnum = random.randint(0, 12345)

                hdr = MediaPacketHeader.from_slice(seqnum, s, t, first=f, last=l)
                self.assertEqual(t, hdr.packet_type)
                self.assertEqual(f, hdr.first)
                self.assertEqual(l, hdr.last)
                self.assertEqual(hdr.seqnum, seqnum)
                self.assertEqual(hdr.slice_type, s.slice_type)
                

    # @unittest.skip("failling")
    def test_encode_decode(self):
        for (t, f, l) in header_types:
            with self.subTest():
                s = _slice(12345)
                s.poc = random.randint(3, 12345)
                s.pts = random.randint(12345, 67890)
                s.slice_idx = 8
                s.slice_type = SliceType.P
                seqnum = random.randint(0, 12345)

                enc = MediaPacketHeader.from_slice(seqnum, s, t, first=f, last=l)
                hdr_bytes = MediaPacketHeader.encode(enc)
                dec = MediaPacketHeader.decode(hdr_bytes)
                self.assertEqual(t, dec.packet_type)
                self.assertEqual(f, dec.first)
                self.assertEqual(l, dec.last)
                self.assertEqual(enc.seqnum, dec.seqnum)
                self.assertEqual(enc.timestamp, dec.timestamp)
                self.assertEqual(enc.slice_type, dec.slice_type)

    
    def test_header_size(self):
        for (t, f, l) in header_types:
            with self.subTest():
                s = _slice(12345)
                s.poc = random.randint(3, 12345)
                s.pts = random.randint(12345, 67890)
                s.slice_idx = 8
                s.slice_type = SliceType.P
                seqnum = random.randint(0, 12345)

                hdr = MediaPacketHeader.from_slice(seqnum, s, t, first=f, last=l)
                hdr_bytes = MediaPacketHeader.encode(hdr)
                self.assertEqual(MediaPacketHeader.HEADER_SIZE, len(hdr_bytes))
                self.assertEqual(MediaPacketHeader.HEADER_SIZE, hdr.size)

class TestPackSlice(unittest.TestCase):

    def test_invalid_mtu(self):
        with self.assertRaises(ValueError):
            max_size = 0
            s = _slice(1500)
            _ = [*pack_slice(12345, s, max_size)]

    def test_single_unit(self):
        max_size = 1500
        seqnum = random.randint(12345, 67890)
        s = _slice(max_size - MediaPacketHeader.HEADER_SIZE)
        packets = [*pack_slice(seqnum, s, max_size)]
        self.assertEqual(len(packets), 1)

        p = packets[0]
        self.assertEqual(p.packet_type, MediaPacketType.SINGLE_UNIT)
        self.assertEqual(p.seqnum, seqnum)

    def test_fragmented_units_x2(self):
        max_size = 1500
        seqnum = random.randint(12345, 67890)
        s = _slice(max_size - MediaPacketHeader.HEADER_SIZE + 1)
        packets = [*pack_slice(seqnum, s, max_size)]
        self.assertEqual(len(packets), 2)
        
        p0 = packets[0]
        self.assertEqual(p0.packet_type, MediaPacketType.FRAGMENTED_UNIT)
        self.assertEqual(p0.first, True)
        self.assertEqual(p0.last, False)
        self.assertEqual(p0.seqnum, seqnum)
        self.assertEqual(p0.size, max_size)
        self.assertEqual(p0.payload_size, max_size - MediaPacketHeader.HEADER_SIZE)
        
        p1 = packets[1]
        self.assertEqual(p1.packet_type, MediaPacketType.FRAGMENTED_UNIT)
        self.assertEqual(p1.first, False)
        self.assertEqual(p1.last, True)
        self.assertEqual(p1.seqnum, seqnum + 1)
        self.assertEqual(p1.size, MediaPacketHeader.HEADER_SIZE + 1)
        self.assertEqual(p1.payload_size, 1)


    def test_fragmented_units_many(self):
        
        s = _slice(random.randint(12345, 98765))

        max_size = 1500
        max_payload_size = max_size - MediaPacketHeader.HEADER_SIZE
        nfrags = ((s.bits/8) // max_payload_size) + 1
        last_frag_size = (s.bits/8) - ((nfrags - 1) * max_payload_size)

        seqnum = random.randint(12345, 67890)
        packets = [*pack_slice(seqnum, s, max_size)]
        self.assertEqual(len(packets), nfrags)
        
        while len(packets) > 0:
            is_first = len(packets) == nfrags
            is_last = len(packets) == 1
            p = packets.pop(0)
            self.assertEqual(p.packet_type, MediaPacketType.FRAGMENTED_UNIT)
            self.assertEqual(p.first, is_first)
            self.assertEqual(p.last, is_last)
            self.assertEqual(p.seqnum, seqnum)
            if is_last:
                self.assertEqual(p.size, last_frag_size + MediaPacketHeader.HEADER_SIZE)
                self.assertEqual(p.payload_size, last_frag_size)
            else:
                self.assertEqual(p.size, max_size)
                self.assertEqual(p.payload_size, max_size - MediaPacketHeader.HEADER_SIZE)
            seqnum += 1


def _packets(n=1, mtu=1500):
    max_payload_size = mtu - MediaPacketHeader.HEADER_SIZE
    seqnum = random.randint(0, 12345)
    pts = random.randint(12345, 67890)
    if n == 1:
        s = _slice(max_payload_size, pts=pts)
        return [*pack_slice(seqnum, s, mtu)], s
    else:
        bytesize = max_payload_size * (n-1)
        bytesize += random.randint(1, max_payload_size)
        s = _slice(bytesize, pts=pts)
        return [*pack_slice(seqnum, s, mtu)], s



class TestUnpackSlice(unittest.TestCase):

    def test_single_unit(self):
        p, stx = _packets(1)
        srx = unpack_slice(*p)
        self.assertEqual(stx.pts, srx.pts)

    def test_fragmented_units(self):
        p, stx = _packets(10)
        srx = unpack_slice(*p)
        for k in Slice.keys:
            self.assertEqual(getattr(stx, k), getattr(srx, k))

    def test_fragmented_unordered(self):
        p, stx = _packets(10)

        with self.assertRaises(ValueError):
            p.reverse()
            srx = unpack_slice(*p)
            self.assertEqual(stx.pts, srx.pts)

        with self.assertRaises(ValueError):
            p.append(p.pop(0))
            srx = unpack_slice(*p)
            self.assertEqual(stx.pts, srx.pts)

    def test_fragmented_missing(self):
        pass

    def test_fragmented_duplicated(self):
        pass

    def test_fragmented_invalid(self):
        pass



