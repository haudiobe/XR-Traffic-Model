import unittest
import math

from xrtm.models import VTrace, SliceType, Slice

class TestVTrace(unittest.TestCase):

    def test_slice_type(self):
        v = VTrace()

        with self.assertRaises(AttributeError):
            v.is_full_intra

        v.ctu_intra_pct = 1.0
        v.ctu_inter_pct = 0
        v.ctu_merge_pct = 0
        v.ctu_skip_pct = 0
        self.assertTrue(v.is_full_intra)

        v.ctu_inter_pct = 1.0
        self.assertTrue(v.is_full_intra)

        v.ctu_merge_pct = 1.0
        self.assertTrue(v.is_full_intra)

        v.ctu_skip_pct = 1.0
        self.assertTrue(v.is_full_intra)

        v.ctu_intra_pct = 0
        self.assertFalse(v.is_full_intra)


class TestSlice(unittest.TestCase):

    def test_encode_decode(self):
        stx = Slice(
            12345, 
            67890, 
            slice_idx=8,
            inter_mean=1.2345,
            intra_mean=6.7890,
            view_idx=9,
            slice_type=SliceType.IDR
        )
        stx.bits_new = 987665
        data = Slice.encode(stx)
        self.assertEqual(len(data), math.ceil(stx.bits / 8))
        srx = Slice.decode(data)
        for k in Slice.keys:
            self.assertEqual(getattr(stx, k), getattr(srx, k))