import unittest
from xrtm.models import VTrace, SliceType

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
