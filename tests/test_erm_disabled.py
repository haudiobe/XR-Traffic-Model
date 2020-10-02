from tests.fixtures import gen_vtrace_pattern
from xrtm.models import SliceType
import unittest

from xrtm.encoder import Encoder, EncoderConfig, ErrorResilienceMode

class RefListErmDisabled(unittest.TestCase):

    def test_follow_input_pattern(self):
        cfg = EncoderConfig()
        cfg.error_resilience_mode = ErrorResilienceMode.DISABLED
        m = Encoder(cfg)
        
        for v in gen_vtrace_pattern("IPPPPPIPPPPPIPPPPP"):
            for s in m.encode(v):
                self.assertEqual(v.is_full_intra, s.slice_type == SliceType.IDR)
                self.assertEqual(not v.is_full_intra, s.slice_type == SliceType.P)