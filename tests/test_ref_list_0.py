from tests.fixtures import gen_vtrace, gen_vtrace_pattern, v_trace
import unittest

from xrtm.encoder import RefPicList, Encoder, EncoderConfig, ErrorResilienceMode
from xrtm.models import Frame, SliceType, Slice

def _slice(i, referenceable, slice_type:SliceType):
    s = Slice(i, i, 0, 0, 0, referenceable=referenceable)
    s.slice_type = slice_type
    f = Frame(poc=i)
    f.slices.append(s)
    return f

class TestRefPicList(unittest.TestCase):

    def test_idr_frame_resets_list(self):
        rpl = RefPicList(max_size=33)
        f = Frame(poc=0, idr=True) # IDR
        rpl.add_frame(f)
        self.assertEqual(len(rpl.pics), 1)
        f = Frame(poc=1, idr=False)
        rpl.add_frame(f)
        self.assertEqual(len(rpl.pics), 2)
        f = Frame(poc=2, idr=False)
        rpl.add_frame(f)
        self.assertEqual(len(rpl.pics), 3)
        f = Frame(poc=3, idr=True) # IDR
        rpl.add_frame(f)
        self.assertEqual(len(rpl.pics), 1)
        f = Frame(poc=4, idr=True) # IDR
        rpl.add_frame(f)
        self.assertEqual(len(rpl.pics), 1)
        f = Frame(poc=5, idr=False)
        rpl.add_frame(f)
        self.assertEqual(len(rpl.pics), 2)
        f = Frame(poc=6, idr=False)
        rpl.add_frame(f)
        self.assertEqual(len(rpl.pics), 3)
        f = Frame(poc=7, idr=True) # IDR
        rpl.add_frame(f)
        self.assertEqual(len(rpl.pics), 1)


    def test_max_frame_buffer_size(self):
        max_size = 16
        rpl = RefPicList(max_size=max_size)
        i = 0

        while i < max_size:
            f = Frame(poc=i, idr=False)
            rpl.add_frame(f)
            self.assertEqual(len(rpl.pics), i+1)
            self.assertEqual(rpl.pics[i], f)
            i += 1

        while i < (max_size + 3):
            f = Frame(poc=i, idr=False)
            rpl.add_frame(f)
            self.assertEqual(len(rpl.pics), max_size)
            self.assertEqual(rpl.pics[max_size-1], f)
            i += 1

        f = Frame(poc=i, idr=True) # IDR
        rpl.add_frame(f)
        self.assertEqual(len(rpl.pics), 1)


    def test_refs_iterates_reversed(self):
        rpl = RefPicList(max_size=16)
        i = 25
        
        while i < 50:
            f = Frame(poc=i, idr= i==0)
            f.slices.append(Slice(i, i, 0, 0, 0, referenceable=True))
            rpl.add_frame(f)
            i += 1

        for f in rpl.slice_refs(slice_idx=0):
            i -= 1
            self.assertEqual(f, i)
            return
    
    def test_refs_skip_non_referenceable(self):

        referenceable = []
        rpl = RefPicList(max_size=16)

        i = 0
        rpl.add_frame(_slice(i, True, SliceType.P))
        referenceable.append(i)

        i += 1
        rpl.add_frame(_slice(i, True, None))
        referenceable.append(i)
        
        i += 1
        rpl.add_frame(_slice(i, False, None))
        
        i += 1
        rpl.add_frame(_slice(i, False, SliceType.P))

        i += 1
        rpl.add_frame(_slice(i, True, SliceType.P))
        referenceable.append(i)

        i += 1
        rpl.add_frame(_slice(i, True, SliceType.P))
        referenceable.append(i)

        i += 1
        rpl.add_frame(_slice(i, False, SliceType.P))

        i += 1
        rpl.add_frame(_slice(i, False, SliceType.P))

        i += 1
        rpl.add_frame(_slice(i, False, SliceType.P))

        referenceable.reverse()
        slice_refs = [f for f in rpl.slice_refs(slice_idx=0)]
        self.assertEqual(len(referenceable), len(slice_refs))
        z = zip(referenceable, slice_refs)
        for t in z:
            self.assertEqual(t[0], t[1])


    def test_refs_stops_on_idr(self):

        referenceable = []
        rpl = RefPicList(max_size=16)

        i = 0
        rpl.add_frame(_slice(i, True, SliceType.P))
        i += 1
        rpl.add_frame(_slice(i, True, SliceType.P))
        i += 1
        rpl.add_frame(_slice(i, True, SliceType.P))
        i += 1
        rpl.add_frame(_slice(i, True, SliceType.P))

        # frames before IDR may not be referenced
        i += 1
        rpl.add_frame(_slice(i, True, SliceType.IDR))
        referenceable.append(i)
        i += 1
        rpl.add_frame(_slice(i, False, SliceType.P))
        i += 1
        rpl.add_frame(_slice(i, False, SliceType.P))
        i += 1
        rpl.add_frame(_slice(i, True, SliceType.P))
        referenceable.append(i)

        referenceable.reverse()
        slice_refs = [f for f in rpl.slice_refs(slice_idx=0)]
        self.assertEqual(len(referenceable), len(slice_refs))
        
        z = zip(referenceable, slice_refs)
        for t in z:
            self.assertEqual(t[0], t[1])

        # even if IDR is not referenceable
        i += 1
        rpl.add_frame(_slice(i, False, SliceType.IDR))
        slice_refs = [f for f in rpl.slice_refs(slice_idx=0)]
        self.assertEqual(0, len(slice_refs))
    

    def test_set_referenceable(self):

        referenceable = []
        rpl = RefPicList(max_size=32)

        for i in range(32):
            rpl.add_frame(_slice(i, False, None))
            referenceable.append(i)

        slice_refs = [f for f in rpl.slice_refs(slice_idx=0)]
        self.assertEqual(0, len(slice_refs))
        
        rpl.set_referenceable_status(25, 0, True)
        slice_refs = [f for f in rpl.slice_refs(slice_idx=0)]
        self.assertEqual(1, len(slice_refs))
        self.assertEqual(25, slice_refs[0])


    def test_set_not_referenceable(self):

        referenceable = []
        rpl = RefPicList(max_size=32)

        for i in range(32):
            rpl.add_frame(_slice(i, True, None))
            referenceable.append(i)
                
        rpl.set_referenceable_status(25, 0, False)
        for f in rpl.slice_refs(slice_idx=0):
            self.assertNotEqual(f, 25)
    
    