from .models import Frame, Slice
from enum import Enum
from logging import Logger
from typing import List
from random import random
from functools import reduce

logger = Logger("xrtm decoder")

class CTU_type(Enum):
    INTRA           = 0
    INTER           = 1
    MERGE           = 2
    SKIP            = 3

class CTU_status(Enum):
    OK              = 0
    DAMAGED         = 1
    UNAVAILABLE     = 2

class CtuMap:
    def __init__(self, rows, cols):
        self._width = cols
        self._height = rows
        self.ctus = [None] * rows * cols

    def get(self, x, y):
        return self.ctus[y + x * self._width]

    def position(self, idx):
        return 

class HrdCTU:
    def __init__(self, ctu_type:int, rpl:List[int]):
        self.ctu_type = ctu_type
        self.refs = rpl # a list of pictures referenced within this CTU

class HrdSlice:
    def __init__(self, ctu_map:List[HrdCTU]):
        self.ctu_map = ctu_map

class HrdFrame:
    def __init__(self, poc:int, slices:List[HrdSlice]):
        self.poc = poc
        self.slices = slices

    @property
    def ctu_map(self):
        return [ c for c in s.ctu_map for s in self.slices]

####################################################################

class DecoderCfg:
    CTU_SIZE = 64
    def __init__(self, *args, **kwargs):
        self.frame_width = kwargs.get("frame_width", 1024)
        self.frame_heigh = kwargs.get("frame_height", 1024)

def validate_decoder_config(cfg):
    pass

def timestep(step, current_time, fps) -> List[Frame]:
    pass


def get_reference_candidates(ctu_idx:int, grid:List[HrdCTU], width:int, height:int):
    """
    ctu_idx: index of the ctu. ctu_idx = ctu_pos.y + (ctu_pos.x * grid.width)
    width: number of colums for the grid
    height: number of rows for the grid
    returns a 3x3 subset of the CTU grid centered around ctu_idx, None for out of bounds values :
    [
        topleft,        top,        topright,
        left,           ctu_idx,    right,
        bottomleft,     bottom,     bottomright
    ]
    """
    assert (ctu_idx >= 0) and (ctu_idx < len(grid)), 'invalid ctu index'
    assert len(grid) == (width * height), 'invalid map parameters'
    def sample(x, y):
        if x < 0 or x >= width or y < 0 or y >= height:
            return None
        else:
            return grid[ y + x * width ]
    posx, posy = ctu_idx % width, ctu_idx // width
    return [sample(x, y) for x in range(posx-1, posx+2) for y in range(posy-1, posy+2)]


def model_ctu_references(decoding:HrdCTU, reference:HrdFrame) -> Iterable[HrdCTU]:
    # get a 3x3 grid of referenceable CTUs centered around the one currently decoding
    rc = get_reference_candidates(decoding.idx, reference.ctu_map, reference.width, reference.height)
    # cast a random vector
    v = [ 2*random()-1 for _ in range(2) ]
    # deal with edges/corners by inverting the vector where needed
    if ((rc[1] is None) and (v[0] < 0)) or ((rc[7] is None) and (v[0] > 0)):
            v[0] *= -1
    if ((rc[3] is None) and (v[1] < 0)) or ((rc[5] is None) and (v[1] > 0)):
            v[1] *= -1
    # rasterize the vector into 3x3 grid of boolean value to mask out non
    top = v[0] < 0
    bottom = v[0] > 0
    left = v[1] < 0
    right = v[1] > 0
    for i, is_ref in  enumerate([top and left, top, top and right, left, True, right, bottom and left, bottom, bottom and right]):
        if is_ref:
            yield rc[i]


def can_be_reconstructed(ctu:HrdCTU, dpb:List[HrdFrame]):
    for frame in dpb:
        if ctu.ref == frame.poc:
            status = [referenced.status == CTU_status.OK for referenced in model_ctu_references(ctu, frame)]
            if len(status) > 0:
                return False
            else:
                return reduce(lambda r, item: (r and item), status)
    return False


class Decoder:

    def __init__(self, cfg:DecoderCfg):
        """
        Create a map of slices, CTU maps (64 x 64) and reference frames
            • Example: 2048 x 2048, 8 slices, 3 reference frames
            • Addresses for 2048 / (8 * 64) = 4 rows with 32 CTUs for 8 slices in 3 frames
            maintained.
            • For each CTU of each frame, store mode:
                • Correct
                • Damaged
                • Unavailable
                • Initialize all CTUs as unavailable
        """
        validate_decoder_config(cfg)

        # if cfg.error_resilience_mode >= ErrorResilienceMode.FEEDBACK_BASED and feedback_dispatcher == None:
        #     raise DecoderConfigException('Feedback based error resilience requires a feedback provider')

        self._cfg = cfg
        # self._view_idx = view_idx # LEFT/RIGHT view
        # self._feedback = feedback_provider
        # self._default_referenceable_status = cfg.error_resilience_mode != ErrorResilienceMode.FEEDBACK_BASED_ACK
        
        w = cfg.frame_width
        h = cfg.frame_height
        self._ctus_per_frame = int(( w * h ) / ( DecoderCfg.CTU_SIZE * DecoderCfg.CTU_SIZE ))
        self._ctus_per_slice = int(( h / DecoderCfg.CTU_SIZE ) * h / ( cfg.slices_per_frame * DecoderCfg.CTU_SIZE ))

        logger.info(f'frame WxH: {w} x {h}')
        logger.info(f'slice height: {int(h / cfg.slices_per_frame)}')
        logger.info(f'ctus p. frame: {self._ctus_per_frame}')
        logger.info(f'ctus p. slice: {self._ctus_per_slice}')

    @property
    def decoding_delay(self):
        pass
    
    # Get all slices from trace for the frame (timestep), some slices may be late/lost. assuming late frames may be passed-in for decode
    def decode(self, received:Frame):
        
        reconstructed = HrdFrame()
        
        for slice_idx in range(self._cfg.slices_per_frame):
            
            # if a slice was not received, or if it was received too late
            if (received.slices[slice_idx] == None) or (received.slices[slice_idx].timestamp < self.decoding_delay):
                reconstructed.slices[slice_idx].set_unvailable() # Mark all CTUs as unavailable
                self.feedback.report_lost_slice(frame_idx, slice_idx) # Indicate the slice loss for feedback

            # if a slice was received
            else:
                # indicate the slice received for “feedback”
                self.feedback.report_received_slice(frame_idx, slice_idx)

                # update CTU map with received data
                for ctu_idx, c in enumerate(received.slices[slice_idx].ctu_map):
                    if c.type == CTU_type.INTRA:
                        # If it is an intra CTU, mark it correct
                        reconstructed.slices[slice_idx].ctu_map[i].status = CTU_status.OK                    
                    else:
                        assert c.type in [CTU_type.INTER, CTU_type.MERGE, CTU_type.SKIP], 'invalid CTU type'
                        if can_be_reconstructed(c, self.dpb):
                            reconstructed.slices[slice_idx].ctu_map[i].status = CTU_status.OK
                        else:
                            reconstructed.slices[slice_idx].ctu_map[i].status = CTU_status.DAMAGED

        # Compute the totally unavailable and damaged CTUs in this frame
        ctu_map = reconstructed.ctu_map
        invalid_ctu_ratio = len([ctu for ctu in ctu_map if ctu.status == CTU_status.OK]), len(ctu_map) 
        reconstructed.PSNR = received.PSNR * invalid_ctu_ratio

        # add reconstructed to dpb
        self.dpb.append(reconstructed)
        return reconstructed