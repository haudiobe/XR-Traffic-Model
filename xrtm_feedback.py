import random
from enum import IntEnum

class FEEDBACK_MODE(IntEnum):
    INTRA_REFRESH = 1
    ACK = 2
    NACK = 3

class Feedback:
    mode:int
    frame_idx:int
    slice_idx:int
    rc_max_bits:int

    def __init__(self, mode:int, frameidx:int=None, sliceidx:int=None):
        self.mode = mode
        self.frame_idx = frameidx
        self.slice_idx = sliceidx


class FeedbackStatus:
    """
    a class to report/retrieve feedback,
    this is a draft / work in progress
    it will need to be designed/used carefully so that client/encoder do not block each other
    """
    _window_size = 3

    def push_status(self, payload:Feedback):
        # a method for incoming feedback
        pass

    def set_encoded(self, sliceidx, frameidx, window_size=3):
        # a method to cleanup when no longer referenced by encoder
        pass

    def get_status(self, sliceidx, frameidx):
        r = random.randint(0,100)
        if r < 25:
            return Feedback(mode=FEEDBACK_MODE.INTRA_REFRESH, frameidx=frameidx, sliceidx=sliceidx) 
        if r < 50:
            nackidx = frameidx - random.randint(1, self._window_size)
            print(nackidx)
            return Feedback(mode=FEEDBACK_MODE.NACK, frameidx=max(0, nackidx), sliceidx=sliceidx)
        return None
