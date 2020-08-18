import random
from enum import IntEnum

class FEEDBACK_MODE(IntEnum):
    INTRA_REFRESH = 1
    ACK = 2
    NACK = 3

class Feedback:
    rc_max_bits:int
    frame_idx:int
    slice_idx:int

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

    def is_intra_refresh(self, frame_idx, slice_idx):
        # a method to retrieve the feedback
        intra_refresh = random.randint(0,10) < 3
        return intra_refresh

    def nack_index(self, frameidx, sliceidx):
        nack = random.randint(1,10)
        if nack < self._window_size:
            return max(0, frameidx - nack)