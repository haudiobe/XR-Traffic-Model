import random
from enum import IntEnum
from abc import ABC, abstractmethod
from typing import Tuple, List

class FeedbackType(IntEnum):
    UNKNOWN = 0
    INTRA_REFRESH = 1 # https://tools.ietf.org/html/rfc4585#section-6.3.1
    NACK = 2 # https://tools.ietf.org/html/rfc4585#section-6.3.2
    ACK = 3 # https://tools.ietf.org/html/rfc4585#section-6.3.3

class Feedback(ABC):

    def __lt__(self, fb):
        return self.frame_idx < fb.frame_idx | self.slice_idx < fb.slice_idx
    
    @property
    @abstractmethod
    def type(self) -> FeedbackType:
        raise NotImplementedError()

    @property
    @abstractmethod
    def frame_idx(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def slice_idx(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def rc_max_bits(self) -> int:
        raise NotImplementedError()

class Referenceable(ABC):

    @abstractmethod
    def get_referenceable_status(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def set_referenceable_status(self, idr:bool):
        raise NotImplementedError()

    referenceable = property(get_referenceable_status, set_referenceable_status)


class ReferenceableList(ABC):
    
    @abstractmethod
    def set_referenceable_status(self, frame_idx:int, slice_idx:int, status:bool):
        raise NotImplementedError()


class FeedbackProvider(ABC):

    @abstractmethod
    def handle_feedback(self, fb:Feedback):
        raise NotImplementedError()
    
    @abstractmethod
    def intra_refresh_status(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def clear_intra_refresh_status(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def rc_max_bits(self) -> int:
        raise NotImplementedError()
    
    @abstractmethod
    def apply_feedback(self, rpl:ReferenceableList) -> ReferenceableList:
        raise NotImplementedError()


class XRTMFeedbackProvider(FeedbackProvider):

    def __init__(self, max_diff:int = 16):
        self._fb_buffer:List[Feedback] = []
        self._intra_refresh:Feedback = None
        self._rc_max_bits:Feedback = None
        self._max_diff = max_diff

    @abstractmethod
    def handle_feedback(self, fb:Feedback):        
        if (self._fb_buffer != None) and (fb.frame_idx < self._fb_buffer[len(self._fb_buffer)-1]):
            return

        if fb.rc_max_bits != -1:
            self._rc_max_bits = fb

        if fb.type == FeedbackType.INTRA_REFRESH:
            if self._intra_refresh == None:
                self._intra_refresh = fb
            elif self._intra_refresh.slice_idx < fb.frame_idx:
                self._intra_refresh = fb

        elif fb.type == FeedbackType.NACK or fb.type == FeedbackType.ACK:
            self._fb_buffer.append(fb)
            if len(self._fb_buffer) > 1:
                self._fb_buffer.sort()
                head = self._fb_buffer[len(self._fb_buffer)-1].slice_idx
                self._fb_buffer = [item for item in self._fb_buffer if item.slice_idx > (head - self._max_diff)]

    def intra_refresh_status(self) -> bool:
        return self._intra_refresh != None

    def clear_intra_refresh_status(self):
        self._intra_refresh = None

    @property
    def rc_max_bits(self) -> int:
        if self._rc_max_bits == None:
            return -1
        return self._rc_max_bits.rc_max_bits
        
    def apply_feedback(self, rpl:ReferenceableList) -> ReferenceableList:
        if len(self._fb_buffer) == 0:
            return 

        for fb in self._fb_buffer:
            if fb.type == FeedbackType.NACK:
                rpl.set_referenceable_status(fb.frame_idx, fb.slice_idx, False)
            elif fb.type == FeedbackType.ACK:
                rpl.set_referenceable_status(fb.frame_idx, fb.slice_idx, True)

        self._fb_buffer = []
        return rpl