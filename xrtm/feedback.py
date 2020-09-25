import random
from enum import IntEnum
from abc import ABC, abstractmethod
from typing import Tuple

from .models import SliceType

class FeedbackType(IntEnum):
    UNKNOWN = 0
    INTRA_REFRESH = 1 # https://tools.ietf.org/html/rfc4585#section-6.3.1
    NACK = 2 # https://tools.ietf.org/html/rfc4585#section-6.3.2
    ACK = 3 # https://tools.ietf.org/html/rfc4585#section-6.3.3

class Feedback(ABC):
    
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


class ReferenceableList(ABC):
    
    @abstractmethod
    def set_referenceable_status(self, frame_idx:int, slice_idx:int, status:bool):
        raise NotImplementedError()


class FeedbackProvider(ABC):

    @abstractmethod
    def handle_feedback(self, fb:Feedback):
        raise NotImplementedError()
    
    @abstractmethod
    def get_full_intra_refresh(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def set_full_intra_refresh(self, idr:bool):
        raise NotImplementedError()

    full_intra_refresh = property(get_full_intra_refresh, set_full_intra_refresh)

    @property
    @abstractmethod
    def rc_max_bits(self) -> int:
        raise NotImplementedError()
    
    @abstractmethod
    def apply_feedback(self, rpl:ReferenceableList) -> ReferenceableList:
        raise NotImplementedError()


class RandomFeedbackProvider(FeedbackProvider):
    
    def __init__(self, full_intra_ratio:float, referenceable_ratio:float, referenceable_default:bool):
        self.full_intra_ratio = int(full_intra_ratio * 100)
        self.referenceable_ratio = int(referenceable_ratio * 100)
        self.referenceable_default = referenceable_default

    def handle_feedback(self, payload:Feedback):
        raise NotImplementedError()
    
    @property
    def rc_max_bits(self) -> int:
        return -1

    def get_full_intra_refresh(self) -> bool:
        return random.randint(0, 100) < self.referenceable_ratio

    def set_full_intra_refresh(self, idr):
        pass
    
    full_intra_refresh = property(get_full_intra_refresh, set_full_intra_refresh)

    def apply_feedback(self, rpl:ReferenceableList) -> ReferenceableList:
        for rp in rpl.pics:
            for s in rp.slices:
                if s.referenceable == self.referenceable_default and random.randint(0, 100) < self.referenceable_ratio:
                    s.referenceable = not self.referenceable_default
        return rpl
