import random
import pickle
from enum import IntEnum
from abc import ABC, abstractmethod
from typing import Tuple, List

class FeedbackType(IntEnum):
    UNKNOWN = 0
    INTRA_REFRESH = 1
    SLICE_NACK = 2
    SLICE_ACK = 3

class AbstractFeedback(ABC):

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
    def max_bits(self) -> int:
        raise NotImplementedError()


class Feedback:

    def __init__(self, fb_type:FeedbackType, frame_idx:int, slice_idx:int, max_bits:int=-1):
        self._frame_idx = frame_idx
        self._slice_idx = slice_idx
        self._type = fb_type
        self._max_bits = max_bits

    def __lt__(self, fb:AbstractFeedback):
        return self.frame_idx < fb.frame_idx | self.slice_idx < fb.slice_idx
    
    @property
    def type(self) -> FeedbackType:
        self._type

    @property
    def frame_idx(self) -> int:
        self._frame_idx

    @property
    def slice_idx(self) -> int:
        self._slice_idx

    @property
    def max_bits(self) -> int:
        self._max_bits

    @staticmethod
    def encode(fb:AbstractFeedback) -> bytes:
        return pickle.dumps(fb)

    @staticmethod
    def decode(data:bytes) -> AbstractFeedback:
        return pickle.loads(data)

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
