import random
from enum import IntEnum
from abc import ABC, abstractmethod
from typing import Tuple

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

