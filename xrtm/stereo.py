from typing import Iterator

from .feedback import (
    FeedbackProvider,
    TxFeedbackHandler,
    Feedback,
    RandomFeedbackGenerator
)

from .encoder import (
    EncoderConfig,
    Encoder
)

from .models import (
    VTrace,
    Slice
)


class RandomStereoFeedback:
    def __init__(self, full_intra_ratio:float, referenceable_ratio:float, referenceable_default:bool):
        self.enc0 = RandomFeedbackGenerator(full_intra_ratio, referenceable_ratio, referenceable_default)
        self.enc1 = RandomFeedbackGenerator(full_intra_ratio, referenceable_ratio, referenceable_default)


class StereoFeedbackHandler:
    def __init__(self, max_diff:int = 16):
        self.enc0 = TxFeedbackHandler(max_diff=max_diff)
        self.enc1 = TxFeedbackHandler(max_diff=max_diff)
        
    def handle_feedback(self, fb:Feedback):
        # handle specific feedback payload for left/right eye here as needed
        self.enc0.handle_feedback(fb)
        self.enc1.handle_feedback(fb)


class StereoEncoder:

    def __init__(self, cfg:EncoderConfig, feedback:StereoFeedbackHandler=None):
        if feedback != None:
            self._enc0 = Encoder(cfg, feedback.enc0, view_idx=0)
            self._enc1 = Encoder(cfg, feedback.enc1, view_idx=1)
        else:
            self._enc0 = Encoder(cfg, None, view_idx=0)
            self._enc1 = Encoder(cfg, None, view_idx=1)

    def encode(self, vtrace:VTrace) -> Iterator[Slice]:
        s0 = self._enc0.encode(vtrace)
        s1 = self._enc1.encode(vtrace)
        while True:
            try:
                yield next(s0)
                yield next(s1)
            except StopIteration:
                break
