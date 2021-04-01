from xrtm.models import EncoderConfig, RC_mode, ErrorResilienceMode
from xrtm_packetizer import PacketizerCfg
from pathlib import Path

def assert_encoder_common(cfg:EncoderConfig):
    def check(cfg:EncoderConfig):
        assert len(cfg.buffers) == 2
        assert cfg.frame_width == 2048
        assert cfg.frame_height == 2048
        assert cfg.frame_rate == 60.
        assert cfg.cu_size == 64

        assert cfg._pre_delay.mode == "equally"
        assert cfg._pre_delay.parameter1 == 10e3 # ms
        assert cfg._pre_delay.parameter2 == 30e3 # ms
        assert cfg._pre_delay.parameter3 == 0e3 # unused

        assert cfg._encoding_delay.mode == "GaussianTrunc"
        assert cfg._encoding_delay.parameter1 == 8e3 # ms
        assert cfg._encoding_delay.parameter2 == 3e3 # ms
        assert cfg._encoding_delay.parameter3 == 0e3 # unused
    return check

def assert_rate_control(mode, bps, minqp, maxqp, window_frame_size):
    def check(cfg:EncoderConfig):
        assert cfg.rc_mode == mode
        assert cfg.rc_bitrate == bps
        assert cfg.rc_window_size == window_frame_size
        assert cfg.rc_qp_min == minqp
        assert cfg.rc_qp_max == maxqp
        assert cfg.target_qp == -1
    return check

def assert_intra_period(p):
    def check(cfg:EncoderConfig):
        assert cfg.error_resilience_mode == ErrorResilienceMode.PERIODIC_INTRA
        assert cfg.intra_refresh_period == p
    return check

def assert_num_slices(s):
    def check(cfg:EncoderConfig):
        assert cfg.slices_per_frame == s
    return check

def assert_buffer_interleaving(bint):
    def check(cfg:EncoderConfig):
        assert cfg.buffer_interleaving == bint
    return check


def assert_packetizer_common(cfg:PacketizerCfg):
    pass

def assert_pckt_bitrate(bps):
    def check(cfg:PacketizerCfg):
        assert cfg.bitrate == bps
    return check

def assert_mtu(max_size, overhead):
    def check(cfg:PacketizerCfg):
        assert cfg.pckt_max_size == max_size
        assert cfg.pckt_overhead == overhead
    return check


def test_config(key, cfg):
    for tool, assetions in cfg.items():
        cfg = Path(f'./VR/{key}.{tool}.json')
        for a in assetions:
            a(cfg)

VR_CFG = {
    'vr2-1': {
        'encoder':[
            assert_encoder_common,
            assert_rate_control(RC_mode.cVBR, 30e6, 0, 51, 12),
            assert_intra_period(1),
            assert_num_slices(8),
            assert_buffer_interleaving(False)
        ],
        'packetizer':[
            assert_packetizer_common,
            assert_mtu(1500, 40),
            assert_pckt_bitrate(45e6)
        ]
    },
    'vr2-2': {
        'encoder':[
            assert_encoder_common,
            assert_rate_control(RC_mode.cVBR, 30e6, 0, 51, 12),
            assert_intra_period(1),
            assert_num_slices(8),
            assert_buffer_interleaving(False)
        ],
        'packetizer':[
            assert_packetizer_common,
            assert_mtu(-1, 40),
            assert_pckt_bitrate(45e6)
        ]
    },

    'vr2-3': {
        'encoder':[
            assert_encoder_common,
            assert_rate_control(RC_mode.CBR, 30e6, 0, 51, 1),
            assert_intra_period(1),
            assert_num_slices(8),
            assert_buffer_interleaving(False)
        ],
        'packetizer':[
            assert_packetizer_common,
            assert_mtu(1500, 40),
            assert_pckt_bitrate(45e6)
        ]
    },
    'vr2-4': {
        'encoder':[
            assert_encoder_common,
            assert_rate_control(RC_mode.CBR, 30e6, 0, 51, 1),
            assert_intra_period(1),
            assert_num_slices(8),
            assert_buffer_interleaving(False)
        ],
        'packetizer':[
            assert_packetizer_common,
            assert_mtu(-1, 40),
            assert_pckt_bitrate(45e6)
        ]
    },

    'vr2-5': {
        'encoder':[
            assert_encoder_common,
            assert_rate_control(RC_mode.cVBR, 30e6, 0, 51, 12),
            assert_intra_period(8),
            assert_num_slices(1),
            assert_buffer_interleaving(False)
        ],
        'packetizer':[
            assert_packetizer_common,
            assert_mtu(1500, 40),
            assert_pckt_bitrate(45e6)
        ]
    },
    'vr2-6': {
        'encoder':[
            assert_encoder_common,
            assert_rate_control(RC_mode.cVBR, 30e6, 0, 51, 12),
            assert_intra_period(8),
            assert_num_slices(1),
            assert_buffer_interleaving(True)
        ],
        'packetizer':[
            assert_packetizer_common,
            assert_mtu(1500, 40),
            assert_pckt_bitrate(45e6)
        ]
    },

    'vr2-7': {
        'encoder':[
            assert_encoder_common,
            assert_rate_control(RC_mode.cVBR, 45e6, 0, 51, 12),
            assert_intra_period(1),
            assert_num_slices(8),
            assert_buffer_interleaving(False)
        ],
        'packetizer':[
            assert_packetizer_common,
            assert_mtu(1500, 40),
            assert_pckt_bitrate(67.5e6)
        ]
    },
    'vr2-8': {
        'encoder':[
            assert_encoder_common,
            assert_rate_control(RC_mode.cVBR, 45e6, 0, 51, 12),
            assert_intra_period(1),
            assert_num_slices(8),
            assert_buffer_interleaving(False)
        ],
        'packetizer':[
            assert_packetizer_common,
            assert_mtu(-1, 40),
            assert_pckt_bitrate(67.5e6)
        ]
    }
}

if __name__ == "__main__":

    for vr, cfg in VR_CFG.items():
        test_config(vr, cfg)