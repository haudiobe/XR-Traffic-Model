from xrtm.models import (
    VTrace,
    SliceType
)

def v_trace(slice_type:SliceType):
    v = VTrace()
    v.pts = 0
    v.poc = 0
    v.crf_ref = 0
    v.intra_total_bits = 1000000
    v.intra_qp_ref = 0
    v.inter_total_bits = 1000000
    v.inter_qp_ref = 0
    
    if slice_type == SliceType.IDR:
        v.ctu_intra_pct = 1.0
        v.ctu_inter_pct = 0
        v.ctu_skip_pct = 0
        v.ctu_merge_pct = 0

    elif slice_type == SliceType.P:
        v.ctu_intra_pct = 0.25
        v.ctu_inter_pct = 0.25
        v.ctu_skip_pct = 0.25
        v.ctu_merge_pct = 0.25

    else:
        v.ctu_intra_pct = 0
        v.ctu_inter_pct = 0
        v.ctu_skip_pct = 0
        v.ctu_merge_pct = 0

    v.intra_y_psnr = 0
    v.intra_u_psnr = 0
    v.intra_v_psnr = 0
    v.intra_yuv_psnr = 0
    v.inter_y_psnr = 0
    v.inter_u_psnr = 0
    v.inter_v_psnr = 0
    v.inter_yuv_psnr = 0
    v.intra_total_time = 0
    v.inter_total_time = 0
    return v


def gen_vtrace_pattern(pattern="IPPPPP", poc=0, fps_num:int=60, fps_den:int=1):
    for i, slice_type in enumerate([SliceType.IDR if s == 'I' else SliceType.P for s in pattern]):
        v = v_trace(slice_type)
        v.poc = poc + i
        v.pts = int( 1000000 * v.poc * fps_den / fps_num )
        yield v


def gen_vtrace(fps_num:int=60, fps_den:int=1, length:int=600, gop=-1):
    i = 0
    for poc in range(length):
        if i == 0:
            v = v_trace(SliceType.IDR)
        elif gop <= 0:
            v = v_trace(SliceType.P)
        else:
            v = v_trace(SliceType.IDR) if ((i+1)%gop) == 0 else v_trace(SliceType.P)
        i += 1
        v.poc = poc
        v.pts = int( 1000000 * poc * fps_den / fps_num )
        yield v