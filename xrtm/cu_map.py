from models import CSV, CsvRecord, CsvEnum



class CTU(CsvRecord):

    attributes = [
        CSV("address", int) # Address of CU in frame.
        CSV("size", int) # Slice size in bytes.
        CSV("mode", CTU_mode.parse, CTU_mode.serialize, None) # The mode of the CU 1=intra, 2=merge, 3=skip, 4=inter
        CSV("reference", int) # The reference frame of the CU 0 n/a, 1=previous, 2=2 in past, etc.
        # CSV("qp", int) # the QP decided for the CU
        # CSV("psnr_y", int) # the estimated Y-PSNR for the CU in db multiplied by 1000
        # CSV("psnr_yuv", int) # the estimated weighted YUV-PSNR for the CU db multiplied by 1000
    ]


