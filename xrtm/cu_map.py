from models import CSV, CsvRecord, CsvEnum

class CTU_type(CsvEnum):
    INTRA           = 0
    INTER           = 1
    MERGE           = 2
    SKIP            = 3

class CTU_status(CsvEnum):
    OK              = 0
    DAMAGED         = 1
    UNAVAILABLE     = 2

class CTU(CsvRecord):

    attributes = [
        CSV("address", int) # Address of CU in frame.
        CSV("size", int) # Slice size in bytes.
        CSV("mode", CTU_type.parse, CTU_type.serialize, None) # The mode of the CU 1=intra, 2=merge, 3=skip, 4=inter
        CSV("reference", int) # The reference frame of the CU 0 n/a, 1=previous, 2=2 in past, etc.
        # CSV("qp", int) # the QP decided for the CU
        # CSV("psnr_y", int) # the estimated Y-PSNR for the CU in db multiplied by 1000
        # CSV("psnr_yuv", int) # the estimated weighted YUV-PSNR for the CU db multiplied by 1000
    ]

class CtuMap:
    
    def __init__(self, count:int):
        self.count = count
        self._map:List[CTU] = [None] * count

    @classmethod
    def draw_ctus(cls, count:int, intra:float, inter:float, skip:float, merge:float, intra_mean:int, inter_mean:int, refs:List[int]=None) -> List[CTU]:
        return random.choices([
            CTU(CTU_type.INTRA, intra_mean, None),
            CTU(CTU_type.INTER, inter_mean, ref),
            CTU(CTU_type.SKIP, inter_mean, ref),
            CTU(CTU_type.MERGE, inter_mean, ref),
        ], weights=[intra, inter, skip, merge], k=count)

    def draw(self, *args, **kwargs):
        self._map = self.draw_ctus(*args, **kwargs)

    def draw_intra(self, intra_mean:int, idx:int=0, count:int=None) -> List[CTU]:
        end = None
        if count:
            end = idx + count
        self._map[idx:end] = [CTU(CTU_type.INTRA, intra_mean, None)] * count

    @staticmethod
    def stats(m:List[CTU]) -> SliceStats:
        s = SliceStats()
        return s

    def get_slice(self, index:int, count:int) -> List[CTU]:
        stop = index + count
        return self._map[index:stop]

    def update_slice(self, i:int, m:List[CTU]):
        stop = i + len(m)
        assert stop <= self.count
        self._map[i:stop] = m

    def dump(self, csv_out:str):
        with open(csv_out, 'w') as f:
            writer = CTU.get_csv_writer(f)
            for ctu in self._map:
                writer.writerow(ctu)

    def load(self, csv_in:str) -> 'CtuMap':
        data = []
        with open(csv_in, 'r') as f:
            data = [ctu for ctu in CTU.iter_csv_file(csv_in)]
        assert len(data) == self.count
        self._map = data
