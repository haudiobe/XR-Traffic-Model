import asyncio
from csv import DictReader
from enum import Enum


class ClientContext:
    _event = asyncio.Event()

    @property
    def state(self):
        return 'IDLE' if not type(self)._event.is_set() else 'CONFIGURED'

    @state.setter
    def state(self, val):
        if val == 'IDLE':
            type(self)._event.clear()
        elif val == 'CONNECTED':
            type(self)._event.clear()
        elif val == 'TRAINED':
            type(self)._event.clear()
        elif val == 'CONFIGURED':
            type(self)._event.set()
        else:
            raise ValueError('Bad state value.')

    async def is_configured(self):
        return await type(self)._event.wait()

    async def handle_control(self, reader, writer):
        peer = writer.get_extra_info('peername')
        hello_msg = "Hello, {0[0]}:{0[1]}!\n".format(peer).encode("utf-8")
        training_req = "Training, {0[0]}:{0[1]}!\n".format(peer).encode("utf-8")
        config_req = "Config, {0[0]}:{0[1]}!\n".format(peer).encode("utf-8")

        writer.write(hello_msg)

        while True:
            data = await reader.read(100)
            if len(data) > 0:
                message = data.decode()
                print('Received:', message)
                if message.startswith('Hello'):
                    self.state = 'CONNECTED'
                    print('New State:', self.state)
                    writer.write(training_req)
                elif message.startswith('Training'):
                    self.state = 'TRAINED'
                    print('New State:', self.state)
                    writer.write(config_req)
                elif message.startswith('Config'):
                    self.state = 'CONFIGURED'
                    print('New State:', self.state)
                elif message.startswith('End'):
                    self.state = 'IDLE'
                    print('New State:', self.state)
                    break

        writer.close()

    async def handle_content(self, reader, writer):
        peer = writer.get_extra_info('peername')

        await self.is_configured()

        for i in range(1, 10):
            with open('00001.csv') as csvfile:
                csv_dict_reader = DictReader(csvfile)
                for row in csv_dict_reader:
                    row_data = CSVRow(row)
                    print('Send: Msg {0}\n'.format(row))
                    writer.write("Msg {0}\n".format(row).encode("utf-8"))
                    await asyncio.sleep(1)

        writer.close()


class CSVRow:
    m_order = None
    m_type = None
    m_poc = None
    m_qp = None
    m_bits = None
    m_scenecut = None
    m_ratefactor = None
    m_psnr_y = None
    m_psnr_u = None
    m_psnr_v = None
    m_psnr_yuv = None
    m_ssim = None
    m_ssim_db = None
    m_latency = None
    m_list0 = None
    m_list1 = None
    m_intra_64_64_dc = None
    m_intra_64_64_planar = None
    m_intra_64_64_ang = None
    m_intra_32_32_dc = None
    m_intra_32_32_planar = None
    m_intra_32_32_ang = None
    m_intra_16_16_dc = None
    m_intra_16_16_planar = None
    m_intra_16_16_ang = None
    m_intra_8_8_dc = None
    m_intra_8_8_planar = None
    m_intra_8_8_ang = None
    m_4_4 = None
    m_inter_64_64 = None
    m_inter_32_32 = None
    m_inter_16_16 = None
    m_inter_8_8 = None
    m_skip_64_64 = None
    m_skip_32_32 = None
    m_skip_16_16 = None
    m_skip_8_8 = None
    m_merge_64_64 = None
    m_merge_32_32 = None
    m_merge_16_16 = None
    m_merge_8_8 = None
    m_avg_luma_dist = None
    m_avg_chroma_dist = None
    m_avg_psy_energy = None
    m_avg_luma_level = None
    m_max_luma_level = None
    m_avg_res_energy = None
    m_decide_wait = None
    m_row0_wait = None
    m_wall_time = None
    m_ref_wait_wall = None
    m_total_ctu_time = None
    m_stall_time = None
    m_total_frame_time = None
    m_avg_wpp = None
    m_row_blocks = None

    def __init__(self, data):
        self.m_order = data['Encode Order']
        self.m_type = data[' Type']
        self.m_poc = data[' POC']
        self.m_qp = data[' QP']
        self.m_bits = data[' Bits']
        self.m_scenecut = data[' Scenecut']
        self.m_ratefactor = data[' RateFactor']
        self.m_psnr_y = data[' Y PSNR']
        self.m_psnr_u = data[' U PSNR']
        self.m_psnr_v = data[' V PSNR']
        self.m_psnr_yuv = data[' YUV PSNR']
        self.m_ssim = data[' SSIM']
        self.m_ssim_db = data[' SSIM(dB)']
        self.m_latency = data[' Latency']
        self.m_list0 = data[' List 0']
        self.m_list1 = data[' List 1']
        self.m_intra_64_64_dc = data[' Intra 64x64 DC']
        self.m_intra_64_64_planar = data[' Intra 64x64 Planar']
        self.m_intra_64_64_ang = data[' Intra 64x64 Ang']
        self.m_intra_32_32_dc = data[' Intra 32x32 DC']
        self.m_intra_32_32_planar = data[' Intra 32x32 Planar']
        self.m_intra_32_32_ang = data[' Intra 32x32 Ang']
        self.m_intra_16_16_dc = data[' Intra 16x16 DC']
        self.m_intra_16_16_planar = data[' Intra 16x16 Planar']
        self.m_intra_16_16_ang = data[' Intra 16x16 Ang']
        self.m_intra_8_8_dc = data[' Intra 8x8 DC']
        self.m_intra_8_8_planar = data[' Intra 8x8 Planar']
        self.m_intra_8_8_ang = data[' Intra 8x8 Ang']
        self.m_4_4 = data[' 4x4']
        self.m_inter_64_64 = data[' Inter 64x64']
        self.m_inter_32_32 = data[' Inter 32x32']
        self.m_inter_16_16 = data[' Inter 16x16']
        self.m_inter_8_8 = data[' Inter 8x8']
        self.m_skip_64_64 = data[' Skip 64x64']
        self.m_skip_32_32 = data[' Skip 32x32']
        self.m_skip_16_16 = data[' Skip 16x16']
        self.m_skip_8_8 = data[' Skip 8x8']
        self.m_merge_64_64 = data[' Merge 64x64']
        self.m_merge_32_32 = data[' Merge 32x32']
        self.m_merge_16_16 = data[' Merge 16x16']
        self.m_merge_8_8 = data[' Merge 8x8']
        self.m_avg_luma_dist = data[' Avg Luma Distortion']
        self.m_avg_chroma_dist = data[' Avg Chroma Distortion']
        self.m_avg_psy_energy = data[' Avg psyEnergy']
        self.m_avg_luma_level = data[' Avg Luma Level']
        self.m_max_luma_level = data[' Max Luma Level']
        self.m_avg_res_energy = data[' Avg Residual Energy']
        self.m_decide_wait = data[' DecideWait (ms)']
        self.m_row0_wait = data[' Row0Wait (ms)']
        self.m_wall_time = data[' Wall time (ms)']
        self.m_ref_wait_wall = data[' Ref Wait Wall (ms)']
        self.m_total_ctu_time = data[' Total CTU time (ms)']
        self.m_stall_time = data[' Stall Time (ms)']
        self.m_total_frame_time = data[' Total frame time (ms)']
        self.m_avg_wpp = data[' Avg WPP']
        self.m_row_blocks = data[' Row Blocks']


if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    ctx = ClientContext()
    servers = []

    i = 0
    print("Starting Control server {0}".format(i+1))
    server = loop.run_until_complete(
            asyncio.start_server(ctx.handle_control, '127.0.0.1', 8000+i, loop=loop))
    servers.append(server)

    i += 1
    server = loop.run_until_complete(
            asyncio.start_server(ctx.handle_content, '127.0.0.1', 8000+i, loop=loop))
    servers.append(server)

    try:
        print("Running... Press ^C to shutdown")
        loop.run_forever()
    except KeyboardInterrupt:
        pass

    for i, server in enumerate(servers):
        print("Closing server {0}".format(i+1))
        server.close()
        loop.run_until_complete(server.wait_closed())
    loop.close()
