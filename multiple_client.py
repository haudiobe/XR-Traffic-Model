import asyncio

async def control_client(loop):
    reader, writer = await asyncio.open_connection('127.0.0.1', 8000, loop=loop)

    peer = writer.get_extra_info('peername')
    hello_msg = "Hello, {0[0]}:{0[1]}!\n".format(peer).encode("utf-8")
    training_res = "Training, {0[0]}:{0[1]}!\n".format(peer).encode("utf-8")
    config_res = "Config, {0[0]}:{0[1]}!\n".format(peer).encode("utf-8")

    while True:

        data = await reader.read(100)
        if len(data) > 0:
            message = data.decode()
            print('Received: %r' % message)

            if message.startswith('Hello'):
                writer.write(hello_msg)
            elif message.startswith('Training'):
                writer.write(training_res)
            elif message.startswith('Config'):
                writer.write(config_res)

    print('Close the socket')
    writer.close()


async def content_client(loop):
    reader, writer = await asyncio.open_connection('127.0.0.1', 8001, loop=loop)
    while True:
        data = await reader.read(100)
        if len(data) > 0:
            print('Received: %r' % data)

    print('Close the socket')
    writer.close()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(control_client(loop))
    loop.create_task(content_client(loop))
    loop.run_forever()
