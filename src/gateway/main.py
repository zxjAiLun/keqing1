import argparse
import asyncio
import datetime
import json
import logging
import sys
from asyncio import StreamReader, StreamWriter
from logging import config
from typing import Awaitable, Callable

from gateway.utils.state import State
from gateway import settings
from gateway.tenhou_bridge import TenhouBridge, TenhouBridgeError, is_valid_tenhou_room

logger = logging.getLogger(__name__)


def sender_to_mjai(reader: StreamReader, writer: StreamWriter) -> Callable[[dict], Awaitable[dict]]:
    async def send_to_mjai(message: dict) -> dict:
        writer.write((json.dumps(message) + '\n').encode())
        await writer.drain()
        received = (await reader.readuntil()).decode()
        return json.loads(received)

    return send_to_mjai


async def websocket_client(send_to_mjai: Callable[[dict], Awaitable[dict]], state: State) -> None:
    bridge = TenhouBridge(state=state, send_to_mjai=send_to_mjai)
    await bridge.run()


async def tcp_server(reader: StreamReader, writer: StreamWriter) -> None:
    send_to_mjai = sender_to_mjai(reader, writer)
    try:
        message = await send_to_mjai(
            {'type': 'hello', 'protocol': 'mjsonp', 'protocol_version': 3}
        )
        name: str = message['name']
        room: str = message['room']

        if is_valid_tenhou_room(room):
            state = State(name, room)
            await websocket_client(send_to_mjai, state)
        else:
            writer.write(
                json.dumps({'type': 'error', 'message': f'invalid room: {room}'}).encode()
            )
            await writer.drain()
    except TenhouBridgeError as exc:
        logger.warning("tenhou bridge closed for client: %s", exc)
    finally:
        writer.close()
        await writer.wait_closed()


async def main() -> None:
    server = await asyncio.start_server(tcp_server, settings.HOST, settings.PORT)

    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-o', '--output', type=str, default='logs')
    args = parser.parse_args()

    settings.DEBUG = args.debug
    settings.LOGGING['handlers']['file']['filename'] = \
        '{}/{}.log'.format(args.output, datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S'))

    config.dictConfig(settings.LOGGING)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
