import logging
import asyncio
from typing import Callable

class WebSocketBroadcastHandler(logging.Handler):
    """
    A custom logging handler that broadcasts log records over a WebSocket.
    """

    def __init__(self, broadcast_func: Callable, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.broadcast_func = broadcast_func
        self.loop = loop

    def emit(self, record: logging.LogRecord):
        """
        Formats the log record and schedules it for broadcast.
        """
        # We need to run the async broadcast function in the event loop
        # from this synchronous context.
        if self.loop and self.loop.is_running():
            log_data = {
                "type": "log",
                "level": record.levelname,
                "name": record.name,
                "message": self.format(record) # Use the handler's formatter
            }
            asyncio.run_coroutine_threadsafe(self.broadcast_func(log_data), self.loop)
