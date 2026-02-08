"""Entry point for python -m src.mcp_server."""

import asyncio

from src.mcp_server.server import main

asyncio.run(main())
