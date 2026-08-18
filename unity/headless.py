#!/usr/bin/env python3
"""Headless provider-only mode. Starts on run, no GUI, same config files as the main app."""

import asyncio
import logging
import signal
import sys

# Ensure PyQt6-related speedups are never imported (saves time/memory)
sys.modules["websockets.speedups"] = None

from config import ConfigManager
from provider_client import ProviderClient


def main():
    # Root logger so provider logs appear on stdout/stderr
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = ConfigManager()

    shutdown_event = asyncio.Event()

    # Allow Ctrl+C to shut down cleanly
    loop = asyncio.new_event_loop()

    def _sigint_handler():
        loop.call_soon_threadsafe(shutdown_event.set)

    signal.signal(signal.SIGINT, lambda *_: _sigint_handler())
    signal.signal(signal.SIGTERM, lambda *_: _sigint_handler())

    async def run():
        p_client = ProviderClient(
            config,
            status_callback=lambda s: print(f"[status] {s}"),
            log_callback=lambda msg, level=logging.INFO: logging.getLogger("thinkfarm.provider").log(level, msg),
        )

        task = loop.create_task(p_client.run())

        # Wait for shutdown signal
        await shutdown_event.wait()
        print("\n[headless] Shutting down provider client...")

        await p_client.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    loop.run_until_complete(run())
    loop.close()
    print("[headless] Done.")


if __name__ == "__main__":
    main()
