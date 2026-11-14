#!/usr/bin/env python3
"""Run the fixed Autoloop command adapter for one protocol10 context."""

import argparse
import json
import os
import sys
import signal
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))
from rsi_vla.autoloop_bridge import AutoloopBridge, AutoloopHost, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", action="store_true", help="serve the trusted fixed core ingress"
    )
    parser.add_argument("--configuration", required=True, type=Path)
    args = parser.parse_args()
    try:
        config = json.loads(args.configuration.read_text())
        if args.host:
            stop = threading.Event()
            signal.signal(signal.SIGTERM, lambda *_: stop.set())
            signal.signal(signal.SIGINT, lambda *_: stop.set())
            AutoloopHost(config).serve(stop)
            return 0
        context = json.loads(Path(os.environ["AGENT_AUTOLOOP_CONTEXT"]).read_text())
        result = AutoloopBridge(config).consume(context)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        result = {
            "summary": "待你处理：" + str(exc),
            "ask": [
                {
                    "title": "protocol10 入口等待核验",
                    "text": str(exc),
                    "options": ["保持暂停"],
                    "key": "protocol10-bridge-input",
                }
            ],
        }
    print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
