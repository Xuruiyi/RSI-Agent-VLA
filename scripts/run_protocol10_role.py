"""Fixed protocol10 production role entry point, invoked by the host broker."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))
from rsi_vla.protocol10_roles import role_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    raise SystemExit(role_main(parser.parse_args().request))
