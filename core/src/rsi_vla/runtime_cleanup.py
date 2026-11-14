"""Owner and request closure independent of successful execution validity."""

import json
from pathlib import Path


def inspect_cleanup(output, identity):
    try:
        root = Path(output)
        supervisor, gate, launcher = [
            json.loads((root / name).read_text())
            for name in ("supervisor.json", "request_gate.json", "launcher_costs.json")
        ]
        if not all(
            all(
                document.get("identity", document).get(k) == v
                for k, v in identity.items()
            )
            for document in (supervisor, gate, launcher)
        ):
            return False
        return (
            supervisor.get("cleanup_confirmed") is True
            and supervisor.get("remaining_process_ids") == []
            and gate.get("cleanup_complete") is True
            and gate.get("active_requests_at_close") == 0
            and gate.get("unfinished_request_numbers") == []
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
