"""Reversible JSON factoring for complete development requests, not summarization."""

import json
from collections import Counter


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def pack(value):
    """Share repeated subtrees/constants and dictionary field names; keep all values."""
    counts = Counter()

    def count(item):
        if isinstance(item, (dict, list, str)):
            counts[compact(item)] += 1
        if isinstance(item, dict):
            for child in item.values():
                count(child)
        elif isinstance(item, list):
            for child in item:
                count(child)

    count(value)
    fields, shapes, shared, ids = [], {}, [], {}

    def encode(item, defining=False):
        key = compact(item) if isinstance(item, (dict, list, str)) else None
        if key is not None and counts[key] > 1 and len(key) > 32 and not defining:
            if key not in ids:
                ids[key] = len(shared)
                shared.append(None)
                shared[ids[key]] = encode(item, True)
            return {"r": ids[key]}
        if isinstance(item, dict):
            names = tuple(item)
            if names not in shapes:
                shapes[names] = len(fields)
                fields.append(list(names))
            return {"d": [shapes[names], [encode(v) for v in item.values()]]}
        if isinstance(item, list):
            return [encode(v) for v in item]
        return item

    root = encode(value)
    return {
        "schema": "rsi-lossless-json-v1",
        "decoding": "All facts are included. Recursively decode plain arrays elementwise; {r:i} is shared[i]; "
        "{d:[i,values]} zips fields[i] with decoded values into a dictionary. Other scalars are literal. "
        "No L0 facts are selected out. Missing/unknown and source references retain their original meaning.",
        "fields": fields,
        "shared": shared,
        "root": root,
    }


def unpack(document):
    """Restore the exact JSON value represented by this finite transport codec."""
    if document.get("schema") != "rsi-lossless-json-v1":
        raise ValueError("unknown research transport")

    def decode(item):
        if isinstance(item, list):
            return [decode(v) for v in item]
        if isinstance(item, dict):
            if set(item) == {"r"}:
                return decode(document["shared"][item["r"]])
            if set(item) == {"d"}:
                index, values = item["d"]
                names = document["fields"][index]
                if len(names) != len(values):
                    raise ValueError("transport field count mismatch")
                return dict(zip(names, [decode(v) for v in values]))
            raise ValueError("unknown transport node")
        return item

    return decode(document["root"])


def model_prompt(transport):
    """This exact representation is consumed by the real Research SDK path."""
    unpack(transport)
    return (
        "Review the complete legal development request below using its reversible codec. "
        "Retain uncertainty, fault validity and competing explanations. Choose one allowed "
        "evidenced intervention or defer. Do not inspect release/sealed sources. "
        "No external tools or experiments are available. Return the bounded response schema.\n"
        + compact(transport)
    )
