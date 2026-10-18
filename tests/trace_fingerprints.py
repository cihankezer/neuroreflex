"""Historical raw hashes plus a portable, nine-decimal trace fingerprint.

Branch decisions, labels, counts, and structure remain exact. Only finite float
serialization is quantized for cross-version comparison; original raw digests
are still asserted on the recorded CPython 3.13/Windows runtime family.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def portable_digest(ticks):
    def canonical(value):
        if isinstance(value, float) and math.isfinite(value):
            rounded = round(value, 9)
            return 0.0 if rounded == 0 else rounded
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        return value
    plain = json.loads(json.dumps(ticks))
    return hashlib.sha256(json.dumps(canonical(plain), sort_keys=True,
                         separators=(",", ":")).encode()).hexdigest()

def assert_reference_trace(test, ticks, expected_raw):
    raw = hashlib.sha256(json.dumps(ticks, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 13) and sys.platform == "win32":
        test.assertEqual(raw, expected_raw)
    reference = json.loads((ROOT/"benchmark_results/portable_trace_fingerprints.json").read_text())
    test.assertEqual(portable_digest(ticks), reference["raw_to_portable"][expected_raw])
