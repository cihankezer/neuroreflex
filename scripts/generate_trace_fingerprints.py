"""Derive portable fingerprints only after matching every original raw trace hash."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tests"))
from trace_fingerprints import portable_digest
from run_dynamic_cases import run_dynamic_cases
from run_release_ablation import run_policy_set
from perception import PerceptionConfig
from reflex_release import ReflexReleaseConfig
from threat_gate import ThreatGateConfig


def main():
    if sys.version_info[:2] != (3, 13) or sys.platform != "win32":
        raise SystemExit("Generate only on the original CPython 3.13/Windows reference family.")
    mappings = {}
    def add(ticks, expected):
        raw=hashlib.sha256(json.dumps(ticks,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
        assert raw == expected, "Historical reference changed; do not regenerate expected outcomes."
        mappings[raw]=portable_digest(ticks)
    release=json.loads((ROOT/"benchmark_results/release_baseline.json").read_text())
    for label,count,lo,hi in (("fixed",6,1.2,1.2),("intrusion",24,.3,2),("stress",24,2,6)):
        for noisy in (False,True):
            for mode in ("legacy","geometric"):
                cfg=PerceptionConfig(.01,.1,.02,.05) if noisy else PerceptionConfig()
                results=run_dynamic_cases(count,42,cfg,lo,hi,threat_gate=ThreatGateConfig(mode))
                expected=release[f"{label}_{'noisy' if noisy else 'clean'}_{mode}"]
                for result,old in zip(results,expected):
                    add([{k:v for k,v in tick.items() if k in old["tick_fields"]} for tick in result.ticks],old["ticks_sha256"])
    stateful=json.loads((ROOT/"benchmark_results/robustness_check/stateful_before.json").read_text())
    for (benchmark,noise),results in run_policy_set(42,"geometric_hysteresis",ReflexReleaseConfig(),.5).items():
        for result,old in zip(results,stateful[benchmark+"_"+noise]):
            add(result.ticks,old["ticks_sha256"])
    document={"reference_runtime":"CPython 3.13 on Windows", "finite_float_decimal_places":9,
              "original_raw_hashes_preserved":True,"raw_to_portable":mappings}
    (ROOT/"benchmark_results/portable_trace_fingerprints.json").write_text(json.dumps(document,indent=2)+"\n",encoding="utf-8")
    print("Verified original raw traces and derived",len(mappings),"portable fingerprints.")

if __name__ == "__main__":
    main()
