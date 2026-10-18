"""Inspect the complete staged release for excluded content, sensitive strings and broken links."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import zipfile
import io

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "github_token": r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})",
    "aws_access_key": r"(?:AKIA|ASIA)[A-Z0-9]{16}",
    "slack_token": r"xox[baprs]-[A-Za-z0-9-]{20,}",
    "local_profile_path": r"(?:[A-Za-z]:[\\/]+Users[\\/]+|/[h]ome/|/[U]sers/)",
    "internal_url": r"https?://(?:localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|[^/\s]+\.internal)(?:[:/\s]|$)",
    "assigned_secret": r"(?im)^(?:api_key|access_token|secret_key|password)\s*=\s*[\"'][^\"'\n]{12,}[\"']",
}

def main():
    names = subprocess.check_output(["git", "diff", "--cached", "--name-only", "-z"], cwd=ROOT).decode().split("\0")
    names = [n for n in names if n]
    findings=[]
    manifest=[]
    for name in names:
        data=subprocess.check_output(["git", "show", ":"+name], cwd=ROOT)
        manifest.append({"path":name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
        if name.startswith(("_private_review/", "experiments/", ".venv", "data/")) or name.endswith((".aedat4", ".pptx", ".pdf")) or name == "looming_arm_3d.py":
            findings.append({"file":name,"kind":"excluded_asset"})
        texts=[data.decode("utf-8",errors="replace")]
        if name.endswith(".xlsx"):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                texts.extend(archive.read(n).decode("utf-8",errors="replace") for n in archive.namelist() if n.endswith((".xml",".rels")))
        for text in texts:
            for kind,pattern in PATTERNS.items():
                if re.search(pattern,text):findings.append({"file":name,"kind":kind})
        if name.endswith(".md"):
            for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",texts[0]):
                if target.startswith(("http://","https://","#","mailto:")):continue
                path=target.split("#",1)[0]
                if path and not (ROOT/name).parent.joinpath(path).exists():
                    findings.append({"file":name,"kind":"broken_relative_link","target":target})
    report={"staged_file_count":len(names),"staged_bytes":sum(x["bytes"] for x in manifest),"findings":findings,"files":manifest}
    out=ROOT/"_private_review/staged-review.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="files"}))
    raise SystemExit(bool(findings))

if __name__ == "__main__":
    main()
