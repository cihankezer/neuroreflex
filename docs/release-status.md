# Release scope and verification

The author authorized only their own work, identified their Git identity as `ge58puv`, and identified the shared `looming_arm_3d.py` as excluded. A per-file line-authorship audit found only that identity in 51 selected source/configuration/dependency files. Shared and teammate code, recordings, presentations, PDF papers, and archived private material are excluded from the staged release. The author explicitly selected no license grant.

The working directory was created outside the original repository and initialized with `git init` on a new `main` branch. No original `.git` content, branch, remote, or commit history was copied. The original tracked files, pre-existing modifications, untracked presentation files, Git HEAD/config/index/packed-refs, and status are checked against a pre-work fingerprint in the private audit folder.

Only the explicit reviewed release file list is staged. `_private_review/`, `experiments/`, environments, raw event data, and runtime outputs remain excluded. The local private review snapshot is not part of the Git repository and must never be distributed as an archive.

GitHub target: `cihankezer/neuroreflex`, a new personal repository. Authentication was verified read-only and the target did not exist at preflight. Publication follows completed file, secret, attribution, command, link, package, and test review. GitHub metadata is in `docs/github-metadata.json`.
