# Evidence release policy for supervisor review

Status: REVIEW_REQUIRED. This document grants no release permission.

The plan requests raw results in Git. The current project keeps raw predictions,
matching GT, images and checkpoints outside public Git. Neither rule is silently waived.

| Material | Current handling | Decision required |
|---|---|---|
| Source, UVLs, small source-linked summaries | Local Git commits | Review before public push |
| Raw predictions and derived matching GT | Immutable local RunIDs and SHA256 | Check data terms and privacy before release |
| Source images and model weights | External local storage | Confirm redistribution rights |
| Detailed environment and local logs | Private evidence | Remove credentials and private paths before release |

Two options remain for supervisor approval: an access-controlled/LFS raw repository,
or an immutable external evidence archive with version identifiers, SHA256 and Git-commit
links. The latter needs an explicit exception to the plan's raw-in-repository requirement.
No external archive has been published and no remote push was performed.

For each released Run, include its manifest, exact command, source commit, environment,
input hashes, raw predictions, matching GT, metric and bootstrap replay commands,
figure script and output checksums. Verify hashes before replay. Missing archives
must produce a clear error, never a fallback metric.
