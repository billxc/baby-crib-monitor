# Project boundaries

- `tools/crop/` is the browser-only crop tool. `runtime/` is an offline historical single-image inference runner with a causal three-sample majority filter; live collection is not implemented.
- Keep this repository suitable for future publication. Never add real camera images, recordings, base64 camera footage, real calibration parameters, model outputs, credentials, local addresses or machine-specific deployment scripts, even while the repository is private.
- Keep runtime/private data outside the repository. `.local/` is ignored for temporary development only; do not force-add ignored files.
- Use synthetic fixtures for tests. Run `npm ci --include=dev` and `npm test`; browser tests keep Chromium sandbox enabled. Runtime tests: `python -m unittest discover -s runtime -p 'test_*.py'` after installing `runtime/requirements.txt`.
- Do not introduce an upload API or server-side persistence for the crop tool. It processes images locally and exports PNG/JSON for the user.
- JSON operations are ordered; crop coordinates refer to the immediately preceding output. Preserve clockwise rotation, expanded canvas and fill-color semantics.
- Check staged content and Git history before publishing; deleting a private asset from the current tree does not remove it from history.
