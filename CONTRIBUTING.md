# Contributing

This repository accompanies an anonymous submission under double-blind review.
During the review period, external contributions are not accepted.

After acceptance, the repository will be de-anonymized and open to contributions.
At that point:

- Fork the repository and create a feature branch.
- Add tests for any new method or metric under `tests/`.
- Ensure `python -m pytest` passes and `python -c "import core, methods"` succeeds.
- Run `python tools/scrub_appids.py --check` before committing if you have
  edited `config.yaml`, to avoid leaking gateway credentials.

## Code layout

See the "Repository layout" section of the README. New agents should depend
only on the abstract `BaseLLM` / `BaseRetriever` interfaces in
`core/base.py`, not on concrete API clients. Prompt strings for new
methods should live under `prompts/`.
