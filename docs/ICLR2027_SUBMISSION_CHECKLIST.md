# ICLR 2027 submission checklist

The authoritative schedule is the
[ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).
As verified on 2026-08-10:

- **2026-09-11 AOE:** genuine abstract registration; no authors may be added later.
- **2026-09-16 AOE:** full paper and supplement deadline.
- Main text is at most nine pages; references and appendices follow conference rules.
- The paper and supplement are double-blind. Related public preprints are cited in
  the third person.
- Include the required Artificial Intelligence use statement.

## Operational dates

- **August 10–13:** freeze protocol semantics and pilot/confirmatory manifests.
- **August 14–18:** generator, detector invariant, macro compiler, trace schema, and
  10,000-seed integrity tests.
- **August 19–23:** exact controls, five-seed BC runs, local serving preflight, and
  per-model output-ceiling selection.
- **August 24–31:** zero-shot confirmation and principal demonstration runs.
- **September 1–5:** replications, intervals, plots, exact-number audit.
- **September 6:** go/no-go and final claim freeze.
- **September 7–10:** nine-page assembly, appendix, anonymous supplement, AI-use and
  reproducibility statements.
- **September 11:** submit genuine abstract and final author list.
- **September 12–15:** adversarial, anonymity, metadata, IP/privileged-information,
  and PDF audits.
- **September 16:** full submission.

## Desk-rejection and disclosure checks

- [ ] Anonymous title page; no author names, email, acknowledgements, identity-bearing
  paths, PDF metadata, citation metadata, or Git history in the supplement.
- [ ] Cite the public v0.1 preprint in the third person; never say “our v0.1 paper.”
- [ ] Create an anonymous repository snapshot/mirror that does not point to a
  username-bearing public repository.
- [ ] Maintain an AI-use ledger and include the conference-required statement.
- [ ] Confirm all authors' OpenReview profiles and author list before September 11.
- [ ] Complete reciprocal-review eligibility/registration steps when notified.
- [ ] Run a privileged-information, IP, and legal-compliance pass before OpenReview
  becomes public; escalate uncertain material for qualified review.
- [ ] Run `pytest tests/ -v`, `ruff check .`, and `ruff format --check .`. No static
  type checker is claimed unless separately added and configured.
