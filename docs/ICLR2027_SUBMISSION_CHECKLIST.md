# ICLR 2027 submission checklist

The authoritative schedule is the
[ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).
As verified on 2026-08-12:

- **2026-09-18 AOE:** official genuine-abstract deadline; no authors may be added later.
- **2026-09-25 AOE:** official full-paper and supplement deadline.
- **2026-09-11:** internal abstract/author freeze.
- **2026-09-16:** internal paper freeze for final audits.
- Main text is at most nine pages; references and appendices follow conference rules.
- The paper and supplement are double-blind. This submission treats EC as its first
  public paper; unpublished developmental manuscripts remain outside the review package.
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
- **September 11:** internal genuine-abstract and final-author freeze.
- **September 12–16:** adversarial, anonymity, metadata, and PDF audits; internal
  paper freeze on September 16.
- **September 17:** submit the genuine abstract and author list ahead of the official
  September 18 AOE deadline.
- **September 18–24:** final reproducibility and privileged-information/IP/legal pass.
- **September 24:** submit the paper ahead of the official September 25 AOE deadline.

## Desk-rejection and disclosure checks

- [ ] Anonymous title page; no author names, email, acknowledgements, identity-bearing
  paths, PDF metadata, citation metadata, or Git history in the supplement.
- [ ] Confirm that no unpublished-developmental-history framing appears in the paper
  or anonymous supplement.
- [ ] Choose the least-exposing anonymous code route: an anonymous supplement/repo,
  or after discussion opens, a reviewer-and-AC-only OpenReview comment containing
  the anonymous repository link.
- [ ] Maintain an AI-use ledger and include the conference-required statement.
- [ ] Create or audit every author's OpenReview profile now. Profiles without an
  institutional email may take up to two weeks to moderate; verify every email and
  profile field accurately before the internal September 11 author freeze.
- [ ] Determine reciprocal-review eligibility only from qualifying primary papers
  accepted by the abstract deadline. Workshop/position papers do not qualify. If no
  author qualifies, record the exemption and enforce the one-submission-per-author cap.
- [ ] If a substantially similar paper is under NeurIPS review, use only the explicit
  FAQ allowance: register an ICLR abstract, then withdraw before full submission if
  NeurIPS accepts. Do not maintain simultaneous full submissions.
- [ ] Treat September 25 as the final privileged-information/IP gate. Withdrawal after
  the paper deadline leaves a public, immediately de-anonymized, non-deletable record.
- [ ] Run a privileged-information, IP, and legal-compliance pass before OpenReview
  becomes public; escalate uncertain material for qualified review.
- [ ] Run `pytest tests/ -v`, `ruff check .`, and `ruff format --check .`. No static
  type checker is claimed unless separately added and configured.
- [ ] Regenerate Tables 1--4 and Figures 1--4 with
  `python experiments/build_submission_assets.py`; verify the committed asset manifest.
- [ ] Build the anonymous code/results ZIP with
  `python experiments/build_anonymous_supplement.py --output <submission.zip>` and
  extract it into a clean directory before running its documented tests and asset command.
- [ ] Verify the local final-submission manifest hashes both anonymous packages, the
  final PDF, and every compact scientific artifact before upload.
