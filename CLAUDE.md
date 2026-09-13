# NLP-Lab — working agreements

## Commits
- **Never add `Co-Authored-By: Claude ...` trailers.** Commits are authored by the repo
  owner alone, so GitHub shows a single avatar rather than "Sophia-Min12 and claude".
- No "Generated with Claude Code" footers in commit messages or PR descriptions.

## README roadmap
- **Do not add dates** to roadmap entries. Checking the box (`- [x]`) is the only
  completion marker; the git history already carries the timestamps.

## Language
- All repo content — README, docstrings, comments, commit messages — is written in English.
- Korean strings inside test data and demo samples stay as-is: they are functional
  fixtures for Unicode-aware tokenization, not prose.
