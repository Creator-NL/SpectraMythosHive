# Accessibility Statement

Biohazard Corp strives to keep **Spectra Mythos Hive** usable and readable for everyone across Windows, Linux, terminal workflows, and web documentation.

## Terminal & CLI Accessibility

* **UTF-8 Stream Safety:** Core entrypoints invoke `ensure_utf8_stdio()` in [`hive_common.py`](hive_common.py) so diagnostic tables and status lines render reliably in Windows CMD and PowerShell without encoding crashes.
* **Plain-Text Readability:** Report cards in [`Tools/eval_mythos.py`](Tools/eval_mythos.py) and layer diffs in [`Tools/checkpoint_diff.py`](Tools/checkpoint_diff.py) include numeric scores, text labels (`Master`, `Advanced`, `Proficient`, `Developing`, `Novice`, `Untrained`), and tabular alignments rather than relying on color or emojis alone.
* **No Forced Audio/Timing Hurdles:** Interactive model testers ([`test_model.py`](test_model.py), [`test_model_phase2.py`](test_model_phase2.py)) use standard line-buffered text prompts with clean `quit`/`exit` and `Ctrl+C` exits.

## Documentation Accessibility

* Interactive browser documentation at [biohazard-corp.nl/openmythos](https://biohazard-corp.nl/openmythos) supports automatic system dark/light contrast themes, keyboard tab navigation, `prefers-reduced-motion`, ARIA live regions, and semantic HTML landmarks.
* Repository documentation pairs Mermaid diagrams with clear prose descriptions and step-by-step commands.

## Feedback & Accommodations

If any CLI script, `.bat` launcher, output table, or documentation page creates an accessibility barrier (such as screen-reader verbosity, color-contrast issues, or keyboard navigation bugs), please open a GitHub Issue or contact **contact@biohazard-corp.nl**.
