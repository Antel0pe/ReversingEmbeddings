---
name: experiment-recap-doc
description: Report experiment results back as a plain-English recap doc (a published Artifact page) that takes each of the user's own questions and ideas in order, restates their hope, then says what was done, what happened, and why.
---

# Experiment recap doc

Use this whenever more than one experiment or what-if needs reporting, when the user says
they did not understand a result, or when they ask for a doc or a recap. The full
instructions and the page template are in `.claude/skills/experiment-recap-doc/`
(`SKILL.md` and `template.html`). This file is the summary.

The reader stepped away 30 minutes ago, remembers the gist but none of the specifics, and
had no hand in the plan or the code. The doc exists so they understand what happened
well enough to propose the next experiment.

- **Language:** no code names (function names, dataset or sheet ids, file names), no
  coined terms, and no local paths. Describe test cases by what they look like. Define
  the one shared term in the setup section. Every number sits next to its plain
  sentence.
- **Order:**
  1. A reminder of the setup.
  2. A contents list of the user's questions, in the order they raised them.
  3. One section per question, each with: *You said* (a verbatim quote plus their
     hope), *What I did*, *What happened* (a one-line yes / no / partly verdict, then
     bullets with numbers), and *Why* (one or two causal sentences).
  4. A summary table.
  5. What this points to.
- **Honesty:** untested ideas keep a "Not tested yet" section. Your own additions go in
  a separate, labelled block. Corrections to the user's reasoning say "you were right
  that X, but Y".
- **Figures:** relabel them without code names, and embed them as `data:` URIs inside
  the section they support. Keep them to two or three big panels, and apply
  `skills/experiment-figures/SKILL.md`.
- **Delivery:** before publishing, grep the HTML for code names and `/home/`. Publish
  with the Artifact tool, then reply in chat with the link and the section titles only.
