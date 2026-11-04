---
name: read-paper
description: Read a research paper with the user at the pace they set, keep one note file per paper in literature/, and mine each paper for ways to recover a low-dimensional parameterisation of a high-dimensional dataset.
---

# Read a paper with the user

Assume high-school mathematics and no calculus or linear algebra course, but
never water the idea down — build a ladder to the real one instead.

Create `literature/<kebab-title>.md` first, with the paper's title, a link, and
the bibliographic line. Never mirror the paper's text into the repo and never
write an absolute local path into the file. The "What it says (my notes)"
section belongs to the user.

Then deliver **Level 1 only**: the question, the answer, the one surprising
thing, and why it matters here, in plain English with no notation. Stop, say
what the next rung adds, and let the user ask. Deepen through L2 the argument
with evidence labelled honestly, L3 the machinery built from familiar objects,
L4 the one derivation that carries the result, and L5 reproducing the claim on
`grey_ones` / `smooth_ones` and reporting whether it held.

When the user is confused, find the broken sentence, change the object rather
than the wording, shrink to the smallest case that still shows the effect, put
numbers in it, then restore the paper's language.

Read every paper against the standing question: recover the map from a few
knobs to high-dimensional data, in as few coordinates as possible, by
construction or closed form rather than by training a network. Record a method
to steal, a diagnostic to steal, a warning, a confirmation, a contradiction,
and a named concept — "nothing here" is a valid answer. Mark every claim
**[paper]**, **[paper, one example]**, **[mine]**, or **[check]**, with a
section number for anything attributed to the paper.

For the full note template, the ladder in detail, the confusion protocol, and
the mining checklist, read `.claude/skills/read-paper/SKILL.md`, which is also
loaded automatically as the `read-paper` skill.
