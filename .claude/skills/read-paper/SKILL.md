---
name: read-paper
description: Read a research paper together with the user, at a pace they set. Use whenever the user pastes, uploads, or names a paper and wants to understand it, mine it for ideas, or take notes on it. Creates one note file per paper in literature/, delivers a plain-English overview first, then deepens level by level on request, and mines the paper for anything bearing on recovering a low-dimensional parameterisation of a high-dimensional dataset. Triggers: "read this paper", "here's a paper", "explain this paper", "what's useful in this", "add this to literature", a pasted abstract, an arXiv or DOI link.
---

# Read a paper with the user

The user is curiosity-driven, has a strong intuitive grasp of their own
experiments, and assumes **high-school mathematics**: arithmetic, algebra,
geometry, a little trigonometry, the idea of a slope. They have not had
calculus, linear algebra, functional analysis, or measure theory as courses.
They can absorb any of it when it is built from an object they already know.

Nothing in this skill licenses vagueness. Plain English means the *real* idea
said in words, not a watered-down idea. If a concept cannot be reached from
high-school footing in one message, build the ladder across several.

## Never do these

- **Never download the paper into the repository.** The note file holds a link
  and the user's notes, never the paper's text in bulk. Short quotes for
  evidence are fine; a mirrored PDF or a full-text dump is not.
- **Never put an absolute local filesystem path in the note file.** Repo-relative
  paths only.
- **Never open at Level 3.** The first message is always Level 1, and it stops
  there. The user decides when to go deeper.
- **Never invent a citation, a number, a section, or a result.** If the paper
  does not say it, say that it does not say it.
- **Never let the paper's framing stand as fact.** A 2005 conference paper is
  an argument, not a settled truth. Flag what is proved, what is shown on one
  example, and what is asserted.

## Step 1 — Create the note file, before explaining anything

Filename: the paper's title, lowercased, kebab-case, `.md`, in `literature/`.
Example: `literature/multiscale-structure-of-non-differentiable-image-manifolds.md`.

Write it from this template, filling only what the source actually supports.
Leave a field as `unknown` rather than guessing:

```markdown
# <Full Paper Title>

<canonical link>

<Authors> · <Venue>, <Year> · <DOI or arXiv id, if known>

**Read with:** <date> · **Depth reached:** L1

---

## What it says (my notes)

<!-- The user's own words go here. Claude does not write in this section
     unless asked; it is theirs. -->

---

## Claude's running summary

### L1 — the gist

### Relevance to this repo

### Open questions

### Claims to check

---

## Links

- Related memory / repo notes: [[...]]
```

Add a one-line pointer to `literature/README.md` (`- [Title](file.md) — hook`),
creating that index if it is missing.

Then tell the user the file exists and move straight to Level 1. Do not ask
permission to start reading.

## Step 2 — The depth ladder

Deliver **one rung per message**. End each rung by naming what the next rung
would add, in one sentence, and stop. The user climbs by asking; they do not
have to use the word "level".

**L1 — The gist.** 200-350 words. Zero notation; no Greek letters, no symbols.
Four things, in this order:

1. The question the paper is asking, stated as a question.
2. The answer, stated as a sentence a non-specialist would understand.
3. The one surprising thing — the finding that makes the paper worth reading
   rather than skimming.
4. Why it lands on this repo's desk. One or two sentences.

A reader should be able to repeat L1 to someone else from memory.

**L2 — The argument.** The skeleton: the paper's claims in the order it makes
them, each with the kind of evidence behind it. Label the evidence honestly as
*proved*, *derived under assumptions*, *computed on one example*, or *asserted*.
Still words and arithmetic. Section numbers on every claim.

**L3 — The machinery.** Now the actual mathematical objects. For each one:

- name the familiar object it generalises
- say what it is in words before any symbol appears
- then give the symbol, and say what each part of it means
- give the concrete case from this repo, with real numbers

Use Unicode inline (θ, ε, ‖·‖, →, ∂, ≈, ½) and never LaTeX in backticks.
Display math only in `$$` blocks, which are the only ones that render.

**L4 — The derivation.** Walk the one derivation the paper's result actually
hangs on, line by line, saying in words what each line does to the line above.
Plug in a number at the end and check the result arithmetically. If a step
needs machinery beyond the ladder, say so plainly and give the *shape* of the
step and what it buys, rather than hand-waving through it.

**L5 — Run it here.** Reproduce the paper's claim on this repo's own objects:
`grey_ones.render` for the edge-sharp family, `smooth_ones` for the regularised
one, `coverage_grids` for grid fineness. Write a script, run it, report numbers
against the paper's prediction, and say plainly whether it held. A claim that
fails to reproduce is a better note than one that was never tested.

## Step 3 — When the user is confused

Confusion is information about which rung broke, not a request to repeat.

1. **Find the break.** Ask which sentence stopped making sense, or infer it
   from the question. Do not re-explain the whole thing.
2. **Do not paraphrase the same explanation.** Change the object. Drop to the
   smallest case that still shows the phenomenon: one pixel instead of 784,
   two dimensions instead of a million, a line instead of a manifold.
3. **Put numbers in it.** An abstract claim with three worked numbers beside it
   stops being abstract.
4. **Re-climb explicitly.** "That was the small version. Here is the same
   sentence with the paper's words restored."
5. **Then check.** Ask one question whose answer reveals whether the idea
   landed, and that cannot be answered by repeating the words back.

If the same idea breaks twice, the prerequisite is missing. Name the
prerequisite, teach it in its own right from a familiar object, and only then
return to the paper.

## Step 4 — Mine it for this repo's question

Every paper gets read against one standing question:

> Given a high-dimensional dataset that was generated by a low-dimensional set
> of knobs, **recover the map from the knobs to the data** — in as few
> coordinates as possible, by construction, measurement, or closed form, not by
> training a network and declaring victory.

So: dimension estimation alone is not the target. Autoencoders, VAEs, generic
"fit a model to the data" answers are explicitly *not* what is wanted — but a
paper that uses one is still worth mining for its geometry.

For each paper, answer these, and say "nothing here" when that is the answer:

- **A method to steal.** Something that could run on `grey_ones` by next week.
- **A diagnostic to steal.** A measurable quantity that would tell this repo
  something it does not currently measure.
- **A warning.** An obstruction the paper proves or demonstrates that this
  repo's methods would walk into.
- **A confirmation.** Something this repo found independently that the paper
  also found — note both sides, since independent agreement is evidence.
- **A contradiction.** Where the paper's result and this repo's result do not
  fit together. These are the most valuable entries. Do not smooth them over.
- **A named concept.** The term of art for something the repo has been
  describing in its own words. This unlocks the literature.

Cross-link findings to the repo's own files by relative path, and to memories
with `[[slug]]`.

## Step 5 — Honesty markings

Inside the note file and in messages, keep these distinct:

- **[paper]** — the paper states this.
- **[paper, one example]** — shown on a single model or figure, not proved.
- **[mine]** — Claude's inference, extrapolation, or connection to the repo.
- **[check]** — plausible but unverified; belongs under "Claims to check".

Give a section number for every **[paper]** claim. If the source is a pasted
excerpt rather than the full paper, say which parts were not seen, and never
summarise a section that was not in the paste.

## Step 6 — Keep the file current

After each rung, append to the note file: the new level's summary, any new
relevance entries, any question raised, any claim to check. Update
**Depth reached**. Never overwrite the user's own section. The file should end
the session being worth reading cold in six months.
