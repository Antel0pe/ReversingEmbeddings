---
name: experiment-recap-doc
description: Write up experiment results as a plain-English recap doc (a published Artifact page) that walks through each of the user's own questions and ideas in order. For each one it restates their hope, says what was done, then what actually happened and why. Use whenever more than one experiment, or a chain of what-if questions, needs reporting back; whenever the user says they did not understand the results; or when they ask for "a doc", "a claude doc", "a recap", "write it up" or "explain what happened". Built for a reader who stepped away for half an hour, remembers the gist but none of the specifics, and had no hand in the plan or the code.
---

# Experiment recap doc

The user runs a loop. They have an intuition about a problem, propose a method, you run
it, and they use what happened to propose a better experiment. **The loop only works if
they correctly and easily understand what happened.** A terminal summary full of numbers,
code names and terms coined mid-experiment breaks the loop, however accurate it is. This
doc is the fix. The user praised the first one ("it was very well done"); this skill
records what made it work.

## The reader

Write for this person and no one else:

- They spoke about 30 minutes ago. They remember roughly what the topic was, but not
  the specifics: not the numbers, not which variant was which, not what they themselves
  guessed.
- They had **no hand in the plan or the implementation**. Every method, filter or
  setting you chose has to be explained from scratch, at a high level.
- They are curious and sharp, with little formal maths. Their intuitions are often right
  before they can formalise them; say so when they are.
- Their messages are long, spoken rambles holding several questions, guesses and
  half-ideas at once. Each of those is a thread they want closed.

## Hard language rules

1. **No code names.** No function names, sheet or dataset identifiers, file names,
   variable names or method registry names (`s_curl`, `check_k10`, `isomap_k10`,
   `K40_g0.5`), unless the user themselves used the name confidently. Describe things
   by what they look like or do: "a sheet rolled so tight its layers are closer
   together than neighbouring dots", "an older filter that throws out connections
   pointing out of the paper's surface".
2. **No coined terms.** If one term is truly needed for the whole doc, such as "wrong
   connection", define it once in the setup section in ordinary words and use only
   that. Never introduce a new noun in a result that was not defined earlier.
   Metaphors are fine (strings, staples) if they are explained when they first appear.
3. **Use the user's own words** for their ideas: their phrasing, their analogies, their
   terms ("1 to 1 mapping", "manifold", "pull apart").
4. **Every number sits next to the plain sentence it supports.** For example: "12 wrong
   connections out of about 42,000 left 35% of the paper folded".
5. **No local file paths** anywhere in the page (repo rule).
6. Maths: if any appears, write it as plain words or Unicode. No LaTeX.

## Structure (in this order)

1. **Title and one-line lede.** The title is a name, not a sentence. The lede says the doc
   recaps the recent messages, one section per question or idea.
2. **"First, a reminder of the setup."** Give enough to make the rest readable:
   - what the test bench or object is;
   - what a method is given and what it must produce;
   - how a result is judged, and what "success" means in plain words;
   - what the failure looks like;
   - one plain name and description for every test case that comes up later;
   - where things stood before this round, as numbers.
3. **"What's in this doc."** A list with one line per user question or idea, **in the
   order they raised them**, each labelled "Your idea", "Your question" or "Your
   objection".
4. **One section per question or idea**, numbered in the same order. Each section uses
   the same labelled blocks:
   - **You said:** a short verbatim quote of the user (typos left in), then one or two
     lines restating their reasoning, their guess, or what they hoped would happen.
   - **What I did:** the high-level procedure in plain English. Mention a setting only if
     the result depends on it. Include any control, such as a "cheat" given the true
     answer, and say *why* it was run ("to ask whether pulling itself works when the
     strings are right").
   - **What happened:** a single-line verdict first, coloured **yes / no / partly**. Then
     bullets, each one claim with its number.
   - **Why:** one or two sentences giving the mechanism in plain words. This is the
     sentence the user builds their next idea on, so make it concrete and causal.

   Use sub-headings (1A, 1B…) when one idea was tested in several ways. Also:
   - Corrections to their reasoning are specific and kind: "you were right that X, but Y."
   - A conceptual question answered from existing results gets "What I found (from
     existing results, no new experiment)".
   - **An idea that was not tested still gets its own section** marked "Not tested
     yet", with any related past attempt. Never drop a thread silently.
   - **Your own additions go in a visually separate "My own extra check (not your idea)"
     block.** Never blend them into the user's result.
5. **Figures**, embedded inside the section they support. Before embedding, relabel them
   so they contain no code names. Each caption says what one mark is, what each colour
   means, which panel is the baseline, and the one thing to look at. Follow
   `.claude/skills/experiment-figures/SKILL.md` and keep the user's limit: two or three
   big panels, not six.
6. **"Everything in one table."** Columns: Idea (mark whose it is) | Hope | What happened
   | Why, in a sentence.
7. **"What this points to."** Three or four bullets of information the user can use
   to propose the next experiment: patterns across failures, what turned out not to
   matter, and threads still open. This is input for them to decide with, not a plan.

## Making the page

1. Call the Artifact tool with `action: "quickstart"`, `intent: "document"`. If no Docs
   type is available it says so; then call it with `intent: "other"` and make a plain page.
2. Copy `template.html` (next to this file) into the scratchpad and fill it in. It
   already contains the style: tokens for light and dark, labelled blocks, verdict
   colours, a separate block for your additions, scrolling figures and tables. Keep its
   look so the docs read as one series.
3. Embed each figure as a base64 `data:` URI by swapping the placeholders with a short
   Python snippet. The page cannot load local files.
4. Before publishing, grep the final HTML for code names you used during the work
   (sheet ids, method names, `_k10`, `.py`) and for `/home/`. The count must be 0.
5. Publish with `icon: "document"` and a one-sentence `description`.
6. Reply in the chat with the link and one line per section title. Don't repeat the
   results in the chat; the doc is the answer.

## Final read-through

Read the page as the reader above. For each section ask:

- Can I tell which of my messages this answers, and what I was hoping for?
- Can I tell what was actually done without knowing the code?
- Is the verdict obvious in one line?
- Does the "why" let me guess what to try next?
- Is there any word I'd have to ask about? If so, define it or remove it.
