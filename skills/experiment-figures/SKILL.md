---
name: experiment-figures
description: Create explanatory figures for this repository's image and manifold experiments, especially when a reader needs to understand an operation or judge a result from the figure itself.
---

# Explain an experiment with figures

Start from the object the reader knows. For a pixel-space experiment, show the source image and define what one pixel value means before plotting a vector, angle, event, or manifold coordinate. Distinguish the controlled generated 1 family from real MNIST.

Give each figure one question and a visible reading order. Introduce the baseline before variations. Define every axis, color, mark, parameter, and unit in plain language on the figure. State whether the view shows the whole object, a fixed-parameter slice, a sampled result, or a projection. Put the measured result next to the visual evidence, with any relevant comparison or threshold.

For pixel changes, use a diverging palette centered at zero and say which sign gains ink. For changes *to a direction vector*, label that second difference separately: its colors describe a change in the pixel-change pattern, not direct ink gain or loss. Keep comparable panels on a common color scale, or state clearly when scales differ. Mark values hidden by clipping, occlusion, or display resolution.

Reserve space for titles, panels, legends, and captions. Render the final file, inspect it at the size the user will see, and correct overlaps or illegible labels. Show the final figure inline using the available chat or artifact mechanism; do not copy Claude-only commands such as `SendUserFile` into a Codex workflow.

## Understanding check before delivery

Read the figure as someone who has not seen the code or the conversation. Ask:

1. What physical or mathematical object am I looking at, and what does one plotted mark represent?
2. What do both axes, colors, and numbers mean? Can I decode them without guessing?
3. Where is the starting case, and which single operation changes it?
4. What result does the figure show, and which visible evidence supports that result?
5. What is fixed, hidden, sampled, or projected? Could the title be read as a broader claim than the evidence supports?

If any answer is missing or requires a paragraph outside the figure, revise the figure first. Then explain it in the message from familiar object to new result, one inference at a time.

For detailed layout and experimental-figure conventions, the source guidance is `.claude/skills/experiment-figures/SKILL.md`; use its applicable parts without importing Claude-specific tool calls or treating every suggested panel type as mandatory.
