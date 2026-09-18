centroid/mean of some subset as a way of finding if items there are on manifold
- finding "holes"


if you were to take two 1s that were say 1 unit away from each other and averaged the pixels, high chance that the result would be a 1 that is a "new" 1 that doesnt exist in the dataset yet. kinda cool but doesnt really tell you anything
- if you were instead to take 1s that were say 5 units away with no path between them meaning the pixel averages dont give you a 1 and there is no path of 1s that are 1.89 units away to get from one to the other - if you "discovered" a path between them that would be a really interesting result because the 1s are new but theyre possibly a class of 1s that dont exist in the dataset in the first place
    - unlikely that these would genuinely improve classification of 1s - think this is basically a 100% solved problem but in terms of the data manifold that would be ridiculously interesting

something to make the points denser
- proactively surfaces cases like is this a 1. enables finding of new 1 areas and making existing areas denser?

something about getting closer to the data and seeing how it really lives
- showing where it is locally smooth
- showing where some attributes are similar but additional parameters?

the graph of mnist 1s is a graph - meaning it a connection of 1d lines between data points
- we dont need methods that unfurl a donut or do anything to a torus like structure in higher dimensions
- we just need a collection of lines that vary in higher dimensions to be visible
- there are some cycles though which could vary from some simple loops in graph to tunnel like structures in higher dimensions

test of whether data exists potentially if you can morph pixels from starting to ending point?
- except for those with disconnected components on the image

look at negative area of the state space
- areas with high speckle across the image
- disconnected components 
    - there are real 1s in the dataset with disconnected components but honestly i think those shouldnt count so we can eliminate these
- can ignore majority of space that clearly isnt 1 using some heuristics around distribution of intensity on the image, disconnected components

can experiment with core of 1 space
- straight 1s taking say 80% of space in straight lines
    - moving sideways/vertically