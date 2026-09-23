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

the space of 1s is probably smooth because the knobs of how to draw a 1 continuously change. the act of drawing a 1 is always smooth?
- possibly an interesting shape in reality. the undersampled graph we have can make even a simple shape seem complex
    - something about sampling areas that would reduce dimension of graph?
- another interesting question is the area of all numbers. things like 6 and 9 are very close. but also what space does the entire order of numbers take up. what "numbers are possible"? 
- is there any way to mine numbers in the past or make better guesses about what they might have looked like? 

an equation that defines what image shows a 1 and what isnt
- sum of various checks where each check is something interpretable like no disconnected segments
    - each filter further reduces the state space
    - is there someway to  uniquely prove/find the fewest number of terms that uniquely define a chunk of state space
        - tradeoff would likely be explainability at the cost of efficiency?
- a requirement for this might be a smooth state space first? likely cant do it with a discontinuous sample of data. 
    - actually you probably could but there would be some OR check and less clean

if you draw a circle on a paper, it actually varies in 2 dimensions but can be explained in 1 dimension: a line
- this is a very strange idea but seemingly common place
- i wonder if there is something here

some random goals that might be worth having
- see the space of 1s in mnist. ideally in actually 784d but how tf would that be possible. even in terms of ~10 intrinsic dimensions would be cool but how. it's not about seeing how the intrinsic dimensions behave but where they are. 
- mathematically define what a 1 is. wouldnt this just be an equation of the intrinsic dimensions?
    - a list of pixels with coefficients does not count. needs to have something understandable about it

really want to be able to describe the unique manifold of 1s from my generator that this is the state space this generator carves out

intrinsic dimensions compress a lot of individual axes together
- leaning 1 does not move individual pixels but identifies multiple pixels moving in tandem as 1 movement
- leaning as an intrinsic dimension is curved because no single pixel linearly increases or decreases. depending on where the 1 is CURRENTLY is what affects where it leans. this means that sometimes the 1 will affect a changing combination of linear dimensions