I want a method that discovers a set of controls that describes the entire generated-one family. The controls do not have to correspond to center, height, width, or lean. They need to generate valid images, reach every image in the family, and have a defined meaning when used together.

The important realization is that a knob is more than a line or curve through a starting image. Its effect depends on the current state. A complete description of a knob must therefore explain both how it changes an image and how that change varies as the other knobs change.

Choosing five curves through a root image gives examples of individual changes. It does not define their combinations. What is missing is a shared rule that extends those examples throughout the space.

PCA gives useful linear directions and measures of variance, but its directions need not generate valid images. Nonlinear embeddings can reveal structure, but coordinates alone do not tell us how to reconstruct an image. An autoencoder does provide coordinates and a decoder, although accurate reconstruction does not automatically mean that every decoded image belongs to the valid family.

The speckle problem exposes that distinction: an image can be numerically close to a valid stroke while violating the rules defining a valid stroke.

The questions I want to investigate are:

- What conditions make a set of controls a complete description of the family?
- What mathematical objects describe directions that change with state?
- How can those changing directions be represented compactly?
- How can we test whether the controls combine consistently?
- Can we discover useful controls through geometry, graphs, or explicit rules?
- Could fewer controls describe the family, and what would “fewer” actually mean?

Because we have a renderer, we can generate targeted examples and test proposed rules against the actual family. The opportunity is to discover a compact explanation of its structure.


Requirements
Validity: Every allowed setting produces a member of the generated-one family.
Coverage: Every image in the family can be produced by some allowed setting.
Editing: Changing a setting has defined behavior throughout its allowed range.
Reconstruction: Given an image, we can find settings that reproduce it.
Compactness: The decoder and allowed domain have a manageable description.
Uniqueness: each image has one coordinate address.

