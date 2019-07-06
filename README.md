# Neural Style Transfer with Pytorch
## Introduction
Implementation of a neural network that can transfer the style of an arbitrary image to another photo.
<br />
Much of the code (e.g the layers) is implemented in my [neural network toolbox](https://github.com/nhatsmrt/nn-toolbox/blob/experimental/nntoolbox/). The training procedure can be found [here](https://github.com/nhatsmrt/nn-toolbox/blob/experimental/nntoolbox/vision/learner/style.py). This repository contains only the testing code.
## Issues
The biggest issue I still have to deal with is checkerboard artifacts (see results section)
## Some results
### Styling Bus:
<img src="demo/content.png" alt="content" width="250" />
<img src="demo/style.png" alt="style" width="250" />
<img src="demo/styled.png" alt="styled" width="250" />
<!-- ![Styled](demo/styled.png) -->
