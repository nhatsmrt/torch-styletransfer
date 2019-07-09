# Neural Style Transfer with Pytorch
## Introduction
Implementation of a neural network that can transfer the style of an arbitrary image to another photo.
<br />
Much of the code (e.g the layers) is implemented in my [neural network toolbox](https://github.com/nhatsmrt/nn-toolbox/blob/experimental/). The training procedure can be found [here](https://github.com/nhatsmrt/nn-toolbox/blob/experimental/nntoolbox/vision/learner/style.py). This repository contains only the testing code. To replicate my work, please also clone the experimental branch of my nntoolbox repository.
## Issues
It seems to work pretty well on resized COCO data (even on untrained/unseen photos), but does not generalize that well too random photos. I suspect the problems lie in resolution discrepancy. Or maybe I just haven't trained for long enough (although in my experiment the quality of the images seem to degrade as I trained more and more).
## Some results
I train the thing for 850 iterations, using COCO dataset (resize to 256 for each size), and the train_9 subset of the wikiart dataset. For each dataset, I split 80% of them as training data and use the rest for evaluating. I train the network for a total of 850 iterations (1 "epoch"). This is the result:
### Kinda Successful:
<img src="demo/PixelShuffle/content_3.png" alt="content" width="250" /> <img src="demo/PixelShuffle/style_3.png" alt="style" width="250" />
<img src="demo/PixelShuffle/styled_3.png" alt="styled" width="250" />
<br />
<img src="demo/PixelShuffle/content_1.png" alt="content" width="250" />
<img src="demo/PixelShuffle/style_1.png" alt="style" width="250" />
<img src="demo/PixelShuffle/styled_1.png" alt="styled" width="250" />
### Less So:
<img src="demo/PixelShuffle/less_successful.png" alt="styled" width="750" />
