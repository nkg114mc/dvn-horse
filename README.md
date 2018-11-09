# DVN-Horse

In this project, we try to replicate the experimental result of paper "Deep Value Networks Learn to Evaluate and Iteratively Refine Structured Outputs" (Michael Gygli, Mohammad Norouzi, Anelia Angelova. ICML 2017) on the 32x32 Weizmann horse dataset.

### Requirements

The current implementation is coded with python 3.5. The following packages are required:
+ tensorflow
+ sklearn
+ skimage
+ imageio

### Run

After download or clone the repository, you need to unzip the data folder `pics32.zip` first. To run the system, simply use the command below:

	python horsedvn2.py

During training, the system will do test after each echo, and dump the prediction images in folder `horse_outpics`.


### Performance and Issues

| Systems | Mean IOU |
| ------:| -----------:|
| This system | 68.8%    |
| Paper reported | 84.1% |

Remaining unclear details about the experiments:

1. About image croppings, I only did a random cutting (random pick upper-left point to get a 24x24 sub-image) over the original image, but no mirroring. Is the mirroring very important for achieving the good performance?
2. About the gradient based inference:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.1) What step size should be applied? I notice that your work use 4, but I have to use 50 since 4 is too small for my system (and the inference will always end up with “all-zero” prediction)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.2) How many steps is needed? I assume that the number of step is 30 (from the paper slides), just want to double confirm.
3. About training:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.1) What optimization algorithm is used? SGD or Adam?
I am using GradientDescent optimizer now. I once changed the optimizer from SGD to Adam, but come across a issue that the inference result is very small then turns to all-zero output after binarization.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.2) What’s the required number of epochs that is needed to get the result in the paper?
I use 300, but may be it is not enough?

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.3) What mini-batch shold be used? I am using 20. Note that the mini-batch size is critical to the example generation.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.4) The most confusing part: how is the training example generated?




---

* Author: [Chao Ma](http://people.oregonstate.edu/~machao)
* Email: machao@engr.orst.edu
