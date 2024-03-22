# Changes that were made to nnUNet as described in ...

To train on two different sets of labels simultaneously ('whole tumor' or 'enhancing tumor', 'tumor core', and 'edema'), several changes to nnU-Net were made. First, the loss function was modified such that it only propagates the loss for the parts of the label for which it has the segmentation. Introducing the change to the loss function required us to modify the online evaluation method that nnU-Net uses to store the best-performing iteration of the model. For simplicity, we removed online evaluation, causing nnU-Net to fall back on the loss of the validation set to determine the best iteration. 

All changes are indicated by a comment stating 'Sander edit'.

The code is dependend on the names of our datafiles (niftis) to distinguish between BraTS data and our own data.

Code was tested with nnunet version 1.6.6
