results.png              -- train vs val total loss per epoch. Both should trend down;
                             val flattening/rising while train keeps falling = overfitting.
results_components.png   -- the four loss terms that sum to the training total: box (CIoU,
                             box shape/position), cls (BCE, is-there-a-cube-here confidence),
                             dfl (distribution focal loss, box regression precision), kpt
                             (OKS-style distance + visibility BCE for the 8 corners).
confidence_hist.png      -- distribution of the model's confidence score on val images.
                             Skewed toward 1.0 with few low-confidence predictions is what a
                             well-trained model looks like.
iou_hist.png             -- distribution of predicted-vs-ground-truth box IoU on val images.
                             Skewed toward 1.0 is good; a lot of mass near 0 means the model
                             often isn't finding the cube at all.
pr_f1_curve.png          -- precision/recall/F1 as the confidence threshold used to accept a
                             detection is swept from 0 to 1. Shows whether the model's own
                             confidence score is trustworthy: ideally, correct detections get
                             high confidence and wrong ones get low confidence, so precision
                             stays high even as you raise the threshold, while recall only
                             drops once you cut into genuinely-correct detections.
keypoint_error.png       -- mean pixel error (at the model's input resolution) for each of
                             the 8 corners individually, only over instances where that
                             corner was labeled visible. Flags whether specific corners
                             (e.g. consistently-occluded back-face ones) are weaker than
                             others.
val_predictions.jpg      -- a contact sheet of a few val images with ground truth (green) vs
                             predicted (yellow) box + 8 keypoints drawn together, plus the
                             confidence/IoU for that image (green text = correct, i.e. IoU >=
                             0.5; red = not) -- the fastest way to eyeball what's actually
                             going wrong instead of reading numbers.
