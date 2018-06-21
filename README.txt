David Li
dzli@jhu.edu
Computer Vision HW 3

Enclosed files:
    Images: bceloss_aug.png
            bceloss.png
            contrastive_aug1.png
            contrastive_aug2.png
            contrastive.png
    Python files:
            augmentation.py
            ContrastiveLoss.py
            LFWDataset.py
            SiameseNet.py
            p1a.py
            p1b.py
    Test files:
            test.txt
            train.txt
    Dataset download script:
            get_data.sh

All files should be in top level directory, along with the image data folder lfw/.            

usage: p1a.py [-h] [--aug] (--load LOAD | --save SAVE)

Process loading or saving.

optional arguments:
  -h, --help   show this help message and exit
  --aug        Toggle data augmentation
  --load LOAD  File from which to load model
  --save SAVE  File to save model to

usage: p1b.py [-h] [--aug] [--epochs EPOCHS] [--batchsize BATCHSIZE]
              [--margin MARGIN] [--threshold THRESHOLD]
              (--load LOAD | --save SAVE)

Process loading or saving.

optional arguments:
  -h, --help            show this help message and exit
  --aug, -a             toggle data augmentation
  --epochs EPOCHS, -e EPOCHS
                        training epochs
  --batchsize BATCHSIZE, -b BATCHSIZE
                        training batch size
  --margin MARGIN, -m MARGIN
                        Set custom margin (default 10
  --threshold THRESHOLD, -t THRESHOLD
                        Set custom threshold
  --load LOAD           File from which to load model
  --save SAVE           File to save model to


Model weights link: https://drive.google.com/open?id=1KNwRnfaCcdFLN7hd97wflvBBGB8srl6Z
Non-augmented weight files: p1a.w, p1b.w
Augmented weight files:     p1a_aug.w, p1b_aug.w


Note:
    I used https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch as a jumping off point
    to learn PyTorch and understand Siamese Networks, so some of the code may seem similar. Although it was very useful,
    our project is very different from the example, so all of the code submitted was written by myself.

Q1a. (without augmentation)

I used a learning rate of 1e-6.
For this part, I used 30 epochs -- the loss stabilizes within 30 epochs. A graph of my loss can be seen in bceloss.png.
Testing results:
    Loaded model: p1a.w
    Accuracy on train set: 1.00
    Accuracy on test set: 0.55

Q1a.i. (with augmentation)

With augmentation, I decided to allow each transform a 80% chance to occur (independently). Augmentations
were done using skimage library functions.

This time, since loss took longer to stabilize (due to the random augmentation), I used 100 training epochs. 
A graph of my loss can be seen in bceloss_aug.png.
Testing results:
    Loaded model: p1a_aug.w
    Accuracy on train set: 1.00 
    Accuracy on test set: 0.54

Q1a.ii.
How well did part a work on the training data? On the test data? 

    The model appears to do very well when we test with training data but is basically indistinguishable
    from guessing on test data. It scores very close to 100% accuracy on training data but close to 50%
    on test data. This makes me think it's overfitting and just memorizing the training
    data instead of actually learning structural similarities in the faces. 
    
    Even when I saved tested models with less training epochs, I still had a big overfitting issue.

Any ideas why or why not it worked well or didn't? Answer this question below in part c.

    In some of the training examples I saw, there are actually multiple faces in the image, and 
    the face may show up anywhere in the image (on the side, in the middle, etc). If we cleaned up the data
    a little and cropped to a box around the face, it might be easier to actually learn facial features
    and increase our accuracy. 



Q1b. (without augmentation)

To speed up training time, I used a learning rate of 5e-5, compared to 1e-6 in part a. I also used a margin of 
2.0 during training and a threshold of 10.0 during testing.

My results from a 30 epoch training period is as follows. The loss seemed to stabilize around 10 epochs of training,
but the accuracy on the test sets was not as high. At 10 epochs, the accuracy on train and test was 0.94 and 0.53
respectively, but at 30 epochs, the train and test accuracy was 0.94 and 0.58.

    Testing results:
        Loaded model: p1b.w
        Accuracy on train set: 0.94
        Accuracy on test set: 0.58

Q1b.i. (with augmentation)

I used the same augmentation as in part a. To speed up training time, I used a learning rate of 5e-5, 
compared to 1e-6 in part a. I also used a margin of 2.0 during training and a threshold of 10.0 during testing.

After 30 epochs, the loss wasn't too stable, but was pretty low. The loss stabilized closer to 40 epochs, and
the loss graph is reflected in the two images contrastive_aug1.png and contrastive_aug2.png (I wasn't able
to generate a loss graph all at once).

The following results are from the model after 30 epochs of training.

    Testing results:
        Loaded model: p1b_aug.w
        Accuracy on train set: 0.74
        Accuracy on test set: 0.67
        
After 10 more epochs (40 total), I had the following results.
    Testing results:
        Loaded model: p1b_aug.w
        Accuracy on train set: 0.77
        Accuracy on test set: 0.67        
        
Q1b.ii. How did contrastive loss do on the training data? Testing data?

Contrastive loss did quite well (better than BCELoss) when testing on both the training and test splits.
In particular, in the augmented version, I achieved almost 67% accuracy on the test set with similar accuracy on
the training set, indicating that we probably were not overfitting heavily as we were in part a.

Q1c.
 
In my experience, the Contrastive Loss model performed better than the BCELoss model, since it had higher
train set accuracy and test set accuracy in both the non-augmented and augmented models. In fact, the non-augmented
contrastive loss model had better test accuracy than the augmented BCELoss model. 
The BCELoss model seemed to overfit quite heavily and give us a very high training accuracy with close to
random guessing on the testing set. As mentioned in my answer to part a, the dataset wasn't very clean
since there were images with multiple people and the faces were not always centered. For instance,
in http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf, the example images are all centered.
If we cleaned up the data a little and cropped to a box around the face, I suspect it would be easier to actually 
learn facial features and increase our accuracy. Also, if we had a larger dataset, I think we would
see better results since it's basically always good to have more training data.
