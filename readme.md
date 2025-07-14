# GAN Colorizer

**This is a Colorizer that can colorize your gray-scale images. My goal was to restore the history photos' color, so that everyone can recreate their memories from the past.**

![Kaohsiung, Taiwan in 1948](Kaohsiung%20Taiwan%20in%201948.jpg)
*Kaohsiung, Taiwan in 1948 [Source](https://www.facebook.com/photo.php?fbid=10151512843949531&id=124164094530&set=a.10151549550209531)*

![18s ~ 19s Taiwanese](18s%20~%2019s%20Taiwan%20Aboriginals.jpg)
*18s ~ 19s Taiwan Aboriginals [Source](https://www.reddit.com/r/TheWayWeWere/comments/192diuh/taiwan_late_1800s_and_early_1900s_by_ryuzo_torii/)*

## Walk through in short
> [!IMPORTANT]
> Need to shout out to [DeOldify](https://github.com/jantic/DeOldify) which gave me a lot of thoughts for building it, also a big credit to this [tutorial](https://www.kaggle.com/code/varunnagpalspyz/pix2pix-is-all-you-need) from Varun Nagpal Spyz.

I used Conditional Generative Adversarial Network, also known as cGAN, as the main sturcture. Specifically, it is the [Pix2Pix](https://github.com/phillipi/pix2pix) structure.

The dataset that I used for training was COCO2017 training set, and I only took 25k images to train. I referred to some discussions that were talking about using Progressive Resizing technique that could make the model generalize to most scenes. 
I converted the training set to CIE-LAB color space, referring to the tutorial which I mentioned above.Use CIE-LAB is a more efficient way in my opinion, because compared to RGB, LAB's brightness channel is separated from color channels, so for the model there are fewer parameters needed to predict, and also LAB's color channels are closer to human eyes perception.

Pix2Pix uses Unet as generator, I didn't change it but I used FastAI's DynamicUnet to replace it, also did some modifications and experiments, more details are in the [Specification](#specification) section. The discriminator that I used was suggested by AI.

The loss weights ratio was hard to tune. Basically, you need to keep trying and inferencing to see whether the results are good or not. It will affect the results a lot, you might be careful with the parameters.

## Specification

## TODO

- [ ] Release the web inference interface
- [ ] Release the model
- [ ] Release the docker image
- [ ] Finish the readme
      

WIP!!!
