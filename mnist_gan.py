import logging
import numpy as np
from datetime import datetime, timedelta


class GAN:
    def __init__(self):
        pass

    def discriminator(self):
        # output: sigmoid scalar (likelihood)
        # no pooling (why?) --> strided conv for downsampling
        # leaky reLu activation (why?)
        # dropout .4-.7 against overfitting
        # loss = binary cross entropy (why? bcs sigmoid? vanishing gradient? as opposed to?)
        # optimizer RMSprop as opposed to adam?
        # weight decay? clip rate?
        # decay ~ learning rate
        # output - feedback?
        pass

    def generator(self):
        # output dim == discriminator input dim
        # generated from 100-dim random normdist (why?)
        # upsampling between first 3 layers (why?) - does that introduce "fuzzyness"?
        # batch normalization between layers
        # relu activation (why not leaky?)
        # dropout .3-.5 at first layer against overfitting (overfitting what, exactly? "learning" the noise???)
        pass




if __name__ == '__main__':
    pass
