# import required library
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid

if __name__ == "__main__":
    # read images from computer
    a = read_image('./img/g1.png')
    b = read_image('./img/g2.png')
    c = read_image('./img/g3.png')
    d = read_image('./img/b1.png')
    e = read_image('./img/b2.png')
    f = read_image('./img/b3.png')
    # make grid from the input images
    # set nrow=3, and padding=25

    Grid = make_grid([a, b, c, d, e, f], nrow=2, padding=25)
    # display result
    img = torchvision.transforms.ToPILImage()(Grid)
    img.show()