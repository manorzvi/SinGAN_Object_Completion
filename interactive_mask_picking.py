import torch
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

class track_mouse_motion:

    def __init__ ( self, fig, h, w ):
        self.ind   = 1
        self.mask  = torch.zeros([h,w], dtype=torch.int)
        self.cid   = fig.canvas.mpl_connect('motion_notify_event', self)

    def __call__ ( self, event ):
        if event.xdata != None and event.ydata != None:
            self.mask[round(event.xdata),round(event.ydata)] = self.ind
            self.ind = self.ind + 1

    def get_mask( ):
        return self.mask

class push_key_and_exit:

    def __init__ ( self, fig, mouse ):
        self.mouse = mouse
        self.cid   = fig.canvas.mpl_connect('key_press_event', self)

    def __call__ ( self, event ):
        h_mask = self.mouse.get_mask();
        # fix me: need to connect edges
        h, w = h_mask.shape
        for row in range(0, h):
            left_ind  = 0
            left_set  = 0
            right_ind = w-1
            right_set = 0
            for col in range(0, w):
                if not(left_set) and h_mask[row, col] > 0:
                    left_ind  = col
                    left_set  = 1
                if not(right_set) and h_mask[row, w-1-col] > 0:
                    right_ind = w-1-col
                    right_set = 1
            if left_set and right_set:
                h_mask[row, left_ind:right_ind] = 1



if __name__ == '__main__':
    img     = plt.imread("im.jpg")
    h, w, c = img.shape
    ax      = plt.gca()
    fig     = plt.gcf()
    implot  = ax.imshow(img)
    mouse   = track_mouse_motion(fig, h, w)
    key     = push_key_and_exit(fig, mouse)
    plt.show()
