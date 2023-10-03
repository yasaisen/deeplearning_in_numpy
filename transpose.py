import numpy as np

class convolution_transpose:
    def im2col(images, flt_h, flt_w, out_h, out_w, stride=1, pad=0):  # 4Darray(input_height, input_width, channel, batch) to 2Darray(filter_height * filter_width * channel ,batch * output_height * output_width)
    
        n_bt, n_ch, img_h, img_w = images.shape  # get batch, channel, input_height, input_width
        
        img_pad = np.pad(images, [(0,0), (0,0), (pad, pad), (pad, pad)], "constant")  # doing padding, constant : fill in same value(?)
        print(img_pad)
        cols = np.zeros((n_bt, n_ch, flt_h, flt_w, out_h, out_w))  # make a all-zero 6Darray
        
        for h in range(flt_h):
            h_lim = h + stride * out_h
            for w in range(flt_w):
                w_lim = w + stride * out_w
                cols[:, :, h, w, :, :] = img_pad[:, :, h:h_lim:stride, w:w_lim:stride]  # walking

        cols = cols.transpose(1, 2, 3, 0, 4, 5).reshape(n_ch * flt_h * flt_w, n_bt * out_h * out_w)
        return cols


    def col2im(cols, img_shape, flt_h, flt_w, out_h, out_w, stride=1, pad=0):
    
        n_bt, n_ch, img_h, img_w = img_shape
        
        cols = cols.reshape(n_ch, flt_h, flt_w, n_bt, out_h, out_w).transpose(3, 0, 1, 2, 4, 5)
        images = np.zeros((n_bt, n_ch, img_h+2*pad+stride-1, img_w+2*pad+stride-1))
        
        for h in range(flt_h):
            h_lim = h + stride*out_h
            for w in range(flt_w):
                w_lim = w + stride*out_w
                images[:, :, h:h_lim:stride, w:w_lim:stride] += cols[:, :, h, w, :, :]

        return images[:, :, pad:img_h+pad, pad:img_w+pad]





img = np.arange(54).reshape(2,3,3,3)
cols = convolution_transpose.im2col(img, 2, 2, 2, 2, 1, 0)

print(cols)


# cols = np.ones((4, 9))
# img_shape = (1, 1, 4, 4)
# images = col2im(cols, img_shape, 2, 2, 3, 3, 1, 0)

# print(images)








