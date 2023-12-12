import numpy as np

# log transform for image
# v: base
# c: constant
def log_transform(img, v=1, c=1):
    img_float = np.float32(img)
    img_float = img_float/255
    
    antilog = 1+v*img_float
    base = v+1

    img = (np.log(antilog) / np.log(base)) *c

    img = np.uint8(img*255)

    return img


# exponential transform
# y: gamma
# c: constant
def exp_tranform(img, y=0.5, c=1):
    img_f32 = np.float32(img)/255.0
    
    I_gamma = np.power(img_f32, y)*c
    
    img_u8 = np.uint8(I_gamma*255.0)

    return img_u8

# histogram equalization
def hist_equalize(img):

    b_channel = img[:,:,0]
    g_channel = img[:,:,1]
    r_channel = img[:,:,2]

    Height, Witdth = np.shape(img)[0], np.shape(img)[1]
    b_acc = np.zeros(256)
    g_acc = np.zeros(256)
    r_acc = np.zeros(256)

    for h in range(Height):
        for w in range(Witdth):

            b_acc[b_channel[h,w]] = b_acc[b_channel[h,w]]+1
            g_acc[g_channel[h,w]] = g_acc[g_channel[h,w]]+1
            r_acc[r_channel[h,w]] = r_acc[r_channel[h,w]]+1

    b_pk = b_acc / (Height*Witdth)
    g_pk = g_acc / (Height*Witdth)
    r_pk = r_acc / (Height*Witdth)

    # for each pixel in image, P' = P * CDF(P)
    for h in range(Height):
        for w in range(Witdth):
            b_channel[h,w] = b_channel[h,w]* np.sum(b_pk[0:b_channel[h,w]])
            g_channel[h,w] = g_channel[h,w]* np.sum(g_pk[0:g_channel[h,w]])
            r_channel[h,w] = r_channel[h,w]* np.sum(r_pk[0:r_channel[h,w]])
    
    img[:,:,0] = b_channel
    img[:,:,1] = g_channel
    img[:,:,2] = r_channel

    return img

# filter
def gaussfilt(img, sig=1, kernelSize=5):

    # generate gaussian kernel
    kernel = np.zeros((kernelSize,kernelSize))
    
    for x in range(kernelSize):
        for y in range(kernelSize):
            kx = x-1
            ky = y-1
            kernel[x,y] = (1/2*sig**2*np.pi) * np.exp(-((kx**2 + ky**2)/2*sig*22))

    kernel = kernel / np.sum(kernel)

    print(kernel)

    # apply kernel with img
    cols, rows, _ = np.shape(img)
    offset = np.uint8(np.floor(kernelSize/2))

    for c in range(offset, cols -offset):
        for r in range(offset, rows - offset):
            acc = 0
            for x in range(offset*2+1):
                for y in range(offset*2+1):
                    acc = acc + kernel[x,y]*img[c+x-offset,r+y-offset]
            img[c,r] = acc

    return img

# Laplacaion filter
def laplacaionfilt(img):

    kernel = np.array(([0, 1, 0],
                      [1,-4, 1],
                      [0, 1, 0]),np.float32)
    
    # apply kernel with img
    cols, rows, _ = np.shape(img)
    offset = np.uint8(np.floor(3/2))

    for c in range(offset, cols - offset):
        for r in range(offset, rows - offset):
            acc = np.zeros(3,np.float32)
            for x in range(offset*2+1):
                for y in range(offset*2+1):
                    acc = acc + kernel[x,y]*img[c+x-offset,r+y-offset]
            acc = np.clip(acc, 0, 255)
            img[c,r] = np.uint8(acc)
    
    return img

    