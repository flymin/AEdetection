import torch
import numpy as np

# http://stackoverflow.com/a/43554072/1864688


def _pad_amount(k):
    added = k - 1
    # note: this imitates scipy, which puts more at the beginning
    end = added // 2
    start = added - end
    return [start, end]


def _neighborhood(x, kh, kw):
    # input: N, C, H, W
    # output: N, H, W, C, KH, KW
    # padding is REFLECT
    # https://discuss.pytorch.org/t/tf-extract-image-patches-in-pytorch/43837/8
    x_np = x.cpu().numpy()
    x_pad = np.pad(x_np, ([0, 0], [0, 0], _pad_amount(kh), _pad_amount(kw)),
                   mode='symmetric')
    x_pad = torch.tensor(x_pad).to(x.device)
    # x_pad = F.pad(x, (*pad_amount(kw), *pad_amount(kh)), "reflect")
    # N, C, H, W, KH, KW
    patches = x_pad.unfold(2, kh, 1).unfold(3, kw, 1).contiguous()
    return patches


def _median_filter_no_reshape(x, kh, kw):
    neigh_size = kh * kw
    # get neighborhoods in shape (whatever, neigh_size)
    x_neigh = _neighborhood(x, kh, kw)  # N, C, H, W, KH, KW
    x_neigh = x_neigh.reshape(-1, neigh_size)  # N*C*H*W*C x KH*KW
    # note: this imitates scipy, which doesn't average with an even number of elements
    # get half, but rounded up
    rank = neigh_size - neigh_size // 2
    x_top, _ = torch.topk(x_neigh, rank)
    # bottom of top half should be middle
    x_mid = x_top[..., -1]
    # return tf.reshape(x_mid, (xs[0], xs[1], xs[2], xs[3]))
    return x_mid


def _truncated_normal(shape, mean=0, std=0.09):
    # Ref: https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    tensor = torch.Tensor(shape)
    tmp = tensor.new_empty(shape + [4]).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def _median_random_filter_no_reshape(x, kh, kw):
    neigh_size = kh * kw
    # get neighborhoods in shape (whatever, neigh_size)
    x_neigh = _neighborhood(x, kh, kw)  # N, H, W, C, KH, KW
    x_neigh = x_neigh.reshape(-1, neigh_size)  # N*H*W*C x KH*KW
    # note: this imitates scipy, which doesn't average with an even number of elements
    # get half, but rounded up
    rank = neigh_size - neigh_size // 2
    rand_int = _truncated_normal([1], mean=0, std=neigh_size / 4)[0]
    rand_int = rand_int.int()
    x_top, _ = torch.topk(x_neigh, rank + rand_int)
    # bottom of top half should be middle
    x_mid = x_top[..., -1]
    return x_mid


def median_filter(x, kh, kw=-1):
    if kw == -1:
        kw = kh
    xs = x.shape
    x_mid = _median_filter_no_reshape(x, kh, kw)
    return x_mid.reshape(xs[0], xs[1], xs[2], xs[3])


def median_random_filter(x, kh, kw):
    xs = x.shape
    x_mid = _median_random_filter_no_reshape(x, kh, kw)
    return x_mid.reshape(xs[0], xs[1], xs[2], xs[3])


def median_random_pos_size_filter(x):
    # Get two/multiple x_mid, randomly select from one .
    s0 = _median_random_filter_no_reshape(x, 2, 2)
    s1 = _median_random_filter_no_reshape(x, 3, 3)
    s2 = _median_random_filter_no_reshape(x, 4, 4)

    xs = x.shape
    nb_pixels = xs[0] * xs[1] * xs[2] * xs[3]
    samples_mnd = torch.multinomial(
        torch.log(torch.tensor([[10., 10., 10.]])),
        nb_pixels, replacement=True).squeeze()

    # return tf.constant([0]*nb_pixels, dtype=tf.int64)
    zeros = torch.zeros([nb_pixels], dtype=torch.int64)
    ones = torch.ones([nb_pixels], dtype=torch.int64)
    twos = torch.ones([nb_pixels], dtype=torch.int64) * 2
    # tmp = tf.cast(tf.equal(samples_mnd, tf.zeros([nb_pixels], dtype=tf.int64)), tf.int64)
    # return zeros, ones, twos

    selected_0 = (samples_mnd == zeros).float()
    selected_1 = (samples_mnd == ones).float()
    selected_2 = (samples_mnd == twos).float()

    # return s0, selected_0
    x_mid = s0 * selected_0 + s1 * selected_1 + s2 * selected_2
    return x_mid.reshape(xs[0], xs[1], xs[2], xs[3])


def median_random_size_filter(x):
    # Get two/multiple x_mid, randomly select from one .
    s0 = _median_filter_no_reshape(x, 2, 2)
    s1 = _median_filter_no_reshape(x, 3, 3)
    # s2 = median_filter_no_reshape(x, 4, 4)

    xs = x.shape
    nb_pixels = xs[0] * xs[1] * xs[2] * xs[3]
    samples_mnd = torch.multinomial(
        torch.log(torch.tensor([[10., 10.]])),
        nb_pixels, replacement=True).squeeze()

    # return tf.constant([0]*nb_pixels, dtype=tf.int64)
    zeros = torch.zeros([nb_pixels], dtype=torch.int64)
    ones = torch.ones([nb_pixels], dtype=torch.int64)
    # twos = tf.ones([nb_pixels], dtype=tf.int64)*2
    # tmp = tf.cast(tf.equal(samples_mnd, tf.zeros([nb_pixels], dtype=tf.int64)), tf.int64)
    # return zeros, ones, twos

    selected_0 = (samples_mnd == zeros).float()
    selected_1 = (samples_mnd == ones).float()
    # selected_2 = tf.cast(tf.equal(samples_mnd, twos), tf.float32)

    # return s0, selected_0
    # x_mid = tf.add_n( [tf.multiply(s0, selected_0), tf.multiply(s1, selected_1), tf.multiply(s2, selected_2)] )
    x_mid = s0 * selected_0 + s1 * selected_1
    return x_mid.reshape(xs[0], xs[1], xs[2], xs[3])


if __name__ == '__main__':
    import numpy as np
    from scipy import ndimage

    vec1 = np.asarray([[[[0, 16], [1, 17], [2, 18], [3, 19]],
                        [[4, 20], [5, 21], [6, 22], [7, 23]],
                        [[8, 24], [9, 25], [10, 26], [11, 27]],
                        [[12, 28], [13, 29], [14, 30], [15, 31]]]], dtype=np.float32)
    vec2 = np.asarray([[[[3, 16], [3, 17], [3, 18], [3, 19]],
                        [[1, 20], [1, 21], [1, 22], [7, 23]],
                        [[1, 24], [2, 25], [3, 26], [11, 27]],
                        [[12, 28], [13, 29], [14, 30], [15, 31]]]], dtype=np.float32)

    for vec in [vec1, vec2]:
        print("vec:", vec)
        mnp = ndimage.filters.median_filter(
            vec, size=(1, 3, 3, 1), mode='reflect')
        print("mnp", mnp)

        vec = torch.tensor(vec).permute(0, 3, 1, 2)
        import ipdb
        mtf = median_filter(vec, 3, 3).permute(0, 2, 3, 1).numpy()
        print("mtf", mtf)
        mtf_rand_1 = median_random_pos_size_filter(
            vec).permute(0, 2, 3, 1).numpy()
        mtf_rand_2 = median_random_pos_size_filter(
            vec).permute(0, 2, 3, 1).numpy()
        print("mtf_rand_1", mtf_rand_1)
        print("mtf_rand_2", mtf_rand_2)

        print("equal", np.array_equal(mnp, mtf))
        print("equal", np.array_equal(mnp, mtf_rand_1))
        print("equal", np.array_equal(mtf_rand_1, mtf_rand_2))

    # print sess.run(g, feed_dict={X: vec})

    import imageio
    import torchvision

    image = imageio.imread('test/panda.png')

    X2 = torchvision.transforms.transforms.F.to_tensor(image).unsqueeze(0)
    ipdb.set_trace()
    images_blur = median_filter(X2, 3, 3).permute(0, 2, 3, 1).numpy() * 255
    images_rand_blur = median_random_pos_size_filter(
        X2).permute(0, 2, 3, 1).numpy() * 255

    from PIL import Image

    names = ['test/panda_orig.png', 'test/panda_blur_3_3.png',
             'test/panda_rand_blur.png']
    for i, img in enumerate([image, images_blur, images_rand_blur]):
        img = Image.fromarray(np.squeeze(img).astype(np.uint8), 'RGB')
        img.save(names[i])
