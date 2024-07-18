from typing import List, Tuple
import torch
import numpy as np

# AES Sbox
sbox = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
], dtype=np.uint32)

xtime_lut = np.array(
    [0x0, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 0x22, 0x24, 0x26,
     0x28, 0x2a, 0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e, 0x40, 0x42, 0x44, 0x46, 0x48, 0x4a,
     0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 0x58, 0x5a, 0x5c, 0x5e, 0x60, 0x62, 0x64, 0x66, 0x68, 0x6a, 0x6c, 0x6e,
     0x70, 0x72, 0x74, 0x76, 0x78, 0x7a, 0x7c, 0x7e, 0x80, 0x82, 0x84, 0x86, 0x88, 0x8a, 0x8c, 0x8e, 0x90, 0x92,
     0x94, 0x96, 0x98, 0x9a, 0x9c, 0x9e, 0xa0, 0xa2, 0xa4, 0xa6, 0xa8, 0xaa, 0xac, 0xae, 0xb0, 0xb2, 0xb4, 0xb6,
     0xb8, 0xba, 0xbc, 0xbe, 0xc0, 0xc2, 0xc4, 0xc6, 0xc8, 0xca, 0xcc, 0xce, 0xd0, 0xd2, 0xd4, 0xd6, 0xd8, 0xda,
     0xdc, 0xde, 0xe0, 0xe2, 0xe4, 0xe6, 0xe8, 0xea, 0xec, 0xee, 0xf0, 0xf2, 0xf4, 0xf6, 0xf8, 0xfa, 0xfc, 0xfe,
     0x1b, 0x19, 0x1f, 0x1d, 0x13, 0x11, 0x17, 0x15, 0xb, 0x9, 0xf, 0xd, 0x3, 0x1, 0x7, 0x5, 0x3b, 0x39, 0x3f, 0x3d,
     0x33, 0x31, 0x37, 0x35, 0x2b, 0x29, 0x2f, 0x2d, 0x23, 0x21, 0x27, 0x25, 0x5b, 0x59, 0x5f, 0x5d, 0x53, 0x51,
     0x57, 0x55, 0x4b, 0x49, 0x4f, 0x4d, 0x43, 0x41, 0x47, 0x45, 0x7b, 0x79, 0x7f, 0x7d, 0x73, 0x71, 0x77, 0x75,
     0x6b, 0x69, 0x6f, 0x6d, 0x63, 0x61, 0x67, 0x65, 0x9b, 0x99, 0x9f, 0x9d, 0x93, 0x91, 0x97, 0x95, 0x8b, 0x89,
     0x8f, 0x8d, 0x83, 0x81, 0x87, 0x85, 0xbb, 0xb9, 0xbf, 0xbd, 0xb3, 0xb1, 0xb7, 0xb5, 0xab, 0xa9, 0xaf, 0xad,
     0xa3, 0xa1, 0xa7, 0xa5, 0xdb, 0xd9, 0xdf, 0xdd, 0xd3, 0xd1, 0xd7, 0xd5, 0xcb, 0xc9, 0xcf, 0xcd, 0xc3, 0xc1,
     0xc7, 0xc5, 0xfb, 0xf9, 0xff, 0xfd, 0xf3, 0xf1, 0xf7, 0xf5, 0xeb, 0xe9, 0xef, 0xed, 0xe3, 0xe1, 0xe7, 0xe5],
    dtype=np.uint32)


def shift_rows(x: np.ndarray, inverse=False) -> np.ndarray:
    # x: (4, 4, batch_size)
    z = np.zeros_like(x)
    for i in range(4):
        if inverse:
            z[i, :, :] = np.roll(x[i, :, :], i, axis=0)
        else:
            z[i, :, :] = np.roll(x[i, :, :], -i, axis=0)
    return z


def mix_columns(x: np.ndarray, use_torch=False) -> Tuple[np.ndarray, dict]:
    # x: (4, batch_size)
    # full_mixcol: (21, batch_size)

    x1, x2, x3, x4 = x[0, :], x[1, :], x[2, :], x[3, :]
    x12 = x1 ^ x2
    x23 = x2 ^ x3
    x34 = x3 ^ x4
    x41 = x4 ^ x1

    g = x12 ^ x34

    xx12 = xtime(x12, use_torch=use_torch)
    xx23 = xtime(x23, use_torch=use_torch)
    xx34 = xtime(x34, use_torch=use_torch)
    xx41 = xtime(x41, use_torch=use_torch)

    xx12g = xx12 ^ g
    xx23g = xx23 ^ g
    xx34g = xx34 ^ g
    xx41g = xx41 ^ g

    xm1 = x1 ^ xx12g
    xm2 = x2 ^ xx23g
    xm3 = x3 ^ xx34g
    xm4 = x4 ^ xx41g

    if use_torch:
        output = torch.stack([x1, x2, x3, x4,
                              xm1, xm2, xm3, xm4,
                              x12, x23, x34, x41,
                              g,
                              xx12, xx23, xx34, xx41,
                              xx12g, xx23g, xx34g, xx41g], dim=0)
    else:
        output = np.stack([x1, x2, x3, x4,
                           xm1, xm2, xm3, xm4,
                           x12, x23, x34, x41,
                           g,
                           xx12, xx23, xx34, xx41,
                           xx12g, xx23g, xx34g, xx41g], axis=0)

    arg_dict = {
        "x1": x1, "x2": x2, "x3": x3, "x4": x4,
        "x12": x12, "x23": x23, "x34": x34, "x41": x41,
        "g": g,
        "xx12": xx12, "xx23": xx23, "xx34": xx34, "xx41": xx41,
        "xx12g": xx12g, "xx23g": xx23g, "xx34g": xx34g, "xx41g": xx41g,
        "xm1": xm1, "xm2": xm2, "xm3": xm3, "xm4": xm4
    }

    assert output.shape == (21, x.shape[1])
    return output, arg_dict


def xtime(x, use_torch=False):
    if use_torch:
        res = (torch.bitwise_xor(x << 1, ((x >> 7) & 1) * 0x1b)) & 0xff
    else:
        res = (np.bitwise_xor(x << 1, ((x >> 7) & 1) * 0x1b)) & 0xff

    return res