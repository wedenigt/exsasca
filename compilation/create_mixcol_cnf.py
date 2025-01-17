import itertools
from pyeda.inter import *

def bit_to_byte_constraints(bits, bytes):
    """
    bits: a list of 8 binary variables
    bytes: a list of 256 binary variables (one for each state of a byte)
    Returns a list of constraints that enforce the bits to be the bits of byte
    """
    assert len(bits) == 8
    assert len(bytes) == 256
    alpha = expr(True)

    # model 00000000 => B_0,
    #           ...
    #       11111111 => B_255
    for byte_val, bit_assignment in enumerate(itertools.product([0, 1], repeat=8)):
        clause = expr(False)
        for bit_idx, bit in enumerate(bit_assignment):
            # negate first part of implication
            if bit == 0:
                clause = clause | bits[bit_idx]
            else:
                clause = clause | ~bits[bit_idx]
        clause = clause | bytes[byte_val]
        alpha = alpha & clause; alpha = alpha.simplify()

    return alpha

def xtimes(a, b):
    c1 = ~a[7] & ~b[0]
    for i in range(1, 8):
        c1 = c1 & (a[i - 1] | ~b[i]) & (~a[i - 1] | b[i])

    c2 = a[7] & b[0] & (~a[0] | ~b[1]) & (a[0] | b[1]) & (a[1] | ~b[2]) & (~a[1] | b[2]) & (~a[2] | ~b[3]) & (
            a[2] | b[3]) & (~a[3] | ~b[4]) & (a[3] | b[4]) & (a[4] | ~b[5]) & (~a[4] | b[5]) & (
                 a[5] | ~b[6]) & (~a[5] | b[6]) & (a[6] | ~b[7]) & (~a[6] | b[7])

    return (c1 | c2).simplify()


def xor_cnf(a_list, b_list, c_list):
    return And(*(Equal(Xor(a, b), c) for a, b, c in zip(a_list, b_list, c_list))).simplify()


def mix_single_column_own(var_list_in, var_list_out, var_list_intermediate):
    assert len(var_list_in) == 4  # there must be 4 variables in a column
    assert len(var_list_out) == 4
    assert len(var_list_intermediate) >= 13  # each column needs exactly 13 intermediates

    x, x_mix, v = var_list_in, var_list_out, var_list_intermediate

    alpha = xor_cnf(x[0], x[1], v[0]) # x01 = x0 ^ x1
    alpha = alpha & xor_cnf(x[1], x[2], v[1]) # x12 = x1 ^ x2
    alpha = alpha & xor_cnf(x[2], x[3], v[2]) # x23 = x2 ^ x3
    alpha = alpha & xor_cnf(x[3], x[0], v[3]) # x30 = x3 ^ x0

    alpha = alpha & xor_cnf(v[0], v[2], v[4]) # v[4] is the global xor
    # v[2] == x[0] ^ x[1] ^ x[2] ^ x[3]
    # v[2] is `Tmp` in tinyaes.c
    tmp = v[4]
    alpha = alpha.simplify()

    alpha = alpha & xtimes(v[0], v[5]) # v[5] = xtime[x01]
    alpha = alpha & xtimes(v[1], v[6]) # v[6] = xtime[x12]
    alpha = alpha & xtimes(v[2], v[7]) # v[7] = xtime[x23]
    alpha = alpha & xtimes(v[3], v[8]) # v[8] = xtime[x30]

    alpha = alpha & xor_cnf(tmp, v[5], v[9]) # v[9] = xtime[x01] ^ Tmp
    alpha = alpha & xor_cnf(tmp, v[6], v[10]) # v[10] = xtime[x12] ^ Tmp
    alpha = alpha & xor_cnf(tmp, v[7], v[11]) # v[11] = xtime[x23] ^ Tmp
    alpha = alpha & xor_cnf(tmp, v[8], v[12]) # v[12] = xtime[x30] ^ Tmp

    alpha = alpha & xor_cnf(x[0], v[9], x_mix[0]) # x_mix[0] = x0 ^ v[9]
    alpha = alpha & xor_cnf(x[1], v[10], x_mix[1]) # x_mix[1] = x1 ^ v[10]
    alpha = alpha & xor_cnf(x[2], v[11], x_mix[2]) # x_mix[2] = x2 ^ v[11]
    alpha = alpha & xor_cnf(x[3], v[12], x_mix[3]) # x_mix[3] = x3 ^ v[12]

    return alpha

def mix_single_column_no_intermediates(var_list_in, var_list_out):
    assert len(var_list_in) == 4  # there must be 4 variables in a column
    assert len(var_list_out) == 4
    x, x_mix = var_list_in, var_list_out

    x01 = x[0] ^ x[1]
    x12 = x[1] ^ x[2]
    x23 = x[2] ^ x[3]
    x30 = x[3] ^ x[0]
    g = x01 ^ x23
    xx01 = xtimes_no_intermediates(x01)
    xx12 = xtimes_no_intermediates(x12)
    xx23 = xtimes_no_intermediates(x23)
    xx30 = xtimes_no_intermediates(x30)

    xx01g = xx01 ^ g
    xx12g = xx12 ^ g
    xx23g = xx23 ^ g
    xx30g = xx30 ^ g

    xm0 = x[0] ^ xx01g
    xm1 = x[1] ^ xx12g
    xm2 = x[2] ^ xx23g
    xm3 = x[3] ^ xx30g

    xm0_equal = And(*(Equal(xm0[i], x_mix[0][i]) for i in range(8)))
    xm1_equal = And(*(Equal(xm1[i], x_mix[1][i]) for i in range(8)))
    xm2_equal = And(*(Equal(xm2[i], x_mix[2][i]) for i in range(8)))
    xm3_equal = And(*(Equal(xm3[i], x_mix[3][i]) for i in range(8)))

    return And(xm0_equal, xm1_equal, xm2_equal, xm3_equal).simplify()

def xtimes_no_intermediates(x):
    assert len(x) == 8, 'x must be a byte'
    out = exprzeros(8)
    out[0] = x[7]
    out[1] = Xor(x[0], x[7])
    out[3] = Xor(x[2], x[7])
    out[4] = Xor(x[3], x[7])

    for i in [2,5,6,7]:
        out[i] = x[i-1]

    return out


def create_mixcol_cnf(output_name='mixcols.cnf', bit_to_byte=False, use_intermediates=False,
                      condition_first_last=False, condition_on_g=False):
    assert not (condition_first_last and condition_on_g), 'Cannot condition on both g and first/last bits'
    final = expr(True)

    sbox_out = exprvars(f'x', 4, 8)
    mixcol_out = exprvars(f'x_mix', 4, 8)
    mixcol_intermediate = exprvars(f'z', 13, 8)  # 13 intermediates needed for mixcol

    if bit_to_byte:
        bytes = exprvars(f'B', 8, 256)  # 8 bytes (4 input, 4 output), 256 states each
        bits = [sbox_out[i] for i in range(4)] + [mixcol_out[i] for i in range(4)]
        for bit, byte in zip(bits, bytes):
            final = final & bit_to_byte_constraints(bit, byte)

    if use_intermediates:
        mix_col_expr = mix_single_column_own(sbox_out, mixcol_out, mixcol_intermediate)
    else:
        mix_col_expr = mix_single_column_no_intermediates(sbox_out, mixcol_out)


    final = final & mix_col_expr.simplify()
    if condition_first_last: # condition 8 bits on zero
        final = And(*final.to_cnf().xs, *[Or(~sbox_out[i, 0]) for i in range(4)],
                    *[Or(~sbox_out[i, 7]) for i in range(4)])

    if condition_on_g:
        # let g = 0000 0000
        final = And(*final.to_cnf().xs, *[Or(~mixcol_intermediate[4, i]) for i in range(8)])
        # the following actually substitutes the g variable with 0000 0000, but the SDD size is actually larger by 1k (!)
        # final = final.restrict(point={mixcol_intermediate[4, i]: 0 for i in range(8)})

    final = final.simplify()

    cnf = final.to_cnf()
    print(f'CNF Size: {cnf.size}')
    t, dimacs = expr2dimacscnf(cnf)
    # write dimacs to file
    with open(output_name, 'w') as f:
        f.write(str(dimacs))

    return 168 if use_intermediates else 64

if __name__ == '__main__':
    CNF_OUT = 'out.cnf'
    CONDITION_FIRST_LAST = False
    CONDITION_ON_G = False
    num_cnf_variables = create_mixcol_cnf(output_name=str(CNF_OUT), use_intermediates=True,
                                        condition_first_last=CONDITION_FIRST_LAST, condition_on_g=CONDITION_ON_G)