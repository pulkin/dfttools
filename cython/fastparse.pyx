# cython: language_level=3
from libc.stdio cimport FILE, fscanf, fgetc, EOF, ftell, fseek, SEEK_SET
from libc.string cimport memset


cdef char to_lower_case(const char x):
    if x >= 0x41 and x <= 0x5A:
        return x + 0x20
    return x


cdef int skip_either(const char **c, int n, FILE *f):
    assert n < 16
    cdef char ch
    cdef int i[16]
    cdef int j
    memset(i, 0, sizeof(i))

    while True:
        ch = fgetc(f)
        for j in range(n):
            if ch == EOF:
                return -1
            if ch == c[j][i[j]]:
                i[j] += 1
            else:
                i[j] = 0
            if c[j][i[j]] == 0:
                return j
    return -1;


cdef int skip(const char *c, FILE *f):
    if skip_either(&c, 1, f) == 0:
        return 1
    return 0


cdef int skip_line(FILE *f):
    return skip("\n", f)


cdef int skip_line_n(FILE *f, int n):
    cdef int i
    for i in range(n):
        if not skip_line(f):
            return 0
    return 1


cdef int present(const char *c, FILE *f):
    cdef long int pos = ftell(f)
    cdef int result = skip(c, f)
    fseek(f, pos, SEEK_SET)
    return result


cdef int present_either(const char **c, int n, FILE *f):
    cdef long int pos = ftell(f)
    cdef int result = skip_either(c, n, f)
    fseek(f, pos, SEEK_SET)
    return result


cdef int present_either2(const char *c1, const char *c2, FILE *f):
    return present_either([c1, c2], 2, f)


cdef long int position_of(const char *c, FILE *f):
    cdef long int pos = ftell(f)
    cdef int result = skip(c, f)
    cdef long int p2 = ftell(f)
    fseek(f, pos, SEEK_SET)
    if result:
        return p2
    else:
        return -1
