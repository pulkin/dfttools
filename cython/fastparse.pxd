from libc.stdio cimport FILE


cdef char to_lower_case(const char x)
cdef int skip_either(const char **c, int n, FILE *f)
cdef int skip(const char *c, FILE *f)
cdef int skip_line(FILE *f)
cdef int skip_line_n(FILE *f, int n)
cdef int present(const char *c, FILE *f)
cdef int present_either(const char **c, int n, FILE *f)
cdef int present_either2(const char *c1, const char *c2, FILE *f)
cdef long int position_of(const char *c, FILE *f)
