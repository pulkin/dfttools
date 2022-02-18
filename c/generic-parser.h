void __debug__(FILE *f);
const char to_lower_case(const char x);

int skip_either(char **c, int n, FILE *f);
int skip(char *c, FILE *f);
int skip_line(FILE *f);
int skip_line_n(FILE *f, int n);

int present_either(char **c, int n, FILE *f);
int present_either2(char *c1, char *c2, FILE *f);
int present(char *c, FILE *f);

long int position_of(char *c, FILE *f);
