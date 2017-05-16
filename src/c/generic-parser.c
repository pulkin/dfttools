#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void __debug__(FILE *f) {
    long int pos = ftell(f);
    char *line = NULL;
    size_t len;
    if (getline(&line, &len, f) != -1) printf("Line: [%s]\n", line);
    if (line) free(line);
    fseek(f,pos,SEEK_SET);
}

int skip_either(char **c, int n, FILE *f) {
    char ch;
    int i[n];
    memset(i, 0, sizeof(i));
    int j;
    
    while (1) {
        
        ch = fgetc(f);
        for (j=0; j<n; j++) {
            if (ch == EOF) return -1;
            if (ch == c[j][i[j]]) i[j]++; else i[j] = 0;
            if (c[j][i[j]] == 0) return j;
        }
    }
    
    return -1;
}

int skip(char *c, FILE *f) {
    if (skip_either(&c, 1, f) == 0) return 1; else return 0;
}

int skip_line(FILE *f) {
    return skip("\n", f);
}

int skip_line_n(FILE *f, int n) {
    int i;
    for (i=0; i<n; i++) if (!skip_line(f)) return 0;
    return 1;
}

int present(char *c, FILE *f) {
    long int pos = ftell(f);
    int result = skip(c,f);
    fseek(f,pos,SEEK_SET);
    return result;
}

int present_either(char **c, int n, FILE *f) {
    long int pos = ftell(f);
    int result = skip_either(c,n,f);
    fseek(f,pos,SEEK_SET);
    return result;
}

int present_either2(char *c1, char *c2, FILE *f) {
    char *c[2];
    c[0] = c1; c[1] = c2;
    return present_either(c, 2, f);
}

long int position_of(char *c, FILE *f) {
    long int pos = ftell(f);
    int result = skip(c,f);
    long int p2 = ftell(f);
    fseek(f,pos,SEEK_SET);
    if (result) return p2; else return -1;
}
