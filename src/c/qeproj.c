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

int next_line(FILE *f) {
    return skip("\n", f);
}

long int position_of(char *c, FILE *f) {
    long int pos = ftell(f);
    int result = skip(c,f);
    long int p2 = ftell(f);
    fseek(f,pos,SEEK_SET);
    if (result) return p2; else return -1;
}

int n_bands(FILE *f) {
    if (!skip("k =",f)) return -1;
    int result = 0;
    while (present_either2("==== e(","k =",f) == 0) {
        skip("==== e(",f);
        result++;
    }
    return result;
}

int n_basis(FILE *f) {
    if (!skip("Calling projwave",f)) return -1;
    if (!skip(":\n\n",f)) return -1;
    if (!present("\n\n",f)) return -1;
    int result = 0;
    while (present_either2("state #","\n\n",f) == 0) {
        skip("state #",f);
        result++;
    }
    return result;
}

int _weights(float **data, int basis_size, int bands_number, FILE *f) {
    
    if (!skip("Calling projwave", f)) return -1;
    
    int nk = 0;
    
    int nk_allocated = 1;
    int multiplier = bands_number*basis_size;
    *data = malloc(sizeof(float)*nk_allocated*multiplier);
    memset(*data, 0, sizeof(*data[0])*nk_allocated*multiplier);
        
    while (present("k =", f)) {
        
        if (!skip("k =",f)) return -1;
        
        if (nk == nk_allocated) {
            nk_allocated = nk_allocated * 2;
            *data = realloc(*data, sizeof(float)*nk_allocated*multiplier);
            memset((*data) + nk_allocated*multiplier/2, 0, sizeof(float)*nk_allocated*multiplier/2);
        }
        
        int ne;
                
        for (ne=0; ne<bands_number; ne++) {
            
            if (!skip("==== e(",f)) return -1;
            if (!next_line(f)) return -1;
            if (!skip("psi =", f)) return -1;
            
            int state;
            int w1,w2;
            while (fscanf(f, "%d.%d*[#%d]+", &w1, &w2, &state) == 3) {
                (*data)[nk*multiplier + ne*basis_size + state-1] = 1.0*w1+1e-3*w2;
            }
        
        }
        
        nk++;
        
    }
    
    return nk;
    
}

int weights(float **data, int dims[3], FILE *f) {
    
    long int pos = ftell(f);
    int result_basis = n_basis(f);
    fseek(f,pos,SEEK_SET);

    if (result_basis<0) return 0;
    
    pos = ftell(f);
    int result_bands = n_bands(f);
    fseek(f,pos,SEEK_SET);

    if (result_bands<0) return 0;
    
    pos = ftell(f);
    int result = _weights(data, result_basis, result_bands, f);
    fseek(f,pos,SEEK_SET);
    
    if (result<0) {
        if (*data) free(*data);
        return 0;
    }
    
    dims[0] = result;
    dims[1] = result_bands;
    dims[2] = result_basis;
    return 1;
}
