#include <stdlib.h>
#include <stdio.h>
#include <bdduser.h>

int main (int argc, char* argv[])
{
    // Create the universe..
    bdd_manager bddm = bdd_init();  

    bdd x1 = bdd_new_var_last(bddm);
    bdd x2 = bdd_new_var_last(bddm);
    bdd x3 = bdd_new_var_last(bddm);
    bdd x4 = bdd_new_var_last(bddm);
    bdd x5 = bdd_new_var_last(bddm);
    bdd x6 = bdd_new_var_last(bddm);
    bdd x7 = bdd_new_var_last(bddm);
    bdd x8 = bdd_new_var_last(bddm);

    bdd s4 = bdd_xor(bddm, x4, x8);
    bdd c4 = bdd_and(bddm, x4, x8);

    bdd temp3 = bdd_xor(bddm, x3, x7);
    bdd s3 = bdd_xor(bddm, c4, temp3);
    bdd c3 = bdd_or(bddm, bdd_and(bddm, x3, x7), bdd_xor(bddm, temp3, c4));

    bdd temp2 = bdd_xor(bddm, x2, x5);
    bdd s2 = bdd_xor(bddm, c3, temp2);
    bdd c2 = bdd_or(bddm, bdd_and(bddm, x2, x5), bdd_xor(bddm, temp2, c3));

    bdd temp1 = bdd_xor(bddm, x1, x4);
    bdd s1 = bdd_xor(bddm, c2, temp1);
    // bdd c1 = bdd_or(bddm, bdd_and(bddm, x1, x4), bdd_xor(bddm, temp1, c2));

    printf("s1:\n");
    bdd_print_bdd(bddm, s1, NULL, NULL, NULL, stdout);
    printf("s2:\n");
    bdd_print_bdd(bddm, s2, NULL, NULL, NULL, stdout);
    printf("s3:\n");
    bdd_print_bdd(bddm, s3, NULL, NULL, NULL, stdout);
    printf("s4:\n");
    bdd_print_bdd(bddm, s4, NULL, NULL, NULL, stdout);
    printf("-------------------------------\n");

    bdd y1 = bdd_new_var_last(bddm);
    bdd y2 = bdd_new_var_last(bddm);
    bdd y3 = bdd_new_var_last(bddm);
    bdd y4 = bdd_new_var_last(bddm);

    bdd eq1 = bdd_xnor(bddm, s1, y1);
    bdd eq2 = bdd_xnor(bddm, s2, y2);
    bdd eq3 = bdd_xnor(bddm, s3, y3);
    bdd eq4 = bdd_xnor(bddm, s4, y4);

    bdd chai = bdd_and(bddm, bdd_and(bddm, eq1, eq2), bdd_and(bddm, eq3, eq4));

    bdd A = bdd_xor(bddm, bdd_xor(bddm, x1, x2), bdd_xor(bddm, x3, x4));
    bdd img_comp = bdd_and(bddm, A, chai);

    bdd assoc_array1[9] = {x1, x2, x3, x4, x5, x6, x7, x8, 0}; // new assoc
    int assoc1 = bdd_new_assoc(bddm, assoc_array1, 0);
    bdd_assoc(bddm, assoc1);

    printf("part b:\n");
    bdd_print_bdd(bddm, bdd_exists(bddm, img_comp), NULL, NULL, NULL, stdout);
    printf("-------------------------------\n");

    bdd B = bdd_xor(bddm, bdd_xor(bddm, y1, y2), bdd_xor(bddm, y3, y4));
    bdd pre_img_comp = bdd_and(bddm, B, chai);
    
    bdd assoc_array2[5] = {y1, y2, y3, y4, 0}; // new assoc
    int assoc2 = bdd_new_assoc(bddm, assoc_array2, 0);
    bdd_assoc(bddm, assoc2);

    printf("part c:\n");
    bdd_print_bdd(bddm, bdd_exists(bddm, pre_img_comp), NULL, NULL, NULL, stdout);

    return(0);
}

