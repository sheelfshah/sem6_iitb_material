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
    bdd y1 = bdd_new_var_last(bddm);
    bdd y2 = bdd_new_var_last(bddm);
    bdd y3 = bdd_new_var_last(bddm);

    bdd assoc_array[9] = {x1, x2, x3, 0}; // new assoc
    int assoc = bdd_new_assoc(bddm, assoc_array, 0);
    bdd_assoc(bddm, assoc);

    // f1 is the permutation function
    bdd f11 = x3;
    bdd f12 = x1;
    bdd f13 = x2;
    // f2 is non-invertible, as one variable is dropped
    bdd f21 = x1;
    bdd f22 = x1;
    bdd f23 = x2;

    bdd temp11 = bdd_xnor(bddm, f11, y1);
    bdd temp12 = bdd_xnor(bddm, f12, y2);
    bdd temp13 = bdd_xnor(bddm, f13, y3);
    bdd chai1 = bdd_and(bddm, temp11, bdd_and(bddm, temp12, temp13));
    printf("f1: \n");
    bdd_print_bdd(bddm, bdd_exists(bddm, chai1), NULL, NULL, NULL, stdout);

    bdd temp21 = bdd_xnor(bddm, f21, y1);
    bdd temp22 = bdd_xnor(bddm, f22, y2);
    bdd temp23 = bdd_xnor(bddm, f23, y3);
    bdd chai2 = bdd_and(bddm, temp21, bdd_and(bddm, temp22, temp23));
    printf("f2: \n");
    bdd_print_bdd(bddm, bdd_exists(bddm, chai2), NULL, NULL, NULL, stdout);

    return(0);
}

