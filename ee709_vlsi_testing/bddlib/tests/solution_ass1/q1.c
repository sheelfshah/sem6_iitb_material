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

    // compute y = x1.x2 + x3.x4 + x5.x6 + x7.x8
    bdd a = bdd_and(bddm, x1, x2);
    bdd b = bdd_and(bddm, x3, x4);
    bdd c = bdd_and(bddm, x5, x6);
    bdd d = bdd_and(bddm, x7, x8);
    bdd p = bdd_or(bddm, a, b);
    bdd q = bdd_or(bddm, c, d);
    bdd y = bdd_or(bddm, p, q);

    printf("----------------------------------------------------\n");

    // print y
    bdd_print_bdd(bddm, y, NULL, NULL, NULL, stdout);
    printf("number of nodes = %i\n", (int) bdd_size(bddm, y, 0));

    return(0);
}

