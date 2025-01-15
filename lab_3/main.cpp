#include <iostream>
#include <cmath>

using namespace std;

#define LEN_LIST_CORR 8

double corr(int x[LEN_LIST_CORR], int y[LEN_LIST_CORR])
{
    double summa = 0;
    for (int i = 0; i < LEN_LIST_CORR; i++)
    {
        summa += x[i] * y[i];
    }

    return summa;
}


double corr_normalization(int x[LEN_LIST_CORR], int y[LEN_LIST_CORR])
{
    double summa = 0, summa_x = 0, summa_y = 0;

    for (int i = 0; i < LEN_LIST_CORR; i++)
    {
        summa += x[i] * y[i];
        summa_x += pow(x[i], 2);
        summa_y += pow(y[i], 2);
    }
    
    double res = summa / sqrt(summa_x * summa_y);

    return res;
}



int main() {

    int a[LEN_LIST_CORR] = {7, 3, 2, -2, -2, -4, 1, 5};
    int b[LEN_LIST_CORR] = {2, 1, 5, 0, -2, -3, 2, 4};
    int c[LEN_LIST_CORR] = {2, -1, 3, -9, -2, -8, 4, -1};

    
    printf("Корреляция a и b: %.2f\n", corr(a, b));
    printf("Корреляция a и c: %.2f\n", corr(a, c));
    printf("Корреляция b и c: %.2f\n", corr(b, c));

    printf("Нормализованная корреляция a и b: %.2f\n", corr_normalization(a, b));
    printf("Нормализованная корреляция a и c: %.2f\n", corr_normalization(a, c));
    printf("Нормализованная корреляция b и c: %.2f\n", corr_normalization(b, c));


    return 0;
}





