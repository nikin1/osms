#include <iostream>
#include <cmath>

using namespace std;

#define LEN_LIST_CORR 8

int corr(int x[LEN_LIST_CORR], int y[LEN_LIST_CORR])
{
    int summa = 0;
    for (int i = 0; i < LEN_LIST_CORR; i++)
    {
        summa += x[i] * y[i];
    }

    return summa;
}


int corr_normalization(int x[LEN_LIST_CORR], int y[LEN_LIST_CORR])
{
    int summa = 0, summa_x = 0, summa_y = 0;

    for (int i = 0; i < LEN_LIST_CORR; i++)
    {
        summa += x[i] * y[i];
        summa_x += pow(x[i], 2);
        summa_y += pow(y[i], 2);
    }
    
    int res = summa / sqrt(summa_x * summa_y);

    return res;
}




int main() {

    cout << "Hello World!";

    int a[LEN_LIST_CORR] = {7, 3, 2, -2, -2, -4, 1, 5};
    int b[LEN_LIST_CORR] = {2, 1, 5, 0, -2, -3, 2, 4};
    int c[LEN_LIST_CORR] = {2, -1, 3, -9, -2, -8, 4, -1};

    






    return 0;
}





