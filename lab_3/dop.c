#include <stdio.h>
#include <math.h>

#define LEN_LIST_CORR 100

int a = { 93, 14, 12, 64, 83, 45, 87, 44, 73, 67, 29, 13, 47, 96, 28, 61, 99, 49, 16, 86, 3, 31, 38, 51, 71, 17, 9, 25, 34, 30, 56, 37, 27, 84, 97, 63, 1, 54, 39, 6, 92, 85, 42, 36, 57, 91, 58, 98, 20, 66, 7, 23, 35, 46, 72, 5, 18, 2, 59, 10, 11, 81, 60, 82, 88, 75, 26, 79, 53, 95, 22, 41, 55, 19, 21, 43, 50, 89, 65, 76, 15, 48, 100, 74, 69, 77, 4, 90, 40, 62, 5, 2, 5, 6, 2, 0, 1, 2, 33, 12, 23, 64};
int b = { 52, 16, 77, 99, 95, 31, 30, 96, 29, 59, 90, 1, 50, 48, 68, 88, 62, 79, 14, 27, 3, 84, 46, 70, 87, 2, 21, 49, 17, 73, 34, 53, 37, 58, 36, 91, 26, 20, 60, 75, 93, 4, 8, 76, 92, 74, 35, 15, 78, 66, 13, 33, 83, 100, 25, 45, 51, 5, 40, 44, 81, 94, 43, 82, 9, 98, 61, 67, 86, 80, 7, 41, 42, 65, 63, 72, 38, 47, 71, 24, 89, 57, 11, 12, 23, 32, 64, 39, 55, 85, 10, 22, 28, 69, 6, 97, 19, 54, 56, 18};


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

// for (int i = 0; i < size; ++i) {
//     shiftedSequence[i] = sequence[(i + shift) % size];
// }

int main() {
    

    return 0;
}