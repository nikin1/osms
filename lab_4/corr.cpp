#include <iostream>
#include <vector>
#include <cmath>
#include <map>

#define LENGTH_SEQUENCE 31

using namespace std;

vector<int> create_gold_sequence(vector<int>& x, vector<int>& y) {
    vector<int> gold_sequence;
    int xor_shift_x, xor_shift_y;
    int last_bit_x, last_bit_y;
    for (int i = 0; i < LENGTH_SEQUENCE; i++){
        xor_shift_x = x[2] ^ x[4];
        xor_shift_y = y[2] ^ y[4];

        gold_sequence.push_back(x.back() ^ y.back());

        x.pop_back();
        y.pop_back();

        x.insert(x.begin(), xor_shift_x);
        y.insert(y.begin(), xor_shift_y);

    }

    return gold_sequence;
}

vector<int> shift_sequence(vector<int>& original_sequence, int shift) {
    int N = original_sequence.size();
    vector<int> shifted_sequence(N);

    for (int i = 0; i < N; i++){
        shifted_sequence[i] = original_sequence[(i - shift + N) % N];
    }

    return shifted_sequence;
}

double NormalizedCorrelation(vector<int>& x, vector<int>& y) {
    int N = x.size();
    double corr = 0;

    double sumXY = 0;
    double sumX2 = 0;
    double sumY2 = 0;

    for (int i = 0; i < N; i++){
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
        sumY2 += y[i] * y[i];
    }

    corr = sumXY / sqrt(sumX2 * sumY2);
    return corr;
}

double autoCorrelation(vector<int>& x, vector<int>& y) {
    int equal = 0, different = 0;
    double auto_corr;
    for (int i = 0; i < LENGTH_SEQUENCE; i++){
        if (x[i] == y[i]){
            equal += 1;
        }
        else{
            different += 1;
        }
    }

    auto_corr = (1.0 / LENGTH_SEQUENCE) * (equal - different);
    return auto_corr;
}

void print_table(vector<int>& sequence) {
    int N = sequence.size();
    cout << "Сдвиг\tПоследовательность\t\t\tАвтокорреляция" << endl;
    for (int i = 0; i < N + 1; i++){
        vector<int> shifted_sequence = shift_sequence(sequence, i);
        double corr = autoCorrelation(sequence, shifted_sequence);
        cout << i << "\t";
        for (int i : shifted_sequence){
            cout << i << " ";
        }
        cout << "\t";

        cout << corr << endl;
    }
}

bool balance(vector<int>& sequence) {
    int zeros = 0;
    int ones = 0;

    for (int i : sequence){
        if (i == 0){
            zeros++;
        }
        else {
            ones++;
        }
    }

    return abs(ones - zeros) <= 1;

}

map<int, int> cycle_lengths(vector<int>& sequence) {
    map<int, int> cycles;
    int count = 1;
    for (size_t i = 1; i < sequence.size(); i++) {
        if (sequence[i] == sequence[i - 1]) {
            count++;
        } else {
            cycles[count]++;
            count = 1;
        }
    }
    cycles[count]++;
    return cycles;
}

bool cycle_check(const map<int, int>& cycle) {
    int total_length = 0;
    int counter = 2;
    for (auto& pair : cycle){
        total_length += pair.second;
    }

    for (auto& pair : cycle){
        if ((total_length / pair.second != counter)){
            return 0;
        }
        if (pair.second != 1){
            counter *= 2;
        }
    }



    return 1;
}

bool check_autocorr(vector<int>& sequence) {
        for (int i = 1; i < LENGTH_SEQUENCE; i++){
        vector<int> shifted_sequence = shift_sequence(sequence, i);
        double corr = autoCorrelation(sequence, shifted_sequence);
        if (abs(corr) >= 0.1){
            return 0;
        }
    }
    return 1;
}


int main(){
    vector<int> x = {1, 0, 1, 0, 1};
    vector<int> y = {1, 0, 1, 1, 1};

    cout << "Последовательность Голда: ";
    vector<int> gold_sequence = create_gold_sequence(x, y);
    for (int i : gold_sequence){
        cout << i;
    }
    cout << endl;

    print_table(gold_sequence);

    x = {1, 0, 0, 0, 1};
    y = {1, 0, 0, 1, 0};
    cout << "Новая последовательность Голда: ";
    vector<int> new_gold_sequence = create_gold_sequence(x, y);
    for (int i : new_gold_sequence){
        cout << i;
    }
    cout << endl;

    double norm_corr = NormalizedCorrelation(gold_sequence, new_gold_sequence);
    cout << "Взаимная корреляция исходной и новой последовательности: " << norm_corr << endl;
    cout << endl;
    cout << "-------------------------------------------------------------------------" << endl;

    bool is_balanced = balance(gold_sequence);
    if (is_balanced == 1){
        cout << "Последовательность сбалансированна" << endl;
    }
    else {
        cout << "Последовательность не сбалансированна" << endl;
    }

    map<int, int> cycles = cycle_lengths(gold_sequence);
    cout << "Проверка на цикличность:\n";
    cout << "Длина (n):" << endl; 
    for (auto& pair : cycles) {
        cout << "\tn = " << pair.first << ": " << pair.second << endl;
    }

    bool is_cycle = cycle_check(cycles);
    if (is_cycle == 1){
        cout << "Последовательность циклична" << endl;
    }
    else {
        cout << "Последовательность циклична" << endl;
    }

    bool is_autocorr = check_autocorr(gold_sequence);
    if (is_autocorr == 1){
        cout << "Последовательность обладает свойством корреляции" << endl;
    }
    else{
        cout << "Последовательность не обладает свойством корреляции" << endl;
    }

    if (is_balanced && is_cycle && is_autocorr){
        cout << "Последовательность является последовательностью Голда" << endl;
    }
    else{
        cout << "Последовательность не является последовательностью Голда" << endl;
    }
}
