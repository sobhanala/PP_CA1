#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <chrono>
#include <immintrin.h> // AVX intrinsics for wider SIMD

using namespace std;

const float THRESHOLD = 1.5;

double ScalculateMean(const vector<double>& data) {
    double sum = accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

double ScalculateStdDev(const vector<double>& data, double mean) {
    double sum = 0.0;
    for (double x : data) {
        sum += (x - mean) * (x - mean);
    }
    return sqrt(sum / data.size());
}

vector<double> SfindOutliers(const vector<double>& data, double threshold = THRESHOLD) {
    vector<double> outliers;
    double mean = ScalculateMean(data);
    double std_dev = ScalculateStdDev(data, mean);

    cout << "mean: " << mean << "    " << "dev :   " << std_dev << endl;
    for (double x : data) {
        double z_score = (x - mean) / std_dev;
        if (abs(z_score) > threshold) {
            outliers.push_back(x);
        }
    }
    return outliers;
}


// SIMD mean and standard deviation
double calculateMean(const vector<double>& data) {
    __m256d sum_vec = _mm256_setzero_pd();
    size_t n = data.size();
    size_t i = 0;

    for (; i + 3 < n; i += 4) {
        __m256d val_vec = _mm256_load_pd(&data[i]);
        sum_vec = _mm256_add_pd(sum_vec, val_vec);
    }

    double sum[4];
    _mm256_store_pd(sum, sum_vec);
    double total_sum = sum[0] + sum[1] + sum[2] + sum[3];

    for (; i < n; i++) {
        total_sum += data[i];
    }

    return total_sum / n;
}

double calculateStdDev(const vector<double>& data, double mean) {
    __m256d sum_vec = _mm256_setzero_pd();
    __m256d mean_vec = _mm256_set1_pd(mean);
    size_t n = data.size();
    size_t i = 0;

    for (; i + 3 < n; i += 4) {
        __m256d val_vec = _mm256_load_pd(&data[i]);
        __m256d diff = _mm256_sub_pd(val_vec, mean_vec);
        __m256d sq_diff = _mm256_mul_pd(diff, diff);
        sum_vec = _mm256_add_pd(sum_vec, sq_diff);
    }

    double sum[4];
    _mm256_store_pd(sum, sum_vec);
    double total_sum = sum[0] + sum[1] + sum[2] + sum[3];

    for (; i < n; i++) {
        double d = data[i] - mean;
        total_sum += d * d;
    }

    return sqrt(total_sum / n);
}

vector<double> findOutliers(const vector<double>& data, double threshold = THRESHOLD) {
    vector<double> outliers;
    double mean = calculateMean(data);
    double std_dev = calculateStdDev(data, mean);

    __m256d mean_vec = _mm256_set1_pd(mean);
    __m256d std_dev_vec = _mm256_set1_pd(std_dev);
    __m256d threshold_vec = _mm256_set1_pd(threshold);
    size_t n = data.size();
    size_t i = 0;

    for (; i + 3 < n; i += 4) {
        __m256d val_vec = _mm256_load_pd(&data[i]);
        __m256d diff = _mm256_sub_pd(val_vec, mean_vec);
        __m256d z_score = _mm256_div_pd(diff, std_dev_vec);

        __m256d abs_z_score = _mm256_andnot_pd(_mm256_set1_pd(-0.0), z_score);
        __m256d mask = _mm256_cmp_pd(abs_z_score, threshold_vec, _CMP_GT_OQ);

        double result[4];
        _mm256_store_pd(result, mask);

        for (int j = 0; j < 4; ++j) {
            if (result[j] != 0.0) {
                outliers.push_back(data[i + j]);
            }
        }
    }

    for (; i < n; i++) {
        double z_score = (data[i] - mean) / std_dev;
        if (abs(z_score) > threshold) {
            outliers.push_back(data[i]);
        }
    }

    return outliers;
}

int main() {
    const int num_elements = 1 <<20; // Adjust to a higher power if needed for timing
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    vector<double> data(num_elements);
    for (int i = 0; i < num_elements; ++i) {
        data[i] = dis(gen);
    }

    const int num_runs = 100;
    double total_simd_time = 0.0;
    double total_scalar_time = 0.0;

    for (int j = 0; j < num_runs; ++j) {
        auto start = chrono::high_resolution_clock::now();
        vector<double> outliers = findOutliers(data);
        auto end = chrono::high_resolution_clock::now();
        total_simd_time += chrono::duration_cast<chrono::microseconds>(end - start).count();

        auto sstart = chrono::high_resolution_clock::now();
        vector<double> Soutliers = SfindOutliers(data);
        auto send = chrono::high_resolution_clock::now();
        total_scalar_time += chrono::duration_cast<chrono::microseconds>(send - sstart).count();
    }

    cout << "Average SIMD Time taken: " << total_simd_time / num_runs << " microseconds" << endl;
    cout << "Average Scalar Time taken: " << total_scalar_time / num_runs << " microseconds" << endl;
    cout << "Average SpeedUp: "<<(total_scalar_time / num_runs) / (total_simd_time / num_runs);
    return 0;
}
