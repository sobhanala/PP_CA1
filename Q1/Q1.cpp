#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <iostream>
#include "emmintrin.h"


void serial(cv::Mat image, cv::Mat logo) {
    cv::Mat blended_roi_serial;

    int logo_height = logo.rows;
    int logo_width = logo.cols;

    cv::Mat roi = image(cv::Rect(0, 0, logo_width, logo_height));

    for (int y = 0; y < logo_height; ++y) {
        for (int x = 0; x < logo_width; ++x) {
            cv::Vec3b img_pixel = roi.at<cv::Vec3b>(y, x);
            cv::Vec3b logo_pixel = logo.at<cv::Vec3b>(y, x);

            for (int c = 0; c < 3; ++c) {
                int blended_value = static_cast<int>(img_pixel[c] + 0.625 * logo_pixel[c]);
                img_pixel[c] = cv::saturate_cast<uchar>(blended_value);
            }
            roi.at<cv::Vec3b>(y, x) = img_pixel;
        }
    }
    image(cv::Rect(0, 0, logo_width, logo_height)) = roi;
}


void parallel_with_SIMD(cv::Mat image, cv::Mat logo, float alpha = 0.625) {
int logo_height = logo.rows;
    int logo_width = logo.cols;
    cv::Mat roi = image(cv::Rect(0, 0, logo_width, logo_height));

    __m128 alpha_vec = _mm_set1_ps(alpha);
    __m128 one_minus_alpha_vec = _mm_set1_ps(1.0f);

    for (int y = 0; y < logo_height; ++y) {
        for (int x = 0; x < logo_width; x += 4) {
            __m128i image_pixels = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&roi.at<cv::Vec3b>(y, x)));
            __m128i logo_pixels = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&logo.at<cv::Vec3b>(y, x)));

            __m128i image_lo = _mm_unpacklo_epi8(image_pixels, _mm_setzero_si128());
            __m128i logo_lo = _mm_unpacklo_epi8(logo_pixels, _mm_setzero_si128());

            __m128 image_lo_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(image_lo, _mm_setzero_si128()));
            __m128 image_hi_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(image_lo, _mm_setzero_si128()));
            __m128 logo_lo_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(logo_lo, _mm_setzero_si128()));
            __m128 logo_hi_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(logo_lo, _mm_setzero_si128()));



            __m128 blended_lo_f = _mm_add_ps(_mm_mul_ps(image_lo_f, one_minus_alpha_vec), _mm_mul_ps(logo_lo_f, alpha_vec));
            __m128 blended_hi_f = _mm_add_ps(_mm_mul_ps(image_hi_f, one_minus_alpha_vec), _mm_mul_ps(logo_hi_f, alpha_vec));

            __m128i blended_lo_i = _mm_cvtps_epi32(blended_lo_f);
            __m128i blended_hi_i = _mm_cvtps_epi32(blended_hi_f);

            __m128i blended_16bit = _mm_packs_epi32(blended_lo_i, blended_hi_i);
            __m128i blended_8bit = _mm_packus_epi16(blended_16bit, _mm_setzero_si128());

            _mm_storel_epi64(reinterpret_cast<__m128i*>(&roi.at<cv::Vec3b>(y, x)), blended_8bit);
        }
    }
}

int main() {
    cv::Mat const image = cv::imread("F:/Program Files (x86)/MMD.Soor/University/Term 7/PP/CAs/CA1/PP_CA1/assets/Q1/front.png");
    cv::Mat const logo = cv::imread("F:/Program Files (x86)/MMD.Soor/University/Term 7/PP/CAs/CA1/PP_CA1/assets/Q1/logo.png");

    cv::Mat serialImg = image.clone();
    cv::Mat serialLogo = logo.clone();
    cv::Mat parallelImg = image.clone();
    cv::Mat parallelLogo = logo.clone();

    auto start_serial = std::chrono::high_resolution_clock::now();
    serial(serialImg, serialLogo);
    auto end_serial = std::chrono::high_resolution_clock::now();
    cv::imshow("Serial Blend", serialImg);
    std::chrono::duration<double> serial_duration = end_serial - start_serial;

    auto start_parallel = std::chrono::high_resolution_clock::now();
    parallel_with_SIMD(parallelImg, parallelLogo);
    auto end_parallel = std::chrono::high_resolution_clock::now();
    cv::imshow("Blended Image", parallelImg);
    std::chrono::duration<double> parallel_duration = end_parallel - start_parallel;

    std::cout << "Serial Processing Time: " << serial_duration.count() * 1000 << " milliseconds." << std::endl;
    std::cout << "Parallel Processing Time: " << parallel_duration.count() * 1000 << " milliseconds." << std::endl;

    double speedup = serial_duration.count() / parallel_duration.count();
    std::cout << "Speedup: " << speedup << std::endl;

    cv::waitKey(0);

    return 0;
}
