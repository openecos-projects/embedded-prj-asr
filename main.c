/*
 * FBank Algorithm Ported to Main
 * 
 * Ported from: testdir/fbank/fbnk.c
 * Target: main.c
 * 
 * Changes:
 * - Removed dependency on ysyx.h, am.h, klib.h
 * - Implemented simple static memory allocator (my_malloc) to replace malloc
 * - Optimized DTW to use O(M) memory instead of O(N*M)
 * - Adapted to use audio_assets.h directly without copying to RAM
 * - Using standard board.h macros for hardware access (where available)
 */

#include "main.h"
#include "board.h"
#include "audio_assets.h"
#include <stdint.h>
#include <string.h> // for memcpy, memset
#include <stdio.h>  // for printf

// --- Configuration ---
#define Q15_SHIFT 15
#define WINDOW_SIZE 128
#define STEP_SIZE 64
#define SAMPLE_RATE 8000
#define NUM_MEL_FILTERS 8
#define VAD_THRESHOLD 10000000 

// --- Memory Management ---
// Simple bump allocator since we don't have standard malloc
#define HEAP_SIZE (64 * 1024) 
static uint8_t heap_memory[HEAP_SIZE] __attribute__((aligned(4)));
static size_t heap_offset = 0;

static void *my_malloc(size_t size) {
    // Align to 4 bytes
    size = (size + 3) & ~3;
    if (heap_offset + size > HEAP_SIZE) {
        printf("ERR: Out of memory! req=%d, avail=%d\n", (int)size, (int)(HEAP_SIZE - heap_offset));
        return NULL;
    }
    void *ptr = &heap_memory[heap_offset];
    heap_offset += size;
    return ptr;
}

static void my_free(void *ptr) {
    // No-op for simple bump allocator
    (void)ptr;
}

static void reset_heap(void) {
    heap_offset = 0;
}

static int abs(int x) {
    return (x < 0) ? -x : x;
}

// --- Data Structures ---
typedef struct {
    int32_t real;
    int32_t imag;
} Complex32;

typedef struct {
    int32_t *data; // Q16.16
    int rows;
    int cols;
} FeatureMatrix;

// --- Lookup Tables ---
const int16_t SIN_TABLE_128[65] = {
    0, 1608, 3212, 4808, 6393, 7962, 9512, 11039, 12539, 14010, 15446, 16846, 18204, 19519, 20787, 22005, 
    23170, 24279, 25329, 26319, 27245, 28105, 28898, 29621, 30273, 30852, 31356, 31785, 32137, 32412, 32609, 32728, 
    32767, 32728, 32609, 32412, 32137, 31785, 31356, 30852, 30273, 29621, 28898, 28105, 27245, 26319, 25329, 24279, 
    23170, 22005, 20787, 19519, 18204, 16846, 15446, 14010, 12539, 11039, 9512, 7962, 6393, 4808, 3212, 1608, 0
};

static int16_t get_sin_q15(int k) {
    k = k % 128;
    if (k < 0) k += 128;
    if (k <= 64) return SIN_TABLE_128[k];
    else return -SIN_TABLE_128[k - 64];
}

static int16_t get_cos_q15(int k) {
    return get_sin_q15(k + 32);
}

// --- Math Helpers ---
static int32_t ilog2(uint32_t x) {
    if (x == 0) return -2147483648; 
    int32_t msb = 0;
    if (x >= 0x10000) { x >>= 16; msb += 16; }
    if (x >= 0x100) { x >>= 8; msb += 8; }
    if (x >= 0x10) { x >>= 4; msb += 4; }
    if (x >= 0x4) { x >>= 2; msb += 2; }
    if (x >= 0x2) { x >>= 1; msb += 1; }
    return (msb << 16); 
}

static int32_t iln_q16(uint32_t x) {
    int32_t log2_val = ilog2(x); 
    int64_t temp = (int64_t)log2_val * 45426;
    return (int32_t)(temp >> 16);
}

// --- Algorithms ---
static void fft_fixed(Complex32 *x, int n) {
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j) {
            Complex32 temp = x[i];
            x[i] = x[j];
            x[j] = temp;
        }
        int k = n / 2;
        while (k <= j) { j -= k; k /= 2; }
        j += k;
    }
    for (int len = 2; len <= n; len <<= 1) {
        int half_len = len >> 1;
        int step = 128 / len;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < half_len; k++) {
                int table_idx = k * step;
                int16_t cos_val = get_cos_q15(table_idx);
                int16_t sin_val = get_sin_q15(table_idx);
                Complex32 odd = x[i + k + half_len];
                Complex32 even = x[i + k];
                int32_t t_real = ((int64_t)cos_val * odd.real + (int64_t)sin_val * odd.imag) >> 15;
                int32_t t_imag = ((int64_t)cos_val * odd.imag - (int64_t)sin_val * odd.real) >> 15;
                x[i + k + half_len].real = even.real - t_real;
                x[i + k + half_len].imag = even.imag - t_imag;
                x[i + k].real = even.real + t_real;
                x[i + k].imag = even.imag + t_imag;
            }
        }
    }
}

static void compute_fft_power(const int16_t *window, int32_t *power_spectrum) {
    Complex32 x[WINDOW_SIZE];
    for (int i = 0; i < WINDOW_SIZE; i++) { x[i].real = (int32_t)window[i]; x[i].imag = 0; }
    fft_fixed(x, WINDOW_SIZE);
    for (int i = 0; i <= WINDOW_SIZE / 2; i++) {
        int64_t p = (int64_t)x[i].real * x[i].real + (int64_t)x[i].imag * x[i].imag;
        power_spectrum[i] = (int32_t)(p >> 10); 
    }
}

const int MEL_BIN_POINTS[10] = {0, 6, 12, 19, 27, 35, 43, 51, 58, 64};

static void apply_mel_filters_fixed(const int32_t *power_spectrum, int32_t *mel_energy) {
    for (int m = 1; m <= NUM_MEL_FILTERS; m++) {
        int f_m_minus = MEL_BIN_POINTS[m - 1], f_m = MEL_BIN_POINTS[m], f_m_plus = MEL_BIN_POINTS[m + 1];
        int64_t sum = 0;
        for (int k = f_m_minus; k < f_m; k++) {
            int32_t weight = ((k - f_m_minus) << 15) / (f_m - f_m_minus);
            sum += (int64_t)power_spectrum[k] * weight;
        }
        for (int k = f_m; k < f_m_plus; k++) {
            int32_t weight = ((f_m_plus - k) << 15) / (f_m_plus - f_m);
            sum += (int64_t)power_spectrum[k] * weight;
        }
        mel_energy[m - 1] = (int32_t)(sum >> 15);
    }
}

static FeatureMatrix *extract_features_fixed(const int16_t *pcm_data, int num_samples) {
    int num_frames = (num_samples - WINDOW_SIZE) / STEP_SIZE + 1;
    if (num_frames <= 0) return NULL;
    
    // Allocate temp storage from our static heap
    int32_t *temp_features = (int32_t *)my_malloc(num_frames * NUM_MEL_FILTERS * sizeof(int32_t));
    if (!temp_features) return NULL;
    
    int valid_frames = 0;
    for (int i = 0; i < num_frames; i++) {
        const int16_t *window = &pcm_data[i * STEP_SIZE];
        int64_t energy = 0;
        for (int j = 0; j < WINDOW_SIZE; j++) energy += (int64_t)window[j] * window[j];
        
        if (energy > VAD_THRESHOLD) {
            int32_t power_spectrum[WINDOW_SIZE / 2 + 1];
            int32_t mel_energy[NUM_MEL_FILTERS];
            
            compute_fft_power(window, power_spectrum);
            apply_mel_filters_fixed(power_spectrum, mel_energy);
            
            for (int k = 0; k < NUM_MEL_FILTERS; k++) {
                int32_t val = mel_energy[k];
                if (val <= 0) val = 1; 
                temp_features[valid_frames * NUM_MEL_FILTERS + k] = iln_q16(val);
            }
            valid_frames++;
        }
    }
    
    FeatureMatrix *fm = (FeatureMatrix *)my_malloc(sizeof(FeatureMatrix));
    if (!fm) { my_free(temp_features); return NULL; }
    
    fm->rows = valid_frames; 
    fm->cols = NUM_MEL_FILTERS;
    
    if (valid_frames == 0) { 
        fm->data = NULL; 
        my_free(temp_features); 
        return fm; 
    }
    
    fm->data = (int32_t *)my_malloc(valid_frames * NUM_MEL_FILTERS * sizeof(int32_t));
    if (!fm->data) { my_free(fm); my_free(temp_features); return NULL; }
    
    memcpy(fm->data, temp_features, valid_frames * NUM_MEL_FILTERS * sizeof(int32_t));
    my_free(temp_features); // Free temp storage
    
    if (valid_frames > 0) {
        int32_t *mean = (int32_t *)my_malloc(NUM_MEL_FILTERS * sizeof(int32_t));
        if (mean) {
            memset(mean, 0, NUM_MEL_FILTERS * sizeof(int32_t));
            for (int i = 0; i < valid_frames; i++) 
                for (int j = 0; j < NUM_MEL_FILTERS; j++) mean[j] += fm->data[i * NUM_MEL_FILTERS + j];
            
            for (int j = 0; j < NUM_MEL_FILTERS; j++) mean[j] /= valid_frames;
            
            for (int i = 0; i < valid_frames; i++) 
                for (int j = 0; j < NUM_MEL_FILTERS; j++) fm->data[i * NUM_MEL_FILTERS + j] -= mean[j];
            
            my_free(mean);
        }
    }
    return fm;
}

static int32_t compute_dtw_distance_fixed(FeatureMatrix *feat1, FeatureMatrix *feat2) {
    int n = feat1->rows, m = feat2->rows, d = feat1->cols;
    if (n == 0 || m == 0) return 2147483647; 
    
    // Optimized DTW using O(2*M) memory instead of O(N*M)
    int32_t *row_prev = (int32_t *)my_malloc(m * sizeof(int32_t));
    int32_t *row_curr = (int32_t *)my_malloc(m * sizeof(int32_t));
    
    if (!row_prev || !row_curr) return 2147483647;
    
    // Initialization for i=0
    int32_t dist = 0;
    for (int k = 0; k < d; k++) dist += abs(feat1->data[0*d+k] - feat2->data[0*d+k]);
    row_prev[0] = dist;
    
    for (int j = 1; j < m; j++) {
        dist = 0; 
        for (int k = 0; k < d; k++) dist += abs(feat1->data[0*d+k] - feat2->data[j*d+k]);
        row_prev[j] = dist + row_prev[j-1];
    }
    
    // DP Loop
    for (int i = 1; i < n; i++) {
        // j=0 case
        dist = 0;
        for (int k = 0; k < d; k++) dist += abs(feat1->data[i*d+k] - feat2->data[0*d+k]);
        row_curr[0] = dist + row_prev[0];
        
        for (int j = 1; j < m; j++) {
            dist = 0;
            for (int k = 0; k < d; k++) dist += abs(feat1->data[i*d+k] - feat2->data[j*d+k]);
            
            int32_t min_prev = row_prev[j];       // top
            if (row_curr[j-1] < min_prev) min_prev = row_curr[j-1]; // left
            if (row_prev[j-1] < min_prev) min_prev = row_prev[j-1]; // diag
            
            row_curr[j] = dist + min_prev;
        }
        
        // Swap pointers
        int32_t *temp = row_prev;
        row_prev = row_curr;
        row_curr = temp;
    }
    
    int32_t result = row_prev[m-1];
    
    my_free(row_prev);
    my_free(row_curr);
    
    return result;
}

static void free_feature_matrix(FeatureMatrix *fm) { 
    if (fm) { 
        if (fm->data) my_free(fm->data); 
        my_free(fm); 
    } 
}

#define CS(x) gpio_set_level(GPIO_NUM_2, x)
#define SCK(x) gpio_set_level(GPIO_NUM_1, x)
#define MISO()  gpio_get_level(GPIO_NUM_3)

#define KEY1()  gpio_get_level(GPIO_NUM_14)
#define KEY2()  gpio_get_level(GPIO_NUM_13)

static int16_t adc_data[16000];

// --- ADC/SPI Stub ---
// Original code used direct register access which might not be portable or available here.
// We provide a placeholder.
uint16_t SPI_Get(void) {
    REG_QSPI_0_LEN = 0x150000;
    //启动传输
    REG_QSPI_0_STATUS =0x10;
    REG_QSPI_0_STATUS =1+256;
    int temp;
    while ((REG_QSPI_0_STATUS & 0x000000FF) !=1){
    }
    return (uint16_t)(REG_QSPI_0_RXFIFO&0xFFFF);
}
void ADC_read(void) {
    for (int i = 0; i < 16000; i++) {
        adc_data[i] = SPI_Get()-16384;
    }
}
void adc_init(void) {
    // Placeholder for GPIO/SPI initialization
   qspi_config_t qspi_config = {
        .clkdiv = 16,
    };
    qspi_init(&qspi_config);
        // 配置GPIO0和GPIO1为输出模式
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << GPIO_NUM_1) | (1ULL << GPIO_NUM_2),
        .mode = GPIO_MODE_OUTPUT,
    };
    gpio_config(&io_conf);
    // 配置GPIO2为输入模式
    gpio_config_t input_conf = {
        .pin_bit_mask = (1ULL << GPIO_NUM_3) | (1ULL << GPIO_NUM_13) | (1ULL << GPIO_NUM_14),
        .mode = GPIO_MODE_INPUT,
    };
    gpio_config(&input_conf);
    CS(1);
    SCK(0);
}

// --- Main ---
void main(void) {
    sys_uart_init();
    printf("Audio DTW Test (Fixed Point) - Ported\n");
    
    // Reset our static heap
    reset_heap();
    
    // 1. Prepare Reference
    // Note: In audio_assets.h, these are uint8_t arrays (byte stream), 
    // but the algorithm expects int16_t (PCM samples).
    // We cast them directly. Ensure endianness matches (usually Little Endian).
    const int16_t *ref_open = (const int16_t *)open_pcm;
    const int16_t *ref_close = (const int16_t *)close_pcm;
    
    // 2. Prepare Test
    const int16_t *test_open = (const int16_t *)open_test_pcm;
    const int16_t *test_close = (const int16_t *)close_test_pcm;

    // 3. Feature Extraction
    // Lengths in audio_assets are in bytes, so divide by 2 for int16 samples
    printf("Extracting features...\n");
    FeatureMatrix *f_ref_open = extract_features_fixed(ref_open, open_pcm_len/2);
    FeatureMatrix *f_ref_close = extract_features_fixed(ref_close, close_pcm_len/2);
    FeatureMatrix *f_test_open = extract_features_fixed(test_open, open_test_pcm_len/2);
    FeatureMatrix *f_test_close = extract_features_fixed(test_close, close_test_pcm_len/2);

    if (!f_ref_open || !f_ref_close || !f_test_open || !f_test_close) {
        printf("Feature Extraction Failed (Out of Memory?)\n");
        return;
    }

    // 4. Matching Logic
    printf("\nTest 1: Matching open_test_pcm\n");
    int32_t d1_open = compute_dtw_distance_fixed(f_test_open, f_ref_open);
    int32_t d1_close = compute_dtw_distance_fixed(f_test_open, f_ref_close);
    printf("Dist to Open: %d, Dist to Close: %d -> Result: %s\n", d1_open, d1_close, (d1_open < d1_close ? "OPEN" : "CLOSE"));

    printf("\nTest 2: Matching close_test_pcm\n");
    int32_t d2_open = compute_dtw_distance_fixed(f_test_close, f_ref_open);
    int32_t d2_close = compute_dtw_distance_fixed(f_test_close, f_ref_close);
    printf("Dist to Open: %d, Dist to Close: %d -> Result: %s\n", d2_open, d2_close, (d2_close < d2_open ? "CLOSE" : "OPEN"));

    // Cleanup
    free_feature_matrix(f_ref_open);
    free_feature_matrix(f_ref_close);
    free_feature_matrix(f_test_open);
    free_feature_matrix(f_test_close);

    printf("\nTests Completed.\n");
    
    // 6. ADC Loop (Optional/Placeholder)
    
    adc_init();
    while (1)
    {
        if (KEY1() == 0) {
            printf("KEY1 pressed\n");
            f_test_open = extract_features_fixed((const int16_t *)adc_data, 16000);
            d1_open = compute_dtw_distance_fixed(f_test_open, f_ref_open);
            d1_close = compute_dtw_distance_fixed(f_test_open, f_ref_close);
            printf("Dist to Open: %d, Dist to Close: %d -> Result: %s\n", d1_open, d1_close, (d1_open < d1_close ? "OPEN" : "CLOSE"));
        }
        if (KEY2() == 0) {
            printf("KEY2 pressed\n");
            ADC_read();
            printf("ADC OK");
        }
    }
}
