#include "ysyx.h"
#include <am.h>
#include <klib-macros.h>
#include <klib.h>
#include <stdint.h>
#include "audio_assets.h"

// --- Fixed Point Configuration ---
#define Q15_SHIFT 15
#define WINDOW_SIZE 128
#define STEP_SIZE 64
#define SAMPLE_RATE 8000
#define NUM_MEL_FILTERS 8
#define VAD_THRESHOLD 10000000 

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

FeatureMatrix *extract_features_fixed(const int16_t *pcm_data, int num_samples) {
    int num_frames = (num_samples - WINDOW_SIZE) / STEP_SIZE + 1;
    if (num_frames <= 0) return NULL;
    int32_t *temp_features = (int32_t *)malloc(num_frames * NUM_MEL_FILTERS * sizeof(int32_t));
    if (!temp_features) return NULL;
    int valid_frames = 0;
    for (int i = 0; i < num_frames; i++) {
        const int16_t *window = &pcm_data[i * STEP_SIZE];
        int64_t energy = 0;
        for (int j = 0; j < WINDOW_SIZE; j++) energy += (int64_t)window[j] * window[j];
        if (energy > VAD_THRESHOLD) {
            int32_t power_spectrum[WINDOW_SIZE / 2 + 1], mel_energy[NUM_MEL_FILTERS];
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
    FeatureMatrix *fm = (FeatureMatrix *)malloc(sizeof(FeatureMatrix));
    if (!fm) { free(temp_features); return NULL; }
    fm->rows = valid_frames; fm->cols = NUM_MEL_FILTERS;
    if (valid_frames == 0) { fm->data = NULL; free(temp_features); return fm; }
    fm->data = (int32_t *)malloc(valid_frames * NUM_MEL_FILTERS * sizeof(int32_t));
    if (!fm->data) { free(fm); free(temp_features); return NULL; }
    memcpy(fm->data, temp_features, valid_frames * NUM_MEL_FILTERS * sizeof(int32_t));
    free(temp_features);
    if (valid_frames > 0) {
        int32_t *mean = (int32_t *)malloc(NUM_MEL_FILTERS * sizeof(int32_t));
        if (mean) {
            memset(mean, 0, NUM_MEL_FILTERS * sizeof(int32_t));
            for (int i = 0; i < valid_frames; i++) 
                for (int j = 0; j < NUM_MEL_FILTERS; j++) mean[j] += fm->data[i * NUM_MEL_FILTERS + j];
            for (int j = 0; j < NUM_MEL_FILTERS; j++) mean[j] /= valid_frames;
            for (int i = 0; i < valid_frames; i++) 
                for (int j = 0; j < NUM_MEL_FILTERS; j++) fm->data[i * NUM_MEL_FILTERS + j] -= mean[j];
            free(mean);
        }
    }
    return fm;
}

int32_t compute_dtw_distance_fixed(FeatureMatrix *feat1, FeatureMatrix *feat2) {
    int n = feat1->rows, m = feat2->rows, d = feat1->cols;
    if (n == 0 || m == 0) return 2147483647; 
    int32_t **acc_cost = (int32_t **)malloc(n * sizeof(int32_t *));
    if (!acc_cost) return 2147483647;
    for (int i = 0; i < n; i++) acc_cost[i] = (int32_t *)malloc(m * sizeof(int32_t));
    int32_t dist = 0;
    for (int k = 0; k < d; k++) dist += abs(feat1->data[0*d+k] - feat2->data[0*d+k]);
    acc_cost[0][0] = dist;
    for (int i = 1; i < n; i++) {
        dist = 0; for (int k = 0; k < d; k++) dist += abs(feat1->data[i*d+k] - feat2->data[0*d+k]);
        acc_cost[i][0] = dist + acc_cost[i-1][0];
    }
    for (int j = 1; j < m; j++) {
        dist = 0; for (int k = 0; k < d; k++) dist += abs(feat1->data[0*d+k] - feat2->data[j*d+k]);
        acc_cost[0][j] = dist + acc_cost[0][j-1];
    }
    for (int i = 1; i < n; i++) {
        for (int j = 1; j < m; j++) {
            dist = 0; for (int k = 0; k < d; k++) dist += abs(feat1->data[i*d+k] - feat2->data[j*d+k]);
            int32_t min_prev = acc_cost[i-1][j];
            if (acc_cost[i][j-1] < min_prev) min_prev = acc_cost[i][j-1];
            if (acc_cost[i-1][j-1] < min_prev) min_prev = acc_cost[i-1][j-1];
            acc_cost[i][j] = dist + min_prev;
        }
    }
    int32_t result = acc_cost[n-1][m-1];
    for (int i = 0; i < n; i++) free(acc_cost[i]);
    free(acc_cost);
    return result;
}

void free_feature_matrix(FeatureMatrix *fm) { if (fm) { if (fm->data) free(fm->data); free(fm); } }

// --- ADC ---
void adc_init(void) {
    printf("GPIO INIT:\n");
    GPIO_1_REG_IOFCFG |= (uint32_t)(1 << 0);
    GPIO_1_REG_IOFCFG |= (uint32_t)(1 << 1);
    GPIO_1_REG_IOFCFG |= (uint32_t)(1 << 2);
    GPIO_1_REG_IOFCFG |= (uint32_t)(1 << 3);
    GPIO_1_REG_IOFCFG |= (uint32_t)(1 << 4);
    GPIO_1_REG_IOFCFG |= (uint32_t)(1 << 5);
    GPIO_1_REG_PINMUX = 0; // FUNC0
    printf("GPIO_1_PADDIR: %08x\n", GPIO_1_REG_PADDIR);
    printf("GPIO_1_IOFCFG: %08x\n", GPIO_1_REG_IOFCFG);
    printf("GPIO_1_PINMUX: %08x\n", GPIO_1_REG_PINMUX);
    printf("GPIO INIT DONE\n");

    printf("SPI INIT:\n");
    SPI1_REG_STATUS = (uint32_t)0b10000;
    SPI1_REG_STATUS = (uint32_t)0b00000;
    SPI1_REG_INTCFG = (uint32_t)0b00000;
    SPI1_REG_DUM = (uint32_t)0;
    SPI1_REG_CLKDIV = (uint32_t)15; // sck = apb_clk/2(div+1) 100MHz/2 = 50MHz
    printf("SPI1_STATUS: %08x\n", SPI1_REG_STATUS);
    printf("SPI1_CLKDIV: %08x\n", SPI1_REG_CLKDIV);
    printf("SPI1_INTCFG: %08x\n", SPI1_REG_INTCFG);
    printf("SPI1_DUM: %08x\n", SPI1_REG_DUM); 
    printf("SPI INIT DONE\n");

}

uint16_t SPI_Get(void){
    SPI1_REG_LEN = 0x150000;
    //启动传输
    SPI1_REG_STATUS =0x10;
    SPI1_REG_STATUS =1+256;
    while ((SPI1_REG_STATUS & 0x000000FF) !=1){
    }
    return (uint16_t)(SPI1_REG_RXFIFO&0xFFFF);
}

// --- Main ---
int main(const char *args) {
    printf("Audio DTW Test (Fixed Point)\n");
    
    // 1. Prepare Reference (open_pcm, close_pcm)
    int16_t *ref_open = (int16_t *)malloc(open_pcm_len);
    int16_t *ref_close = (int16_t *)malloc(close_pcm_len);
    if (ref_open) memcpy(ref_open, open_pcm, open_pcm_len);
    if (ref_close) memcpy(ref_close, close_pcm, close_pcm_len);

    // 2. Prepare Test (open_test_pcm, close_test_pcm)
    int16_t *test_open = (int16_t *)malloc(open_test_pcm_len);
    int16_t *test_close = (int16_t *)malloc(close_test_pcm_len);
    if (test_open) memcpy(test_open, open_test_pcm, open_test_pcm_len);
    if (test_close) memcpy(test_close, close_test_pcm, close_test_pcm_len);

    if (!ref_open || !ref_close || !test_open || !test_close) { printf("Memory Fail\n"); return 1; }

    // 3. Feature Extraction
    FeatureMatrix *f_ref_open = extract_features_fixed(ref_open, open_pcm_len/2);
    FeatureMatrix *f_ref_close = extract_features_fixed(ref_close, close_pcm_len/2);
    FeatureMatrix *f_test_open = extract_features_fixed(test_open, open_test_pcm_len/2);
    FeatureMatrix *f_test_close = extract_features_fixed(test_close, close_test_pcm_len/2);

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
    free(test_open); free(test_close);
    free_feature_matrix(f_test_open); free_feature_matrix(f_test_close);

    // 6. ADC get
    GPIO_SetDir_Num(1,7,0);
    uint16_t adc_data[16 * 1000] = {0};
    adc_init();
    while(1){
        // wait key press
        while(GPIO_GetVal_Num(1,7) == 0);
        // wait key release
        while(GPIO_GetVal_Num(1,7) == 1);
        printf("Key Pressed\n");

        for(uint32_t i=0; i<16 * 1000; i++){
            adc_data[i] = SPI_Get();
        }
        for(uint32_t i=0; i<16 * 1000; i++){
            printf("ADC: %d\n", adc_data[i]);
        }
    }
}
