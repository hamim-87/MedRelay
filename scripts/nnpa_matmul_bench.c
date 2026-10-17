// Device-level NNPA test through IBM zDNN: matmul on the accelerator vs a plain CPU loop, with numerical-difference check.
// Build: gcc -O2 -o nnpa_matmul_bench nnpa_matmul_bench.c -lzdnn -lm      Usage: ./nnpa_matmul_bench M K N iters
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <zdnn.h>

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

int main(int argc, char **argv) {
    int M = argc > 1 ? atoi(argv[1]) : 256, K = argc > 2 ? atoi(argv[2]) : 256, N = argc > 3 ? atoi(argv[3]) : 256, iters = argc > 4 ? atoi(argv[4]) : 20;
    zdnn_init();
    int installed = zdnn_is_nnpa_installed();
    if (!installed) { printf("{\"nnpa_installed\": false}\n"); return 2; }
    float *A = malloc(sizeof(float) * M * K), *B = malloc(sizeof(float) * K * N), *C0 = calloc(M * N, sizeof(float)), *C1 = calloc(M * N, sizeof(float)), *bias = calloc(N, sizeof(float));
    srand(1);
    for (int i = 0; i < M * K; i++) A[i] = (rand() / (float)RAND_MAX) - 0.5f;
    for (int i = 0; i < K * N; i++) B[i] = (rand() / (float)RAND_MAX) - 0.5f;
    zdnn_tensor_desc pa, ta, pb, tb, pc, tc, po, to;
    zdnn_ztensor za, zb, zc, zo;
    zdnn_status st;
    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &pa, 1, M, K); zdnn_generate_transformed_desc(&pa, &ta);
    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &pb, 1, K, N); zdnn_generate_transformed_desc(&pb, &tb);
    zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &pc, 1, N);    zdnn_generate_transformed_desc(&pc, &tc);
    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &po, 1, M, N); zdnn_generate_transformed_desc(&po, &to);
    if ((st = zdnn_init_ztensor_with_malloc(&pa, &ta, &za)) != ZDNN_OK || (st = zdnn_init_ztensor_with_malloc(&pb, &tb, &zb)) != ZDNN_OK ||
        (st = zdnn_init_ztensor_with_malloc(&pc, &tc, &zc)) != ZDNN_OK || (st = zdnn_init_ztensor_with_malloc(&po, &to, &zo)) != ZDNN_OK) {
        printf("{\"error\": \"init_ztensor: %s\"}\n", zdnn_get_status_message(st)); return 3; }
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) { za.is_transformed = false; zb.is_transformed = false; zc.is_transformed = false; zdnn_transform_ztensor(&za, A); zdnn_transform_ztensor(&zb, B); zdnn_transform_ztensor(&zc, bias); }
    double transform_in_ms = (now_ms() - t0) / iters;
    st = zdnn_matmul_op(&za, &zb, &zc, MATMUL_OP_ADDITION, &zo);        // warm-up + status check
    if (st != ZDNN_OK) { printf("{\"error\": \"matmul_op: %s\"}\n", zdnn_get_status_message(st)); return 4; }
    t0 = now_ms();
    for (int i = 0; i < iters; i++) zdnn_matmul_op(&za, &zb, &zc, MATMUL_OP_ADDITION, &zo);
    double op_ms = (now_ms() - t0) / iters;
    zdnn_transform_origtensor(&zo, C1);
    t0 = now_ms();
    for (int i = 0; i < iters; i++) zdnn_transform_origtensor(&zo, C1);
    double transform_out_ms = (now_ms() - t0) / iters;
    int cpu_iters = (double)M * K * N > 3e8 ? 1 : (iters > 3 ? 3 : iters);
    t0 = now_ms();
    for (int it = 0; it < cpu_iters; it++)
        for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) { float a = A[i * K + k]; for (int j = 0; j < N; j++) C0[i * N + j] += (it == 0 ? a * B[k * N + j] : 0.0f); }
    double cpu_ms = (now_ms() - t0) / cpu_iters;
    double maxerr = 0, maxref = 0;
    for (int i = 0; i < M * N; i++) { double e = fabs((double)C0[i] - C1[i]); if (e > maxerr) maxerr = e; if (fabs(C0[i]) > maxref) maxref = fabs(C0[i]); }
    double flop = 2.0 * M * K * N;
    printf("{\"nnpa_installed\": true, \"zdnn_version\": \"%s\", \"M\": %d, \"K\": %d, \"N\": %d, \"iters\": %d, \"nnpa_op_ms\": %.4f, \"nnpa_transform_in_ms\": %.4f, \"nnpa_transform_out_ms\": %.4f, "
           "\"nnpa_total_ms\": %.4f, \"cpu_naive_c_ms\": %.4f, \"nnpa_op_gflops\": %.2f, \"cpu_naive_gflops\": %.2f, \"max_abs_err\": %.5f, \"max_abs_ref\": %.4f, \"status\": \"%s\"}\n",
           ZDNN_VERSION, M, K, N, iters, op_ms, transform_in_ms, transform_out_ms, op_ms + transform_in_ms + transform_out_ms, cpu_ms, flop / op_ms / 1e6, flop / cpu_ms / 1e6, maxerr, maxref,
           zdnn_get_status_message(ZDNN_OK));
    return 0;
}
