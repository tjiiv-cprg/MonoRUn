void pnp_uncert(
    double* pts2d,  // n, 2
    double* pts3d,  // n, 3
    double* wgt2d,  // n, 2
    double* K,      // 3, 3
    double* init_pose,    // 4
    int* result_val,
    double* result_pose,  // 4
    double* result_cov, // 4, 4
    double* result_tr,
    int pn,
    double* clips   // 5
);

void pnp_noc_uncert(
    double* pts2d,          // n, 2
    double* pts3d,          // n, 3
    double* wgt2d,          // n, 2
    double* logdim,         // 3
    double* logdim_wgt,     // 3
    double* K,              // 3, 3
    double* init_dimpose,      // 7
    int* result_val,
    double* result_dimpose,    // 7
    int pn,
    double* clips,          // 5
    double delta
);

void pnp_noc_cov_uncert(
    double* pts2d,          // n, 2
    double* pts3d,          // n, 3
    double* wgt2d,          // n, 3 [wxx, wxy, wyy]
    double* logdim,         // 3
    double* logdim_wgt,     // 3
    double* K,              // 3, 3
    double* init_dimpose,      // 7
    int* result_val,
    double* result_dimpose,    // 7
    int pn,
    double* clips,          // 5
    double delta
);
