// #include <iostream>
#include <ceres/rotation.h>
#include <ceres/ceres.h>
#include <algorithm>
// #include <random>

using namespace std;

struct ReprojectionErrorArray
{
    ReprojectionErrorArray(
        double x2d, double y2d,
        double x3d, double y3d, double z3d,
        double wxx, double wyy,
        double fx, double fy, double cx, double cy,
        double z_min, double u_min, double u_max,
        double v_min, double v_max
        ) : 
        x2d(x2d), y2d(y2d), x3d(x3d), y3d(y3d), z3d(z3d),
        fx(fx), fy(fy), cx(cx), cy(cy), wxx(wxx), wyy(wyy),
        z_min(z_min), u_min(u_min), u_max(u_max),
        v_min(v_min), v_max(v_max) {}

    template <typename T>
    bool operator()(const T *const pose, T *residuals) const
    {
        T pts3d[] = {T(x3d), T(y3d), T(z3d)};
        T r_vec[] = {T(0), pose[0], T(0)};
        T trans_pts3d[3];
        T clips[] = {T(z_min), T(u_min), T(u_max), T(v_min), T(v_max)};
        ceres::AngleAxisRotatePoint(r_vec, pts3d, trans_pts3d);
        trans_pts3d[0] += pose[1];
        trans_pts3d[1] += pose[2];
        trans_pts3d[2] += pose[3];

        trans_pts3d[2] = max(trans_pts3d[2], clips[0]);

        T proj_x = T(fx) * trans_pts3d[0] / trans_pts3d[2] + T(cx);
        T proj_y = T(fy) * trans_pts3d[1] / trans_pts3d[2] + T(cy);

        proj_x = (proj_x < clips[1]) ? clips[1] : (proj_x > clips[2]) ? clips[2] : proj_x;
        proj_y = (proj_y < clips[3]) ? clips[3] : (proj_y > clips[4]) ? clips[4] : proj_y;

        T diff_x = proj_x - T(x2d);
        T diff_y = proj_y - T(y2d);

        residuals[0] = T(wxx) * diff_x;
        residuals[1] = T(wyy) * diff_y;
        // cout<<residuals[0]<<" "<<residuals[1]<<endl;
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction *Create(
        double x2d, double y2d,
        double x3d, double y3d, double z3d,
        double wxx, double wyy,
        double fx, double fy, double cx, double cy,
        double z_min, double u_min, double u_max, double v_min, double v_max)
    {
        // cout<<wxx<<" "<<wyy<<endl;
        return (new ceres::AutoDiffCostFunction<ReprojectionErrorArray, 2, 4>(
            new ReprojectionErrorArray(x2d, y2d, x3d, y3d, z3d, wxx, wyy, fx, fy, cx, cy,
                                       z_min, u_min, u_max, v_min, v_max)));
    }

    double x2d, y2d;
    double x3d, y3d, z3d;
    double fx, fy;
    double cx, cy;
    double wxx, wyy;
    double z_min, u_min, u_max, v_min, v_max;
};


struct DimErrorArray
{
    DimErrorArray(
        double logl, double logh, double logw, 
        double logl_wgt, double logh_wgt, double logw_wgt 
        ) : 
        logl(logl), logh(logh), logw(logw), 
        logl_wgt(logl_wgt), logh_wgt(logh_wgt), logw_wgt(logw_wgt) {}

    template <typename T>
    bool operator()(const T *const dimpose, T *residuals) const
    {
        residuals[0] = T(logl_wgt) * (dimpose[0] - T(logl));
        residuals[1] = T(logh_wgt) * (dimpose[1] - T(logh));
        residuals[2] = T(logw_wgt) * (dimpose[2] - T(logw));
        return true;
    }

    static ceres::CostFunction *Create(
        double logl, double logh, double logw, 
        double logl_wgt, double logh_wgt, double logw_wgt)
    {
        return (new ceres::AutoDiffCostFunction<DimErrorArray, 3, 7>(
            new DimErrorArray(logl, logh, logw, logl_wgt, logh_wgt, logw_wgt)));
    }

    double logl, logh, logw, logl_wgt, logh_wgt, logw_wgt;
};


struct NocReprojectionErrorArray
{
    NocReprojectionErrorArray(
        double x2d, double y2d,
        double x3d, double y3d, double z3d,
        double wxx, double wyy,
        double fx, double fy, double cx, double cy,
        double z_min, double u_min, double u_max,
        double v_min, double v_max
        ) : 
        x2d(x2d), y2d(y2d), x3d(x3d), y3d(y3d), z3d(z3d),
        fx(fx), fy(fy), cx(cx), cy(cy), wxx(wxx), wyy(wyy),
        z_min(z_min), u_min(u_min), u_max(u_max),
        v_min(v_min), v_max(v_max) {}

    template <typename T>
    bool operator()(const T *const dimpose, T *residuals) const
    {
        T pts3d[] = {T(x3d) * exp(T(dimpose[0])), 
                     T(y3d) * exp(T(dimpose[1])), 
                     T(z3d) * exp(T(dimpose[2]))};
        T r_vec[] = {T(0), dimpose[3], T(0)};
        T trans_pts3d[3];
        T clips[] = {T(z_min), T(u_min), T(u_max), T(v_min), T(v_max)};
        ceres::AngleAxisRotatePoint(r_vec, pts3d, trans_pts3d);
        trans_pts3d[0] += dimpose[4];
        trans_pts3d[1] += dimpose[5];
        trans_pts3d[2] += dimpose[6];

        trans_pts3d[2] = max(trans_pts3d[2], clips[0]);

        T proj_x = T(fx) * trans_pts3d[0] / trans_pts3d[2] + T(cx);
        T proj_y = T(fy) * trans_pts3d[1] / trans_pts3d[2] + T(cy);

        proj_x = (proj_x < clips[1]) ? clips[1] : (proj_x > clips[2]) ? clips[2] : proj_x;
        proj_y = (proj_y < clips[3]) ? clips[3] : (proj_y > clips[4]) ? clips[4] : proj_y;

        T diff_x = proj_x - T(x2d);
        T diff_y = proj_y - T(y2d);

        residuals[0] = T(wxx) * diff_x;
        residuals[1] = T(wyy) * diff_y;
        return true;
    }

    static ceres::CostFunction *Create(
        double x2d, double y2d,
        double x3d, double y3d, double z3d,
        double wxx, double wyy,
        double fx, double fy, double cx, double cy,
        double z_min, double u_min, double u_max, double v_min, double v_max)
    {
        return (new ceres::AutoDiffCostFunction<NocReprojectionErrorArray, 2, 7>(
            new NocReprojectionErrorArray(
                x2d, y2d, x3d, y3d, z3d, wxx, wyy, fx, fy, cx, cy,
                z_min, u_min, u_max, v_min, v_max)));
    }

    double x2d, y2d;
    double x3d, y3d, z3d;
    double fx, fy;
    double cx, cy;
    double wxx, wyy;
    double z_min, u_min, u_max, v_min, v_max;
};


struct NocCovReprojectionErrorArray
{
    NocCovReprojectionErrorArray(
        double x2d, double y2d,
        double x3d, double y3d, double z3d,
        double wxx, double wxy, double wyy,
        double fx, double fy, double cx, double cy,
        double z_min, double u_min, double u_max,
        double v_min, double v_max
        ) :
        x2d(x2d), y2d(y2d), x3d(x3d), y3d(y3d), z3d(z3d),
        fx(fx), fy(fy), cx(cx), cy(cy), wxx(wxx), wxy(wxy), wyy(wyy),
        z_min(z_min), u_min(u_min), u_max(u_max),
        v_min(v_min), v_max(v_max) {}

    template <typename T>
    bool operator()(const T *const dimpose, T *residuals) const
    {
        T pts3d[] = {T(x3d) * exp(T(dimpose[0])),
                     T(y3d) * exp(T(dimpose[1])),
                     T(z3d) * exp(T(dimpose[2]))};
        T r_vec[] = {T(0), dimpose[3], T(0)};
        T trans_pts3d[3];
        T clips[] = {T(z_min), T(u_min), T(u_max), T(v_min), T(v_max)};
        ceres::AngleAxisRotatePoint(r_vec, pts3d, trans_pts3d);
        trans_pts3d[0] += dimpose[4];
        trans_pts3d[1] += dimpose[5];
        trans_pts3d[2] += dimpose[6];

        trans_pts3d[2] = max(trans_pts3d[2], clips[0]);

        T proj_x = T(fx) * trans_pts3d[0] / trans_pts3d[2] + T(cx);
        T proj_y = T(fy) * trans_pts3d[1] / trans_pts3d[2] + T(cy);

        proj_x = (proj_x < clips[1]) ? clips[1] : (proj_x > clips[2]) ? clips[2] : proj_x;
        proj_y = (proj_y < clips[3]) ? clips[3] : (proj_y > clips[4]) ? clips[4] : proj_y;

        T diff_x = proj_x - T(x2d);
        T diff_y = proj_y - T(y2d);

        residuals[0] = T(wxx) * diff_x + T(wxy) * diff_y;
        residuals[1] = T(wxy) * diff_x + T(wyy) * diff_y;
        return true;
    }

    static ceres::CostFunction *Create(
        double x2d, double y2d,
        double x3d, double y3d, double z3d,
        double wxx, double wxy, double wyy,
        double fx, double fy, double cx, double cy,
        double z_min, double u_min, double u_max, double v_min, double v_max)
    {
        return (new ceres::AutoDiffCostFunction<NocCovReprojectionErrorArray, 2, 7>(
            new NocCovReprojectionErrorArray(
                x2d, y2d, x3d, y3d, z3d, wxx, wxy, wyy, fx, fy, cx, cy,
                z_min, u_min, u_max, v_min, v_max)));
    }

    double x2d, y2d;
    double x3d, y3d, z3d;
    double fx, fy;
    double cx, cy;
    double wxx, wxy, wyy;
    double z_min, u_min, u_max, v_min, v_max;
};


#ifdef __cplusplus
extern "C"
{
#endif
    void pnp_uncert(
        double* pts2d,      // n, 2
        double* pts3d,      // n, 3
        double* wgt2d,      // n, 2
        double* K,          // 3, 3
        double* init_pose,    // 4
        int* result_val,
        double* result_pose,  // 4
        double* result_cov, // 4, 4
        double* result_tr,
        int pn, 
        double* clips)      // z_min, u_min, u_max, v_min, v_max
    {
        ceres::Problem problem;
        memcpy(result_pose, init_pose, 4 * sizeof(double));
        for (int i = 0; i < pn; ++i) {
            ceres::CostFunction* cost_function = ReprojectionErrorArray::Create(
                pts2d[i * 2], pts2d[i * 2 + 1],
                pts3d[i * 3], pts3d[i * 3 + 1], pts3d[i * 3 + 2],
                wgt2d[i * 2], wgt2d[i * 2 + 1],
                K[0], K[4], K[2], K[5],
                clips[0], clips[1], clips[2], clips[3], clips[4]);
            problem.AddResidualBlock(cost_function, NULL, result_pose);
        }

        ceres::Solver::Options sol_options;
        sol_options.linear_solver_type = ceres::DENSE_QR;
        // sol_options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(sol_options, &problem, &summary);
        // cout<<summary.FullReport()<<"\n";
        *result_val = int(summary.IsSolutionUsable());
        *result_tr = summary.iterations.back().trust_region_radius;
        
        if (*result_val && result_cov) {    // result_cov can be NULL when it is not needed 
            ceres::Covariance::Options cov_options;
            cov_options.algorithm_type = ceres::SPARSE_QR;  // or try DENSE_SVD
            cov_options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
            ceres::Covariance covariance(cov_options);

            vector<pair<const double*, const double*>> covariance_blocks;
            covariance_blocks.push_back(make_pair(result_pose, result_pose));
            *result_val = int(covariance.Compute(covariance_blocks, &problem));
            if (*result_val) {
                covariance.GetCovarianceBlock(result_pose, result_pose, result_cov);
            }
        }
    }

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
        double* clips,          // z_min, u_min, u_max, v_min, v_max
        double delta)
    {
        ceres::Problem problem;
        memcpy(result_dimpose, init_dimpose, 7 * sizeof(double));

        ceres::HuberLoss* lossfun = new ceres::HuberLoss(delta);
        for (int i = 0; i < pn; ++i) {
            ceres::CostFunction* cost_function = NocReprojectionErrorArray::Create(
                pts2d[i * 2], pts2d[i * 2 + 1],
                pts3d[i * 3], pts3d[i * 3 + 1], pts3d[i * 3 + 2],
                wgt2d[i * 2], wgt2d[i * 2 + 1],
                K[0], K[4], K[2], K[5],
                clips[0], clips[1], clips[2], clips[3], clips[4]);
            problem.AddResidualBlock(cost_function, lossfun, result_dimpose);
        }

        ceres::CostFunction* cost_function = DimErrorArray::Create(
            logdim[0], logdim[1], logdim[2],
            logdim_wgt[0], logdim_wgt[1], logdim_wgt[2]);
        problem.AddResidualBlock(cost_function, lossfun, result_dimpose);

        ceres::Solver::Options sol_options;
        sol_options.linear_solver_type = ceres::DENSE_QR;
        // sol_options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(sol_options, &problem, &summary);
        // cout<<summary.FullReport()<<"\n";
        *result_val = int(summary.IsSolutionUsable());
    }

    void pnp_noc_cov_uncert(
        double* pts2d,          // n, 2
        double* pts3d,          // n, 3
        double* wgt2d,          // n, 3
        double* logdim,         // 3
        double* logdim_wgt,     // 3
        double* K,              // 3, 3
        double* init_dimpose,      // 7
        int* result_val,
        double* result_dimpose,    // 7
        int pn,
        double* clips,          // z_min, u_min, u_max, v_min, v_max
        double delta)
    {
        ceres::Problem problem;
        memcpy(result_dimpose, init_dimpose, 7 * sizeof(double));

        ceres::HuberLoss* lossfun = new ceres::HuberLoss(delta);
        for (int i = 0; i < pn; ++i) {
            ceres::CostFunction* cost_function = NocCovReprojectionErrorArray::Create(
                pts2d[i * 2], pts2d[i * 2 + 1],
                pts3d[i * 3], pts3d[i * 3 + 1], pts3d[i * 3 + 2],
                wgt2d[i * 3], wgt2d[i * 3 + 1], wgt2d[i * 3 + 2],
                K[0], K[4], K[2], K[5],
                clips[0], clips[1], clips[2], clips[3], clips[4]);

            problem.AddResidualBlock(cost_function, lossfun, result_dimpose);
        }

        ceres::CostFunction* cost_function = DimErrorArray::Create(
            logdim[0], logdim[1], logdim[2],
            logdim_wgt[0], logdim_wgt[1], logdim_wgt[2]);
        problem.AddResidualBlock(cost_function, lossfun, result_dimpose);

        ceres::Solver::Options sol_options;
        sol_options.linear_solver_type = ceres::DENSE_QR;
        // sol_options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(sol_options, &problem, &summary);
        // cout<<summary.FullReport()<<"\n";
        *result_val = int(summary.IsSolutionUsable());
    }
#ifdef __cplusplus
}
#endif
