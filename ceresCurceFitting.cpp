#include <iostream>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <chrono>

// Cost function model
struct CURVE_FITTING_COST {
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

    // Residual value
    template<typename T>
    bool operator() (
            const T* const abc, // a, b, c parameters
            T* residual) const { // residual value
        // y - exp(ax^2 + bx + c)
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        return true;
    }

    const double _x, _y; // x, y data
};

int main(int argc, char** argv) {
    double ar = 1.0, br = 2.0, cr = 1.0; // real
    double ae = 2.0, be = -1.0, ce = 5.0; // estimated
    int N = 100; // number of data
    double w_sigma = 1.0; // sigma of noise
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng; // random value

    // Generate real value
    std::vector<double> x_data, y_data;
    for (int i = 0; i < N; ++i) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr)
                         + rng.gaussian(w_sigma * w_sigma)); // y = exp(ax^2 + bx + c) + w
    }

    double abc[3] = {ae, be, ce};

    // construct LSP
    ceres::Problem problem;
    for (int j = 0; j < N; ++j) {
        problem.AddResidualBlock(
                // use autodiffcost calculation function
                // ceres::AutoDiffCostFunction<cost, output dim, input dim>
                new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
                        new CURVE_FITTING_COST(x_data[j], y_data[j])
                        ),
                nullptr, // kernel function: null
                abc // estimated parameters
                );
    }

    // configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true; // output to cout

    ceres::Solver::Summary summary; // optimization information
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary); // start to optimization
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "solve time cost = " << time_used.count() << " seconds." << std::endl;

    // output results
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "estimated a, b, c = ";
    for (auto a:abc) std::cout << a << " ";

    return 0;
}

