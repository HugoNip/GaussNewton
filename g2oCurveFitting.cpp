#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

// Vertex, template parameters: optimized variable dimension and data type
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // vertex reset
    virtual void setToOriginImpl() override {
        _estimate << 0, 0, 0;
    }

    // vertex update function, x_k+1 = x_k + d_x
    virtual void oplusImpl(const double *update) override {
        _estimate += Eigen::Vector3d(update);
    }

    // write and read: blank
    virtual bool read(std::istream &in) {}
    virtual bool write(std::ostream &out) const {}
};

// error model template parameters: real value dimension, type, vertex type
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

    // calculate curve model error
    virtual void computeError() override {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
    }

    // calculate Jacobian matrix
    virtual void linearizeOplus() override {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
        _jacobianOplusXi[0] = -_x * _x * y;
        _jacobianOplusXi[1] = -_x * y;
        _jacobianOplusXi[2] = -y;
    }

    virtual bool read(std::istream &in) {}
    virtual bool write(std::ostream &out) const {}

public:
    double _x; // x, "y" is _measurement
};

int main(int argc, char** argv) {
    double ar = 1.0, br = 2.0, cr = 1.0;
    double ae = 2.0, be = -1.0, ce = 5.0;
    int N = 100;
    double w_sigma = 1.0;
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;

    std::vector<double> x_data, y_data;
    for (int i = 0; i < N; ++i) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    // construct graph optimization
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    // Gradient decent method
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
            );
    g2o::SparseOptimizer optimizer; // Graph model
    optimizer.setAlgorithm(solver); // set solver
    optimizer.setVerbose(true); // open debug output

    // add vertex
    CurveFittingVertex *v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(ae, be, ce));
    v->setId(0);
    optimizer.addVertex(v);

    // add Edge
    for (int j = 0; j < N; ++j) {
        CurveFittingEdge *edge = new CurveFittingEdge(x_data[j]);
        edge->setId(j);
        edge->setVertex(0, v);
        edge->setMeasurement(y_data[j]);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma));
        optimizer.addEdge(edge);
    }

    // optimizing
    std::cout << "Start optimization" << std::endl;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "solve time cost = " << time_used.count() << " seconds." << std::endl;

    // output optimized value
    Eigen::Vector3d abc_estimate = v->estimate();
    std::cout << "estimated model: " << abc_estimate.transpose() << std::endl;

    return 0;
}