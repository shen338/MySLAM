#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

using namespace std; 

/* Curve fitting for function "y = e^(a*x*x + b*x + c)"
Only need a single vertex with three parameter a,b,c in it. 
All edges are unary edges connected to the only vertex
*/

// vertices as optimization target
// inherent from g2o::BaseVertex, predefined Vertex
class CurveFittingVertex: public g2o::BaseVertex<3, Eigen::Vector3d>
{
    public: 
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        virtual void setToOriginImpl(){
            _estimate << 0, 0, 0;
        }

        virtual void oplusImpl( const double* update ) // 更新
        {
            _estimate += Eigen::Vector3d(update);
        }
        
        // read and write leave empty
        virtual bool read( istream& in ) {}
        virtual bool write( ostream& out ) const {}
};

// edges to compute error
class CurveFittingEdge: public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge( double x ): BaseUnaryEdge(), _x(x) {}

    void computeError(){
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0, 0) = _measurement - std::exp(abc(0, 0)*_x*_x + abc(1, 0)*_x + abc(2, 0));
    }
    virtual bool read( istream& in ) {}
    virtual bool write( ostream& out ) const {}
    public: 
        double _x;
};

int main(){
    cv::RNG rng;  
    double a=1.0, b=2.0, c=1.0;   // Real parameters     
    int N = 200; 
    double w_sigma=1.0;                   
<<<<<<< HEAD
=======

>>>>>>> d3aa7eab7a66633ee6b6716534884a999f4bfb7a
    vector<double> x_data, y_data;      // generated data

    cout<<"generating data: "<<endl;
    for ( int i=0; i<N; i++ )
    {
        double x = i/100.0;
        x_data.push_back ( x );
        y_data.push_back (
            exp ( a*x*x + b*x + c ) + rng.gaussian ( w_sigma )
        );
        cout<<x_data[i]<<" "<<y_data[i]<<endl;
    }

    double parameters[3] = {0,0,0};  // Parameters need to be solved

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> Block;   // input 3 dims, output 1 dims
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>(); 

    Block *solver_ptr = new Block(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm (solver);
    optimizer.setVerbose(true);

    // add Vertex as optimization target 

    CurveFittingVertex *v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(0, 0, 0));
    v->setId(0);
    optimizer.addVertex(v);

    for(int i=0;i<N;i++){
        CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, v);
        edge->setMeasurement(y_data[i]);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity()*1/(w_sigma*w_sigma));
        optimizer.addEdge(edge);
    } 

    // Execute optimization. 
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;

    Eigen::Vector3d parameter_estimate = v->estimate();
    cout << "Estimate model" << parameter_estimate.transpose() << endl;

}