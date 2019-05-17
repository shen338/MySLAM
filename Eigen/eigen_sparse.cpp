#include <vector>
#include <iostream>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/IterativeLinearSolvers>
// #include <eigen3/Eigen/Core>
// #include <eigen3/Eigen/SparseCore>
// #include <Eigen/SparseCholesky>
// #include<Eigen/SparseLU> 
// #include<Eigen/SparseQR>
// #include <Eigen/IterativeLinearSolvers>

using namespace std;
using namespace Eigen;

int main(){
    typedef Eigen::Triplet<double> T;
    typedef Eigen::SparseMatrix<double> SpMat;

    Eigen::SparseMatrix<std::complex<float> > complex_mat(1000,2000);
    Eigen::SparseMatrix<double> mat(1000,2000); 
    
    cout << "Sparse matrix sizes" << endl;
    cout << mat.rows();
    cout << mat.cols();
    cout << mat.innerSize();
    cout << mat.outerSize();
    cout << mat.nonZeros();

    // use triplet to fill in sparse matrix
    vector<T> triplet_list; 
    for (int i = 0;i<1000;i=i+100){
        triplet_list.push_back(T(i, i, i));
    }

    mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
    
    cout << "Sparse Matrix entries after initialization" << endl;
    for (int k=0; k<mat.outerSize(); ++k)
        for (SparseMatrix<double>::InnerIterator it(mat,k); it; ++it)
        {
            cout << it.value() << endl;
            cout << it.row() << endl;   // row index
            cout << it.col() << endl;   // col index (here it is equal to k)
            cout << it.index() << endl; // inner index, here it is equal to it.row()
        }

    // Supported operators and functions 
    // sparse matrices cannot offer the same level of flexibility than dense matrices
    mat.transpose();
    mat.adjoint();

    mat.pruned();
    double ref = 1e-8;
    mat.pruned(ref);
    

    // // Eigen Sparse matrix solver 
    // SparseMatrix<double> A;
    // // fill A
    // VectorXd b, x;
    // // fill b
    // // solve Ax = b
    // ConjugateGradient<SparseMatrix<double> > solver;
    // solver.compute(A);
    // if(solver.info()!=Success) {
    // // decomposition failed
    // return;
    // }
    // x = solver.solve(b);
    // if(solver.info()!=Success) {
    // // solving failed
    // return;
    // }
    // // solve for another right hand side:
    // x = solver.solve(b);

    

    return 0;
}