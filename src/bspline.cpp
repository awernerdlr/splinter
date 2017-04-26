/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "bspline.h"
#include "bspline_basis.h"
#include "kronecker_product.h"
#include "unsupported/Eigen/KroneckerProduct"
#include <linear_solvers.h>
#include <serializer.h>
#include <iostream>
#include <utilities.h>

namespace SPLINTER
{

/**
 * Constructor used by serializer
 */
BSpline::BSpline()
    : Function()
{}

/**
 * Constructors for multivariate B-spline using explicit data
 */
BSpline::BSpline(
        unsigned int dimX,
        unsigned int dimY,
        const std::vector<std::vector<double>> &knotVectors,
        const std::vector<unsigned int> &degrees)
    : Function(dimX, dimY),
      basis(BSplineBasis(knotVectors, degrees)),
      controlPoints(DenseMatrix::Zero(basis.getNumBasisFunctions(), dimY))
{
    checkControlPoints();
}

BSpline::BSpline(const std::vector<std::vector<double>> &controlPoints,
                 const std::vector<std::vector<double>> &knotVectors,
                 const std::vector<unsigned int> &degrees)
    : Function(knotVectors.size(), controlPoints.at(0).size()),
      basis(BSplineBasis(knotVectors, degrees)),
      controlPoints(stdVecVecToEigMat(controlPoints))
{
    checkControlPoints();
}

/**
 * Construct B-spline from saved data
 */
BSpline::BSpline(const char *fileName)
    : BSpline(std::string(fileName))
{
}

BSpline::BSpline(const std::string &fileName)
    : Function()
{
    load(fileName);
}

/**
 * Returns the function value at x
 */
std::vector<double> BSpline::eval(const std::vector<double> &x) const {
    return eigToStdVec(eval(stdToEigVec(x)));
}

DenseVector BSpline::eval(const DenseVector &x) const
{
    checkInput(x);
    DenseVector res = controlPoints.transpose()*evalBasis(x);
    return res;
}

/**
 * Returns the (dimY x dimX) Jacobian evaluated at x
 */
DenseMatrix BSpline::evalJacobian(const DenseVector &x) const
{
    checkInput(x);
    return controlPoints.transpose()*evalBasisJacobian(x);
}

/*
 * Returns the Hessian evaluated at x. The Hessian is a (dimY x dimX x dimX) tensor.
 */
std::vector<std::vector<std::vector<double>>> BSpline::evalHessian(const std::vector<double> &x) const
{
    DenseVector eigX = stdToEigVec(x);
    checkInput(eigX);

    std::vector<std::vector<std::vector<double>>> hessian;

    DenseMatrix identity = DenseMatrix::Identity(dimX, dimX);

    DenseMatrix cpCopy = DenseMatrix(controlPoints);

    for (size_t i = 0; i < dimY; ++i)
    {
        DenseMatrix H = DenseMatrix::Zero(1, 1);
        DenseMatrix cp = cpCopy.col(i);
        DenseMatrix caug = kroneckerProduct(identity, cp.transpose());
        DenseMatrix DB = basis.evalBasisHessian(eigX);

        H = caug*DB;

//        std::cout << cp << std::endl;

        // Fill in upper triangular of Hessian
        for (size_t j = 0; j < dimX; ++j)
            for (size_t k = j+1; k < dimX; ++k)
                H(j, k) = H(k, j);

        hessian.push_back(eigMatToStdVecVec(H));
    }

    return hessian;
}

// Evaluation of B-spline basis functions
SparseVector BSpline::evalBasis(const DenseVector &x) const
{
#ifndef NDEBUG
    if (!isSupported(x))
        std::cout << "BSpline::evalBasis: Evaluation at point outside of support." << std::endl;
#endif // NDEBUG

    return basis.eval(x);
}

SparseMatrix BSpline::evalBasisJacobian(const DenseVector &x) const
{
#ifndef NDEBUG
    if (!isSupported(x))
        std::cout << "BSpline::evalBasisJacobian: Evaluation at point outside of support." << std::endl;
#endif // NDEBUG

    //SparseMatrix Bi = basis.evalBasisJacobian(x);       // Sparse Jacobian implementation
    //SparseMatrix Bi = basis.evalBasisJacobian2(x);  // Sparse Jacobian implementation
    DenseMatrix Bi = basis.evalBasisJacobianOld(x);  // Old Jacobian implementation

    return Bi.sparseView();
}

std::vector<unsigned int> BSpline::getNumBasisFunctionsPerVariable() const
{
    std::vector<unsigned int> ret;
    for (unsigned int i = 0; i < dimX; i++)
        ret.push_back(basis.getNumBasisFunctions(i));
    return ret;
}

std::vector<std::vector<double>> BSpline::getKnotVectors() const
{
    return basis.getKnotVectors();
}

std::vector<unsigned int> BSpline::getBasisDegrees() const
{
    return basis.getBasisDegrees();
}

std::vector<double> BSpline::getDomainUpperBound() const
{
    return basis.getSupportUpperBound();
}

std::vector<double> BSpline::getDomainLowerBound() const
{
    return basis.getSupportLowerBound();
}

void BSpline::setControlPoints(const DenseMatrix &newControlPoints)
{
    if (newControlPoints.rows() != getNumBasisFunctions())
        throw Exception("BSpline::setControlPoints: Incompatible size of coefficient vector. " +
                                std::to_string(newControlPoints.rows()) + " not equal to " +
                                std::to_string(getNumBasisFunctions()) + "!");

    this->controlPoints = newControlPoints;
    checkControlPoints();
}

void BSpline::updateControlPoints(const SparseMatrix &A)
{
    if (A.cols() != controlPoints.rows())
        throw Exception("BSpline::updateControlPoints: Incompatible size of linear transformation matrix.");
    controlPoints = A*controlPoints;
}

void BSpline::checkControlPoints() const
{
    if (controlPoints.cols() != getDimY())
        throw Exception("BSpline::checkControlPoints: Inconsistent number of columns of control points matrix.");
    if (controlPoints.rows() != getNumBasisFunctions())
        throw Exception("BSpline::checkControlPoints: Inconsistent number of rows of control points matrix.");
}

bool BSpline::isSupported(const DenseVector &x) const
{
    return basis.insideSupport(x);
}

void BSpline::reduceSupport(const std::vector<double> &lb, const std::vector<double> &ub, bool doRegularizeKnotVectors)
{
    if (lb.size() != dimX || ub.size() != dimX)
        throw Exception("BSpline::reduceSupport: Inconsistent vector sizes!");

    std::vector<double> sl = basis.getSupportLowerBound();
    std::vector<double> su = basis.getSupportUpperBound();

    for (unsigned int dim = 0; dim < dimX; dim++)
    {
        // Check if new domain is empty
        if (ub.at(dim) <= lb.at(dim) || lb.at(dim) >= su.at(dim) || ub.at(dim) <= sl.at(dim))
            throw Exception("BSpline::reduceSupport: Cannot reduce B-spline domain to empty set!");

        // Check if new domain is a strict subset
        if (su.at(dim) < ub.at(dim) || sl.at(dim) > lb.at(dim))
            throw Exception("BSpline::reduceSupport: Cannot expand B-spline domain!");

        // Tightest possible
        sl.at(dim) = lb.at(dim);
        su.at(dim) = ub.at(dim);
    }

    if (doRegularizeKnotVectors)
    {
        regularizeKnotVectors(sl, su);
    }

    // Remove knots and control points that are unsupported with the new bounds
    if (!removeUnsupportedBasisFunctions(sl, su))
    {
        throw Exception("BSpline::reduceSupport: Failed to remove unsupported basis functions!");
    }
}

void BSpline::globalKnotRefinement()
{
    // Compute knot insertion matrix
    SparseMatrix A = basis.refineKnots();

    // Update control points
    updateControlPoints(A);
}

void BSpline::localKnotRefinement(const DenseVector &x)
{
    // Compute knot insertion matrix
    SparseMatrix A = basis.refineKnotsLocally(x);

    // Update control points
    updateControlPoints(A);
}

void BSpline::decomposeToBezierForm()
{
    // Compute knot insertion matrix
    SparseMatrix A = basis.decomposeToBezierForm();

    // Update control points
    updateControlPoints(A);
}

// Computes knot averages: assumes that basis is initialized!
DenseMatrix BSpline::computeKnotAverages() const
{
    // Calculate knot averages for each knot vector
    std::vector<DenseVector> mu_vectors;
    for (unsigned int i = 0; i < dimX; i++)
    {
        std::vector<double> knots = basis.getKnotVector(i);
        DenseVector mu = DenseVector::Zero(basis.getNumBasisFunctions(i));

        for (unsigned int j = 0; j < basis.getNumBasisFunctions(i); j++)
        {
            double knotAvg = 0;
            for (unsigned int k = j+1; k <= j+basis.getBasisDegree(i); k++)
            {
                knotAvg += knots.at(k);
            }
            mu(j) = knotAvg/basis.getBasisDegree(i);
        }
        mu_vectors.push_back(mu);
    }

    // Calculate vectors of ones (with same length as corresponding knot average vector)
    std::vector<DenseVector> knotOnes;
    for (unsigned int i = 0; i < dimX; i++)
        knotOnes.push_back(DenseVector::Ones(mu_vectors.at(i).rows()));

    // Fill knot average matrix one column at the time
    DenseMatrix knot_averages = DenseMatrix::Zero(basis.getNumBasisFunctions(), dimX);

    for (unsigned int i = 0; i < dimX; i++)
    {
        DenseMatrix mu_ext(1,1); mu_ext(0,0) = 1;
        for (unsigned int j = 0; j < dimX; j++)
        {
            DenseMatrix temp = mu_ext;
            if (i == j)
                mu_ext = Eigen::kroneckerProduct(temp, mu_vectors.at(j));
            else
                mu_ext = Eigen::kroneckerProduct(temp, knotOnes.at(j));
        }
        if (mu_ext.rows() != basis.getNumBasisFunctions())
            throw Exception("BSpline::computeKnotAverages: Incompatible size of knot average matrix.");
        knot_averages.block(0, i, basis.getNumBasisFunctions(), 1) = mu_ext;
    }

    return knot_averages;
}

void BSpline::insertKnots(double tau, unsigned int dim, unsigned int multiplicity)
{
    // Insert knots and compute knot insertion matrix
    SparseMatrix A = basis.insertKnots(tau, dim, multiplicity);

    // Update control points
    updateControlPoints(A);
}

void BSpline::regularizeKnotVectors(const std::vector<double> &lb, const std::vector<double> &ub)
{
    // Add and remove controlpoints and knots to make the B-spline p-regular with support [lb, ub]
    if (!(lb.size() == dimX && ub.size() == dimX))
        throw Exception("BSpline::regularizeKnotVectors: Inconsistent vector sizes.");

    for (unsigned int dim = 0; dim < dimX; dim++)
    {
        unsigned int multiplicityTarget = basis.getBasisDegree(dim) + 1;

        // Inserting many knots at the time (to save number of B-spline coefficient calculations)
        // NOTE: This method generates knot insertion matrices with more nonzero elements than
        // the method that inserts one knot at the time. This causes the preallocation of
        // kronecker product matrices to become too small and the speed deteriorates drastically
        // in higher dimensions because reallocation is necessary. This can be prevented by
        // computing the number of nonzeros when preallocating memory (see myKroneckerProduct).
        int numKnotsLB = multiplicityTarget - basis.getKnotMultiplicity(dim, lb.at(dim));
        if (numKnotsLB > 0)
        {
            insertKnots(lb.at(dim), dim, numKnotsLB);
        }

        int numKnotsUB = multiplicityTarget - basis.getKnotMultiplicity(dim, ub.at(dim));
        if (numKnotsUB > 0)
        {
            insertKnots(ub.at(dim), dim, numKnotsUB);
        }
    }
}

bool BSpline::removeUnsupportedBasisFunctions(const std::vector<double> &lb, const std::vector<double> &ub)
{
    if (lb.size() != dimX || ub.size() != dimX)
        throw Exception("BSpline::removeUnsupportedBasisFunctions: Incompatible dimension of domain bounds.");

    SparseMatrix A = basis.reduceSupport(lb, ub);

    // Remove unsupported control points (basis functions)
    updateControlPoints(A);

    return true;
}

void BSpline::save(const std::string &fileName) const
{
    Serializer s;
    s.serialize(*this);
    s.saveToFile(fileName);
}

void BSpline::load(const std::string &fileName)
{
    Serializer s(fileName);
    s.deserialize(*this);
}

std::string BSpline::getDescription() const
{
    std::string description("BSpline of degree");
    auto degrees = getBasisDegrees();
    // See if all degrees are the same.
    bool equal = true;
    for (size_t i = 1; i < degrees.size(); ++i)
    {
        equal = equal && (degrees.at(i) == degrees.at(i-1));
    }

    if(equal)
    {
        description.append(" ");
        description.append(std::to_string(degrees.at(0)));
    }
    else
    {
        description.append("s (");
        for (size_t i = 0; i < degrees.size(); ++i)
        {
            description.append(std::to_string(degrees.at(i)));
            if (i + 1 < degrees.size())
            {
                description.append(", ");
            }
        }
        description.append(")");
    }

    return description;
}

void BSpline::setKnotVectors(const std::vector<std::vector<double> > & knots) {
    basis.setKnotVectors(knots);
}
    
double BSpline::evalPartialDerivative(DenseVector x,
        std::size_t dim, std::size_t order) const {
    static DenseVector r(coefficients.size());
    r = evalBasisPartialDerivative(x,dim,order).transpose() * coefficients;
    return r(0);
}
    
SparseVector BSpline::evalBasisPartialDerivative(DenseVector x,
        std::size_t dim, std::size_t order) const {
    return basis.getSingleBasis(dim).evalDerivative(x(dim),order);
}

SparseVector BSpline::evalKnotPartialDerivative(
        DenseVector x, std::size_t dim, std::size_t order) const {
    return basis.getSingleBasis(dim).evalKnotDerivative(x(dim),order);
}


} // namespace SPLINTER
