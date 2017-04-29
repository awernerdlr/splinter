/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <bspline_basis_1d.h>
#include <knot_utils.h>
#include <algorithm>
#include <utilities.h>
#include <iostream>

namespace SPLINTER
{

BSplineBasis1D::BSplineBasis1D()
{
}

BSplineBasis1D::BSplineBasis1D(const std::vector<double> &knots, unsigned int degree)
    : degree(degree),
      knots(KnotVector(knots)),
      targetNumBasisfunctions((degree+1)+2*degree+1) // Minimum p+1
{
    // Check that knot vector is (p+1)-regular
    if (!this->knots.is_regular(degree))
        throw Exception("BSplineBasis1D::BSplineBasis1D: Knot vector is not regular.");
}

SparseVector BSplineBasis1D::eval(double x) const
{
    SparseVector values(getNumBasisFunctions());

    if (!is_supported(x))
        return values;

    x = supportHack(x);

    auto indexSupported = indexSupportedBasisFunctions(x);

    values.reserve(indexSupported.size());

    // Evaluate nonzero basis functions
    for (auto it = indexSupported.begin(); it != indexSupported.end(); ++it)
    {
        double val = deBoorCox(x, *it, degree);
        if (fabs(val) > 1e-12)
            values.insert(*it) = val;
    }

    // Alternative evaluation using basis matrix
//    int knotIndex = indexHalfopenInterval(x); // knot index

//    SparseMatrix basisvalues2 = buildBsplineMatrix(x, knotIndex, 1);
//    for (int i = 2; i <= basisDegree; i++)
//    {
//        SparseMatrix Ri = buildBsplineMatrix(x, knotIndex, i);
//        basisvalues2 = basisvalues2*Ri;
//    }
//    basisvalues2.makeCompressed();

//    assert(basisvalues2.rows() == 1);
//    assert(basisvalues2.cols() == basisDegree + 1);

    return values;
}

SparseVector BSplineBasis1D::evalDerivative(double x, int r) const
{
    // Evaluate rth derivative of basis functions at x
    // Returns vector [D^(r)B_(u-p,p)(x) ... D^(r)B_(u,p)(x)]
    // where u is the knot index and p is the degree
    int p = degree;

    // Continuity requirement
    if (p < r || !is_supported(x))
    {
        // Return zero-gradient
        SparseVector DB(getNumBasisFunctions());
        return DB;
    }

    // TODO: Check for knot multiplicity here!

    x = supportHack(x);

    int knotIndex = knots.index_interval(x);

    // Algorithm 3.18 from Lyche and Moerken (2011)
    SparseMatrix B(1,1);
    B.insert(0,0) = 1;

    for (int i = 1; i <= p-r; i++)
    {
        SparseMatrix R = buildBasisMatrix(x, knotIndex, i);
        B = B*R;
    }

    for (int i = p-r+1; i <= p; i++)
    {
        SparseMatrix DR = buildBasisMatrix(x, knotIndex, i, true);
        B = B*DR;
    }
    double factorial = std::tgamma(p+1)/std::tgamma(p-r+1);
    B = B*factorial;

    if (B.cols() != p+1)
        throw Exception("BSplineBasis1D::evalDerivative: Wrong number of columns of B matrix.");

    // From row vector to extended column vector
    SparseVector DB(getNumBasisFunctions());
    DB.reserve(p+1);
    int i = knotIndex-p; // First insertion index
    if(i<0)throw std::domain_error("Evaluating of basis function out of fully supported domain");
    for (int k = 0; k < B.outerSize(); ++k)
    for (SparseMatrix::InnerIterator it(B,k); it; ++it)
    {
        DB.insert(i+it.col()) = it.value();
    }

    return DB;
}

// Old implementation of first derivative of basis functions
SparseVector BSplineBasis1D::evalFirstDerivative(double x) const
{
    SparseVector values(getNumBasisFunctions());

    x = supportHack(x);

    auto supportedBasisFunctions = indexSupportedBasisFunctions(x);

    for (auto i : supportedBasisFunctions)
    {
        // Differentiate basis function
        // Equation 3.35 in Lyche & Moerken (2011)
        double b1 = deBoorCox(x, i, degree-1);
        double b2 = deBoorCox(x, i+1, degree-1);

        double t11 = knots.at(i);
        double t12 = knots.at(i+degree);
        double t21 = knots.at(i+1);
        double t22 = knots.at(i+degree+1);

        (t12 == t11) ? b1 = 0 : b1 = b1/(t12-t11);
        (t22 == t21) ? b2 = 0 : b2 = b2/(t22-t21);

        values.insert(i) = degree*(b1 - b2);
    }

    return values;
}

SparseVector BSplineBasis1D::evalDerivativedeBoorCox(
        double x, unsigned int r) const {
    SparseVector values(getNumBasisFunctions());

    x = supportHack(x);

    auto supportedBasisFunctions = indexSupportedBasisFunctions(x);

    for (auto i : supportedBasisFunctions)
    {
        values.insert(i) = 
            evalDerivativedeBoorCoxSingleBasis(x,i,degree,r);
    }

    return values;
}

double BSplineBasis1D::evalDerivativedeBoorCoxSingleBasis(
        double x, int i, int k, unsigned int r) const
{
    if ( r == 0 )
        return deBoorCox(x,i,k);
    
    // Differentiate basis function
    // Equation 3.35 in Lyche & Moerken (2011)
    double b1 = evalDerivativedeBoorCoxSingleBasis(x, i, k-1, r-1);
    double b2 = evalDerivativedeBoorCoxSingleBasis(x, i+1, k-1, r-1);

    double t11 = knots.at(i);
    double t12 = knots.at(i+k);
    double t21 = knots.at(i+1);
    double t22 = knots.at(i+k+1);

    (t12 == t11) ? b1 = 0 : b1 = b1/(t12-t11);
    (t22 == t21) ? b2 = 0 : b2 = b2/(t22-t21);

    return k * (b1 - b2);
}

/**
 * Implements time derivatives for knot vector derivatives in a
 * recursive manner. 
 */
SparseVector BSplineBasis1D::evalKnotDerivativeSingleBasis(double x, int r,
        int i, unsigned int k) const
{
    if ( r == 0 )
        return deBoorCoxKnotDerivative(x, i, k);
        
    // handle higher derivatives using 
    // Equation 3.35 in Lyche & Moerken (2011)
    // recursively

    // test code for r==1
    // for r>1 call evalKnotDerivative(x,r-1) and put it in here
    double b1 = evalDerivativedeBoorCoxSingleBasis(x, i, k-1, r-1);
    SparseVector B1 = evalKnotDerivativeSingleBasis(x, r-1, i, k-1);
    double b2 = evalDerivativedeBoorCoxSingleBasis(x, i+1, k-1, r-1);
    SparseVector B2 = evalKnotDerivativeSingleBasis(x, r-1, i+1, k-1);

    double t11 = knots.at(i);
    double t12 = knots.at(i+k);
    double t21 = knots.at(i+1);
    double t22 = knots.at(i+k+1);
    
    //(t12 == t11) ? b1 = 0 : b1 = b1/(t12-t11);
    if(t12 == t11)
    {
        B1.setZero();
    }
    else
    {
        B1 /= (t12-t11);
        B1.coeffRef(i) += b1/std::pow(t12-t11,2);
        B1.coeffRef(i+k) += - b1/std::pow(t12-t11,2);
    }
    //(t22 == t21) ? b2 = 0 : b2 = b2/(t22-t21);
    if(t22 == t21)
    {
        B2.setZero();
    }
    else
    {
        B2 /= (t22-t21);
        B2.coeffRef(i+1) += b2/std::pow(t22-t21,2);
        B2.coeffRef(i+k+1) += - b2/std::pow(t22-t21,2);
    }
    return k * (B1 - B2);
}

SparseMatrix BSplineBasis1D::evalKnotDerivative(double x, int r) const
{
    SparseMatrix jac(getNumBasisFunctions(),knots.size());
    
    if (!is_supported(x))
        return jac;
    
    x = supportHack(x);

    std::vector<unsigned int> indexSupported = indexSupportedBasisFunctions(x);

    jac.reserve(indexSupported.size()*getBasisDegree()); // TODO: enough?

    for (int i : indexSupported)
    {
        SparseVector grad = evalKnotDerivativeSingleBasis(x,r,i,degree);
        for (SparseVector::InnerIterator it(grad); it; ++it)
        {
            jac.insert(i,it.index()) = it.value();
        }
    }
    
    jac.makeCompressed();

    return jac;
}

// Used to evaluate basis functions - alternative to the recursive deBoorCox
SparseMatrix BSplineBasis1D::buildBasisMatrix(double x, unsigned int u, unsigned int k, bool diff) const
{
    /* Build B-spline Matrix
     * R_k in R^(k,k+1)
     * or, if diff = true, the differentiated basis matrix
     * DR_k in R^(k,k+1)
     */

    if (!(k >= 1 && k <= getBasisDegree()))
    {
        throw Exception("BSplineBasis1D::buildBasisMatrix: Incorrect input parameters!");
    }

//    assert(u >= basisDegree + 1);
//    assert(u < ks.size() - basisDegree);

    unsigned int rows = k;
    unsigned int cols = k+1;
    SparseMatrix R(rows, cols);
    R.reserve(Eigen::VectorXi::Constant(cols, 2));

    for (unsigned int i = 0; i < rows; i++)
    {
        double dk = knots.at(u+1+i) - knots.at(u+1+i-k);
        if (dk == 0)
        {
            continue;
        }
        else
        {
            if (diff)
            {
                // Insert diagonal element
                R.insert(i,i) = -1/dk;

                // Insert super-diagonal element
                R.insert(i,i+1) = 1/dk;
            }
            else
            {
                // Insert diagonal element
                double a = (knots.at(u+1+i) - x)/dk;
                if (!assertNear(a, .0))
                    R.insert(i,i) = a;

                // Insert super-diagonal element
                double b = (x - knots.at(u+1+i-k))/dk;
                if (!assertNear(b, .0))
                    R.insert(i,i+1) = b;
            }
        }
    }

    R.makeCompressed();

    return R;
}

SparseVector BSplineBasis1D::deBoorCoxKnotDerivative(
        double x, unsigned int i, unsigned int k) const {
    SparseVector grad(knots.size());
    if (k == 0)
    {
        return grad;
    }
    else
    {
        double s1 = deBoorCoxCoeff(x, knots.at(i),   knots.at(i+k));
        double s2 = deBoorCoxCoeff(x, knots.at(i+1), knots.at(i+k+1));
        SparseVector S1 = deBoorCoxCoeffKnotDerivative(x, i,   i+k);
        SparseVector S2 = deBoorCoxCoeffKnotDerivative(x, i+1, i+k+1);

        double r1 = deBoorCox(x, i,   k-1);
        double r2 = deBoorCox(x, i+1, k-1);
        SparseVector R1 = deBoorCoxKnotDerivative(x, i,   k-1);
        SparseVector R2 = deBoorCoxKnotDerivative(x, i+1, k-1);

        //return s1*r1 + (1-s2)*r2; -> product rule:
        grad = S1*r1 + s1 * R1;
        grad += - S2 * r2 + (1-s2) * R2;
        grad.prune(1e-12);
        return grad;
    }
}

double BSplineBasis1D::deBoorCox(double x, unsigned int i, unsigned int k) const
{
    if (k == 0)
    {
        if ((knots.at(i) <= x) && (x < knots.at(i+1)))
            return 1;
        else
            return 0;
    }
    else
    {
        double s1,s2,r1,r2;

        s1 = deBoorCoxCoeff(x, knots.at(i),   knots.at(i+k));
        s2 = deBoorCoxCoeff(x, knots.at(i+1), knots.at(i+k+1));

        r1 = deBoorCox(x, i,   k-1);
        r2 = deBoorCox(x, i+1, k-1);

        return s1*r1 + (1-s2)*r2;
    }
}

double BSplineBasis1D::deBoorCoxCoeff(double x, double x_min, double x_max) const
{
    if (x_min < x_max && x_min <= x && x <= x_max)
        return (x - x_min)/(x_max - x_min);
    return 0;
}

SparseVector BSplineBasis1D::deBoorCoxCoeffKnotDerivative(
        double x, int x_min_idx, int x_max_idx) const
{
    double x_min = knots.at(x_min_idx);
    double x_max = knots.at(x_max_idx);
    SparseVector grad(knots.size());
    if (x_min < x_max && x_min <= x && x <= x_max)
    {
        grad.insert(x_min_idx) =
            (x - x_min)/std::pow(x_max-x_min,2.) - 1./(x_max - x_min);
        grad.insert(x_max_idx) =
            - (x - x_min)/std::pow(x_max-x_min,2.);
    }
    return grad;
}

// Insert knots and compute knot insertion matrix (to update control points)
SparseMatrix BSplineBasis1D::insertKnots(double tau, unsigned int multiplicity)
{
    if (!is_supported(tau))
        throw Exception("BSplineBasis1D::insertKnots: Cannot insert knot outside domain!");

    if (knotMultiplicity(tau) + multiplicity > degree + 1)
        throw Exception("BSplineBasis1D::insertKnots: Knot multiplicity is too high!");

    // New knot vector
    int index = knots.index_interval(tau);

    std::vector<double> extKnots = knots.get_values();
    for (unsigned int i = 0; i < multiplicity; i++)
        extKnots.insert(extKnots.begin()+index+1, tau);

    if (!KnotVector(extKnots).is_regular(degree))
        throw Exception("BSplineBasis1D::insertKnots: New knot vector is not regular!");

    // Return knot insertion matrix
    SparseMatrix A = buildKnotInsertionMatrix(extKnots);

    // Update knots
    knots = KnotVector(extKnots);

    return A;
}

SparseMatrix BSplineBasis1D::refineKnots()
{
    // Build refine knot vector
    std::vector<double> refinedKnots = knots.get_values();

    unsigned int targetNumKnots = targetNumBasisfunctions + degree + 1;
    while (refinedKnots.size() < targetNumKnots)
    {
        int index = indexLongestInterval(refinedKnots);
        double newKnot = (refinedKnots.at(index) + refinedKnots.at(index+1))/2.0;
        refinedKnots.insert(std::lower_bound(refinedKnots.begin(), refinedKnots.end(), newKnot), newKnot);
    }

    if (!KnotVector(refinedKnots).is_regular(degree))
        throw Exception("BSplineBasis1D::refineKnots: New knot vector is not regular!");

    if (!knots.is_refinement(refinedKnots))
        throw Exception("BSplineBasis1D::refineKnots: New knot vector is not a proper refinement!");

    // Return knot insertion matrix
    SparseMatrix A = buildKnotInsertionMatrix(refinedKnots);

    // Update knots
    knots = KnotVector(refinedKnots);

    return A;
}

SparseMatrix BSplineBasis1D::refineKnotsLocally(double x)
{
    if (!is_supported(x))
        throw Exception("BSplineBasis1D::refineKnotsLocally: Cannot refine outside support!");

    if (getNumBasisFunctions() >= getNumBasisFunctionsTarget()
            || assertNear(knots.front(), knots.back()))
    {
        unsigned int n = getNumBasisFunctions();
        DenseMatrix A = DenseMatrix::Identity(n, n);
        return A.sparseView();
    }

    // Refined knot vector
    std::vector<double> refinedKnots = knots.get_values();

    auto upper = std::lower_bound(refinedKnots.begin(), refinedKnots.end(), x);

    // Check left boundary
    if (upper == refinedKnots.begin())
        std::advance(upper, degree+1);

    // Get previous iterator
    auto lower = std::prev(upper);

    // Do not insert if upper and lower bounding knot are close
    if (assertNear(*upper, *lower))
    {
        unsigned int n = getNumBasisFunctions();
        DenseMatrix A = DenseMatrix::Identity(n,n);
        return A.sparseView();
    }

    // Insert knot at x
    double insertVal = x;

    // Adjust x if it is on or close to a knot
    if (knotMultiplicity(x) > 0
            || assertNear(*upper, x, 1e-6, 1e-6)
            || assertNear(*lower, x, 1e-6, 1e-6))
    {
        insertVal = (*upper + *lower)/2.0;
    }

    // Insert new knot
    refinedKnots.insert(upper, insertVal);

    if (!KnotVector(refinedKnots).is_regular(degree))
        throw Exception("BSplineBasis1D::refineKnotsLocally: New knot vector is not regular!");

    if (!knots.is_refinement(refinedKnots))
        throw Exception("BSplineBasis1D::refineKnotsLocally: New knot vector is not a proper refinement!");

    // Build knot insertion matrix
    SparseMatrix A = buildKnotInsertionMatrix(refinedKnots);

    // Update knots
    knots = KnotVector(refinedKnots);

    return A;
}

SparseMatrix BSplineBasis1D::decomposeToBezierForm()
{
    // Build refine knot vector
    std::vector<double> refinedKnots = knots.get_values();

    // Start at first knot and add knots until all knots have multiplicity degree + 1
    std::vector<double>::iterator knoti = refinedKnots.begin();
    while (knoti != refinedKnots.end())
    {
        // Insert new knots
        int mult = degree + 1 - knotMultiplicity(*knoti);
        if (mult > 0)
        {
            std::vector<double> newKnots(mult, *knoti);
            refinedKnots.insert(knoti, newKnots.begin(), newKnots.end());
        }

        // Advance to next knot
        knoti = std::upper_bound(refinedKnots.begin(), refinedKnots.end(), *knoti);
    }

    if (!KnotVector(refinedKnots).is_regular(degree))
        throw Exception("BSplineBasis1D::refineKnots: New knot vector is not regular!");

    if (!knots.is_refinement(refinedKnots))
        throw Exception("BSplineBasis1D::refineKnots: New knot vector is not a proper refinement!");

    // Return knot insertion matrix
    SparseMatrix A = buildKnotInsertionMatrix(refinedKnots);

    // Update knots
    knots = KnotVector(refinedKnots);

    return A;
}

SparseMatrix BSplineBasis1D::buildKnotInsertionMatrix(const std::vector<double> &refined_knots) const
{
    if (!KnotVector(refined_knots).is_regular(degree))
        throw Exception("BSplineBasis1D::buildKnotInsertionMatrix: New knot vector is not regular!");

    if (!knots.is_refinement(refined_knots))
        throw Exception("BSplineBasis1D::buildKnotInsertionMatrix: New knot vector is not a proper refinement!");

    auto n = knots.size() - degree - 1;
    auto m = refined_knots.size() - degree - 1;

    SparseMatrix A(m, n);
    //A.resize(m,n);
    A.reserve(Eigen::VectorXi::Constant(n, degree + 1));

    // Build A row-by-row
    for (unsigned int i = 0; i < m; i++)
    {
        int u = knots.index_interval(refined_knots.at(i));

        SparseMatrix R(1,1);
        R.insert(0,0) = 1;

        // For p > 0
        for (unsigned int j = 1; j <= degree; j++)
        {
            SparseMatrix Ri = buildBasisMatrix(refined_knots.at(i + j), u, j);
            R = R*Ri;
        }

        // Size check
        if (R.rows() != 1 || R.cols() != (int)degree + 1)
        {
            throw Exception("BSplineBasis1D::buildKnotInsertionMatrix: Incorrect matrix dimensions!");
        }

        // Insert row values
        int j = u - degree; // First insertion index
        for (int k = 0; k < R.outerSize(); ++k)
        for (SparseMatrix::InnerIterator it(R, k); it; ++it)
        {
            if (!assertNear(it.value(), .0))
                A.insert(i, j + it.col()) = it.value();
        }
    }

    A.makeCompressed();

    return A;
}

/*
 * The B-spline domain is the half-open domain [ knots.first(), knots.end() ).
 * The hack checks if x is at the right boundary (if x = knots.end()), if so,
 * a small number is subtracted from x, moving x into the half-open domain.
 */
double BSplineBasis1D::supportHack(double x) const
{
    if (x == knots.back())
        return std::nextafter(x, std::numeric_limits<double>::lowest());
    return x;
}

SparseMatrix BSplineBasis1D::reduceSupport(double lb, double ub)
{
    // Check bounds
    if (lb < knots.front() || ub > knots.back())
        throw Exception("BSplineBasis1D::reduceSupport: Cannot increase support!");

    unsigned int k = degree + 1;

    auto index_lower = indexSupportedBasisFunctions(lb).front();
    auto index_upper = indexSupportedBasisFunctions(ub).back();

    // Check lower bound index
    if (k != knotMultiplicity(knots.at(index_lower)))
    {
        int suggested_index = index_lower - 1;
        if (0 <= suggested_index)
        {
            index_lower = suggested_index;
        }
        else
        {
            throw Exception("BSplineBasis1D::reduceSupport: Suggested index is negative!");
        }
    }

    // Check upper bound index
    if (knotMultiplicity(ub) == k && knots.at(index_upper) == ub)
    {
        index_upper -= k;
    }

    // New knot vector
    std::vector<double> si(knots.cbegin()+index_lower, knots.cbegin()+index_upper+k+1);

    // Construct selection matrix A
    int num_old = knots.size()-k; // Current number of basis functions
    int num_new = si.size()-k; // Number of basis functions after update

    if (num_old < num_new)
        throw Exception("BSplineBasis1D::reduceSupport: Number of basis functions is increased instead of reduced!");

    DenseMatrix Ad = DenseMatrix::Zero(num_new, num_old);
    Ad.block(0, index_lower, num_new, num_new) = DenseMatrix::Identity(num_new, num_new);

    // Update knots
    knots = si;

    return Ad.sparseView();
}

void BSplineBasis1D::setKnots(const KnotVector & knots) {
    this->knots = knots;
    if (!this->knots.is_regular(degree))
        throw Exception("BSplineBasis1D::BSplineBasis1D: Knot vector is not regular.");

}

unsigned int BSplineBasis1D::getNumBasisFunctions() const
{
    return knots.size() - (degree + 1);
}

unsigned int BSplineBasis1D::getNumBasisFunctionsTarget() const
{
    return targetNumBasisfunctions;
}

// Return indices of supporting basis functions at x
std::vector<unsigned int> BSplineBasis1D::indexSupportedBasisFunctions(double x) const
{
    if (!is_supported(x))
        throw Exception("BSplineBasis1D::indexSupportedBasisFunctions: x not inside support!");

    std::vector<unsigned int> supported;
    for (unsigned int i = 0; i < getNumBasisFunctions(); ++i)
    {
        // Support of basis function i
        if (knots.at(i) <= x && x < knots.at(i+degree+1))
        {
            supported.push_back(i);

            if (supported.size() == degree + 1)
                break;
        }
    }

    // Right edge case
    if (assertNear(x, knots.back()) && knotMultiplicity(knots.back()) == degree + 1)
    {
        auto last_basis_func = getNumBasisFunctions()-1;
        if (find(supported.begin(), supported.end(), last_basis_func) == supported.end())
            supported.push_back(last_basis_func);
    }

    if (supported.size() <= 0)
        throw Exception("BSplineBasis1D::indexSupportedBasisFunctions: Number of supporting basis functions is not positive!");

    if (supported.size() > degree + 1)
        throw Exception("BSplineBasis1D::indexSupportedBasisFunctions: Number of supporting basis functions larger than degree + 1!");

    return supported;
}

unsigned int BSplineBasis1D::indexLongestInterval(const std::vector<double> &vec) const
{
    double longest = 0;
    double interval = 0;
    unsigned int index = 0;

    for (unsigned int i = 0; i < vec.size() - 1; i++)
    {
        interval = vec.at(i+1) - vec.at(i);
        if (longest < interval)
        {
            longest = interval;
            index = i;
        }
    }
    return index;
}

} // namespace SPLINTER
