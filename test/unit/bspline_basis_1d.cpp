/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <Catch.h>
#include <bspline_basis_1d.h>

using namespace SPLINTER;

#define COMMON_TAGS "[unit][bsplinebasis1d]"
#define COMMON_TEXT " unit test"

TEST_CASE("supportHack" COMMON_TEXT, COMMON_TAGS)
{
    std::vector<double> knots = {1, 1, 1, 2.1, 3.1, 4, 4, 4};
    BSplineBasis1D bb(knots, 2);

    double x = 4;
    x = bb.supportHack(x);
    REQUIRE(x < 4);
}

TEST_CASE("indexSupportedBasisFunctions" COMMON_TEXT, COMMON_TAGS)
{
    std::vector<double> knots1 = {1, 1, 1, 2.1, 2.5, 2.5, 2.8, 2.8, 2.8, 3.1, 4, 4, 4};
    BSplineBasis1D bb1(knots1, 2);

    {
        double x = 1;
        std::vector<int> ref = {0, 1, 2};
        auto sup = bb1.indexSupportedBasisFunctions(x);

        for (unsigned int i = 0; i < ref.size(); ++i)
            REQUIRE(ref.at(i) == sup.at(i));
    }

    {
        double x = 2.5;
        std::vector<int> ref = {3, 4, 5};
        auto sup = bb1.indexSupportedBasisFunctions(x);

        for (unsigned int i = 0; i < ref.size(); ++i)
            REQUIRE(ref.at(i) == sup.at(i));
    }

    {
        double x = 3.999;
        std::vector<int> ref = {7, 8, 9};
        auto sup = bb1.indexSupportedBasisFunctions(x);

        for (unsigned int i = 0; i < ref.size(); ++i)
            REQUIRE(ref.at(i) == sup.at(i));
    }

    std::vector<double> knots2 = {0, 1, 2.1, 2.5, 2.8, 2.8, 3.1, 4, 4};
    BSplineBasis1D bb2(knots2, 1);

    {
        double x = 0;
        std::vector<int> ref = {0};
        auto sup = bb2.indexSupportedBasisFunctions(x);

        for (unsigned int i = 0; i < ref.size(); ++i)
            REQUIRE(ref.at(i) == sup.at(i));

    }

    {
        double x = 2.499999999;
        std::vector<int> ref = {1, 2};
        auto sup = bb2.indexSupportedBasisFunctions(x);

        for (unsigned int i = 0; i < ref.size(); ++i)
            REQUIRE(ref.at(i) == sup.at(i));

    }
}

TEST_CASE("knotDerivatives" COMMON_TEXT, COMMON_TAGS)
{
    std::vector<double> knots = {1, 1.1, 1.2, 2.1, 3.1, 4, 4.1, 4.2};
    BSplineBasis1D bb(knots, 2);
    double x = 2.2;
    SparseMatrix jac = bb.evalKnotDerivative(x,0);
    
    SparseMatrix jac_ref(jac.rows(),jac.cols());
    double delta = 1e-6;
    SparseVector center = bb.eval(x);
    for(std::size_t knot_idx=0;knot_idx<knots.size();knot_idx++) {
        std::vector<double> perturbed_knots(knots);
        perturbed_knots.at(knot_idx) += delta;
        bb.setKnots(perturbed_knots);
        SparseVector perturbed = bb.eval(x);
        jac_ref.middleCols(knot_idx,1) = (perturbed - center)/delta;
    }
    
    DenseMatrix delta_jac = jac - jac_ref;
    delta_jac = delta_jac.array().square();
    std::cout << "analytic jacobian\n" << jac << std::endl;
    std::cout << "numeric jacobian\n" << jac_ref << std::endl;
    std::cout << "delta\n" << delta_jac << std::endl;
    for(int row=0;row<delta_jac.rows();row++)
        for(int col=0;col<delta_jac.cols();col++)
            REQUIRE(delta_jac(row,col) < 1e-3);
}
