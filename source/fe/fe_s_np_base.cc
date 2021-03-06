// ---------------------------------------------------------------------
//
// Copyright (C) 2000 - 2014 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/base/derivative_form.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/qprojector.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/serendipity_polynomials.h>
#include <deal.II/fe/fe_s_np_base.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <vector>
#include <sstream>

DEAL_II_NAMESPACE_OPEN

template<class POLY, int dim, int spacedim>
const unsigned int FE_S_NP_Base<POLY,dim,spacedim>::n_shape_functions;

template<class POLY, int dim, int spacedim>
FE_S_NP_Base<POLY,dim,spacedim>::InternalData::InternalData (const unsigned int n_shape_functions)
  :
  n_shape_functions (n_shape_functions)
{}


namespace FE_S_NP_Helper
{
  namespace
  {
    // get the renumbering for faces
    template <int dim>
    inline
    std::vector<unsigned int>
    face_lexicographic_to_hierarchic_numbering (const unsigned int degree)
    {
      std::vector<unsigned int> dpo(dim, 1U);

      for (unsigned int i=1; i<dpo.size(); ++i)
      {
        if(degree/2 >= i)
          dpo[i]=Utilities::n_choose_k(degree-i, i);
        else
          dpo[i]=0U;
      }
      const dealii::FiniteElementData<dim-1> face_data(dpo,1,degree);
      std::vector<unsigned int> face_renumber (face_data.dofs_per_cell);

      // FETools::lexicographic_to_hierarchic_numbering (face_data, face_renumber);
      return face_renumber;
    }

    // dummy specialization for dim == 1 to avoid linker errors
    template <>
    inline
    std::vector<unsigned int>
    face_lexicographic_to_hierarchic_numbering<1> (const unsigned int)
    {
      return std::vector<unsigned int>();
    }



    // in get_restriction_matrix() and get_prolongation_matrix(), want to undo
    // tensorization on inner loops for performance reasons. this clears a
    // dim-array
    template <int dim>
    inline
    void
    zero_indices (unsigned int (&indices)[dim])
    {
      for (unsigned int d=0; d<dim; ++d)
        indices[d] = 0;
    }



    // in get_restriction_matrix() and get_prolongation_matrix(), want to undo
    // tensorization on inner loops for performance reasons. this increments
    // tensor product indices
    template <int dim>
    inline
    void
    increment_indices (unsigned int       (&indices)[dim],
                       const unsigned int   dofs1d)
    {
      ++indices[0];
      for (int d=0; d<dim-1; ++d)
        if (indices[d]==dofs1d)
          {
            indices[d] = 0;
            indices[d+1]++;
          }
    }
  }
}



/**
 * A class with the same purpose as the similarly named class of the
 * Triangulation class. See there for more information.
 */
template <class POLY, int xdim, int xspacedim>
struct FE_S_NP_Base<POLY,xdim,xspacedim>::Implementation
{
  /**
   * Initialize the hanging node constraints matrices. Called from the
   * constructor in case the finite element is based on quadrature points.
   */
  template <int spacedim>
  static
  void initialize_constraints (const std::vector<Point<1> > &,
                               FE_S_NP_Base<POLY,1,spacedim> &)
  {
    // no constraints in 1d
  }


  template <int spacedim>
  static
  void initialize_constraints (const std::vector<Point<1> > &/*points*/,
                               FE_S_NP_Base<POLY,2,spacedim> &fe)
  {
    const unsigned int dim = 2;

    // restricted to each face, the traces of the shape functions is an
    // element of P_{k} (in 2d), or Q_{k} (in 3d), where k is the degree of
    // the element.  from this, we interpolate between mother and cell face.

    // the interpolation process works as follows: on each subface, we want
    // that finite element solutions from both sides coincide. i.e. if a and b
    // are expansion coefficients for the shape functions from both sides, we
    // seek a relation between a and b such that
    //   sum_j a_j phi^c_j(x) == sum_j b_j phi_j(x)
    // for all points x on the interface. here, phi^c_j are the shape
    // functions on the small cell on one side of the face, and phi_j those on
    // the big cell on the other side. To get this relation, it suffices to
    // look at a sufficient number of points for which this has to hold. if
    // there are n functions, then we need n evaluation points, and we choose
    // them equidistantly.
    //
    // we obtain the matrix system
    //    A a  ==  B b
    // where
    //    A_ij = phi^c_j(x_i)
    //    B_ij = phi_j(x_i)
    // and the relation we are looking for is
    //    a = A^-1 B b
    //
    // for the special case of Lagrange interpolation polynomials, A_ij
    // reduces to delta_ij, and
    //    a_i = B_ij b_j
    // Hence, interface_constraints(i,j)=B_ij.
    //
    // for the general case, where we don't have Lagrange interpolation
    // polynomials, this is a little more complicated. Then we would evaluate
    // at a number of points and invert the interpolation matrix A.
    //
    // Note, that we build up these matrices for all subfaces at once, rather
    // than considering them separately. the reason is that we finally will
    // want to have them in this order anyway, as this is the format we need
    // inside deal.II

    // In the following the points x_i are constructed in following order
    // (n=degree-1)
    // *----------*---------*
    //     1..n   0  n+1..2n
    // i.e. first the midpoint of the line, then the support points on subface
    // 0 and on subface 1
    std::vector<Point<dim-1> > constraint_points;
    // Add midpoint
    constraint_points.push_back (Point<dim-1> (0.5));

    if (fe.degree>1)
      {
        const unsigned int n=fe.degree-1;
        const double step=1./fe.degree;
        // subface 0
        for (unsigned int i=1; i<=n; ++i)
          constraint_points.push_back (
            GeometryInfo<dim-1>::child_to_cell_coordinates(Point<dim-1>(i*step),0));
        // subface 1
        for (unsigned int i=1; i<=n; ++i)
          constraint_points.push_back (
            GeometryInfo<dim-1>::child_to_cell_coordinates(Point<dim-1>(i*step),1));
      }

    // Now construct relation between destination (child) and source (mother)
    // dofs.

    fe.interface_constraints
    .TableBase<2,double>::reinit (fe.interface_constraints_size());

    // use that the element evaluates to 1 at index 0 and along the line at
    // zero
    const std::vector<unsigned int> &index_map_inverse =
      fe.poly_space.get_numbering_inverse();
    const std::vector<unsigned int> face_index_map =
      FE_S_NP_Helper::face_lexicographic_to_hierarchic_numbering<dim>(fe.degree);
    Assert(std::abs(fe.poly_space.compute_value(index_map_inverse[0],Point<dim>())
                    - 1.) < 1e-14,
           ExcInternalError());

    for (unsigned int i=0; i<constraint_points.size(); ++i)
      for (unsigned int j=0; j<fe.degree+1; ++j)
        {
          Point<dim> p;
          p[0] = constraint_points[i](0);
          fe.interface_constraints(i,face_index_map[j]) =
            fe.poly_space.compute_value(index_map_inverse[j], p);

          // if the value is small up to round-off, then simply set it to zero
          // to avoid unwanted fill-in of the constraint matrices (which would
          // then increase the number of other DoFs a constrained DoF would
          // couple to)
          if (std::fabs(fe.interface_constraints(i,j)) < 1e-14)
            fe.interface_constraints(i,j) = 0;
        }
  }


  template <int spacedim>
  static
  void initialize_constraints (const std::vector<Point<1> > &/*points*/,
                               FE_S_NP_Base<POLY,3,spacedim> &fe)
  {
    const unsigned int dim = 3;

    // For a detailed documentation of the interpolation see the
    // FE_S_NP_Base<2>::initialize_constraints function.

    // In the following the points x_i are constructed in the order as
    // described in the documentation of the FiniteElement class (fe_base.h),
    // i.e.
    //   *--15--4--16--*
    //   |      |      |
    //   10 19  6  20  12
    //   |      |      |
    //   1--7---0--8---2
    //   |      |      |
    //   9  17  5  18  11
    //   |      |      |
    //   *--13--3--14--*
    std::vector<Point<dim-1> > constraint_points;

    // Add midpoint
    constraint_points.push_back (Point<dim-1> (0.5, 0.5));

    // Add midpoints of lines of "mother-face"
    constraint_points.push_back (Point<dim-1> (0, 0.5));
    constraint_points.push_back (Point<dim-1> (1, 0.5));
    constraint_points.push_back (Point<dim-1> (0.5, 0));
    constraint_points.push_back (Point<dim-1> (0.5, 1));

    if (fe.degree>1)
      {
        const unsigned int n=fe.degree-1;
        const double step=1./fe.degree;
        std::vector<Point<dim-2> > line_support_points(n);
        for (unsigned int i=0; i<n; ++i)
          line_support_points[i](0)=(i+1)*step;
        Quadrature<dim-2> qline(line_support_points);

        // auxiliary points in 2d
        std::vector<Point<dim-1> > p_line(n);

        // Add nodes of lines interior in the "mother-face"

        // line 5: use line 9
        QProjector<dim-1>::project_to_subface(qline, 0, 0, p_line);
        for (unsigned int i=0; i<n; ++i)
          constraint_points.push_back (p_line[i] + Point<dim-1> (0.5, 0));
        // line 6: use line 10
        QProjector<dim-1>::project_to_subface(qline, 0, 1, p_line);
        for (unsigned int i=0; i<n; ++i)
          constraint_points.push_back (p_line[i] + Point<dim-1> (0.5, 0));
        // line 7: use line 13
        QProjector<dim-1>::project_to_subface(qline, 2, 0, p_line);
        for (unsigned int i=0; i<n; ++i)
          constraint_points.push_back (p_line[i] + Point<dim-1> (0, 0.5));
        // line 8: use line 14
        QProjector<dim-1>::project_to_subface(qline, 2, 1, p_line);
        for (unsigned int i=0; i<n; ++i)
          constraint_points.push_back (p_line[i] + Point<dim-1> (0, 0.5));

        // DoFs on bordering lines lines 9-16
        for (unsigned int face=0; face<GeometryInfo<dim-1>::faces_per_cell; ++face)
          for (unsigned int subface=0;
               subface<GeometryInfo<dim-1>::max_children_per_face; ++subface)
            {
              QProjector<dim-1>::project_to_subface(qline, face, subface, p_line);
              constraint_points.insert(constraint_points.end(),
                                       p_line.begin(), p_line.end());
            }

        // Create constraints for interior nodes
        std::vector<Point<dim-1> > inner_points(n*n);
        for (unsigned int i=0, iy=1; iy<=n; ++iy)
          for (unsigned int ix=1; ix<=n; ++ix)
            inner_points[i++] = Point<dim-1> (ix*step, iy*step);

        // at the moment do this for isotropic face refinement only
        for (unsigned int child=0;
             child<GeometryInfo<dim-1>::max_children_per_cell; ++child)
          for (unsigned int i=0; i<inner_points.size(); ++i)
            constraint_points.push_back (
              GeometryInfo<dim-1>::child_to_cell_coordinates(inner_points[i], child));
      }

    // Now construct relation between destination (child) and source (mother)
    // dofs.
    const unsigned int pnts=(fe.degree+1)*(fe.degree+1);

    // use that the element evaluates to 1 at index 0 and along the line at
    // zero
    const std::vector<unsigned int> &index_map_inverse =
      fe.poly_space.get_numbering_inverse();
    const std::vector<unsigned int> face_index_map =
      FE_S_NP_Helper::face_lexicographic_to_hierarchic_numbering<dim>(fe.degree);
    Assert(std::abs(fe.poly_space.compute_value(index_map_inverse[0],Point<dim>())
                    - 1.) < 1e-14,
           ExcInternalError());

    fe.interface_constraints
    .TableBase<2,double>::reinit (fe.interface_constraints_size());

    for (unsigned int i=0; i<constraint_points.size(); ++i)
      {
        const double interval = (double) (fe.degree * 2);
        bool mirror[dim - 1];
        Point<dim> constraint_point;

        // Eliminate FP errors in constraint points. Due to their origin, they
        // must all be fractions of the unit interval. If we have polynomial
        // degree 4, the refined element has 8 intervals.  Hence the
        // coordinates must be 0, 0.125, 0.25, 0.375 etc.  Now the coordinates
        // of the constraint points will be multiplied by the inverse of the
        // interval size (in the example by 8).  After that the coordinates
        // must be integral numbers. Hence a normal truncation is performed
        // and the coordinates will be scaled back. The equal treatment of all
        // coordinates should eliminate any FP errors.
        for (unsigned int k=0; k<dim-1; ++k)
          {
            const int coord_int =
              static_cast<int> (constraint_points[i](k) * interval + 0.25);
            constraint_point(k) = 1.*coord_int / interval;

            // The following lines of code should eliminate the problems with
            // the Constraint-Matrix, which appeared for P>=4. The
            // ConstraintMatrix class complained about different constraints
            // for the same entry of the Constraint-Matrix.  Actually this
            // difference could be attributed to FP errors, as it was in the
            // range of 1.0e-16. These errors originate in the loss of
            // symmetry in the FP approximation of the shape-functions.
            // Considering a 3rd order shape function in 1D, we have
            // N0(x)=N3(1-x) and N1(x)=N2(1-x).  For higher order polynomials
            // the FP approximations of the shape functions do not satisfy
            // these equations any more!  Thus in the following code
            // everything is computed in the interval x \in [0..0.5], which is
            // sufficient to express all values that could come out from a
            // computation of any shape function in the full interval
            // [0..1]. If x > 0.5 the computation is done for 1-x with the
            // shape function N_{p-n} instead of N_n.  Hence symmetry is
            // preserved and everything works fine...
            //
            // For a different explanation of the problem, see the discussion
            // in the FiniteElement class for constraint matrices in 3d.
            mirror[k] = (constraint_point(k) > 0.5);
            if (mirror[k])
              constraint_point(k) = 1.0 - constraint_point(k);
          }

        for (unsigned int j=0; j<pnts; ++j)
          {
            unsigned int indices[2] = { j % (fe.degree+1), j / (fe.degree+1) };

            for (unsigned int k = 0; k<2; ++k)
              if (mirror[k])
                indices[k] = fe.degree - indices[k];

            const unsigned int
            new_index = indices[1] * (fe.degree + 1) + indices[0];

            fe.interface_constraints(i,face_index_map[j]) =
              fe.poly_space.compute_value (index_map_inverse[new_index],
                                           constraint_point);

            // if the value is small up to round-off, then simply set it to
            // zero to avoid unwanted fill-in of the constraint matrices
            // (which would then increase the number of other DoFs a
            // constrained DoF would couple to)
            if (std::fabs(fe.interface_constraints(i,j)) < 1e-14)
              fe.interface_constraints(i,j) = 0;
          }
      }
  }
};



template <class POLY, int dim, int spacedim>
FE_S_NP_Base<POLY,dim,spacedim>::FE_S_NP_Base (const POLY &poly_space,
                                         const FiniteElementData<dim> &fe_data,
                                         const std::vector<bool> &restriction_is_additive_flags)
  :
  FE_Poly<POLY,dim,spacedim>(poly_space, fe_data, restriction_is_additive_flags,
                            std::vector<ComponentMask>(1, std::vector<bool>(1,true)))
{}



template <class POLY, int dim, int spacedim>
void
FE_S_NP_Base<POLY,dim,spacedim>::initialize (const std::vector<Point<1> > &points)
{
  Assert (points[0][0] == 0,
          ExcMessage ("The first support point has to be zero."));
  Assert (points.back()[0] == 1,
          ExcMessage ("The last support point has to be one."));

  // const unsigned int q_dofs_per_cell = Utilities::fixed_power<dim>(this->degree+1);

  unsigned int d = ( dim > (this->degree)/2 )? (this->degree)/2 : dim;

  unsigned int q_dofs_per_cell = 0;
  for(unsigned int i=0; i<=d; i++)
  {
    q_dofs_per_cell += (unsigned int)std::pow(2,dim-i)* 
    Utilities::n_choose_k(dim, i)*
    Utilities::n_choose_k(this->degree-i, i);
  }

  Assert(q_dofs_per_cell == this->dofs_per_cell, ExcInternalError());

  // don't need to renumber for serendipity polynomials
  // {
  //   std::vector<unsigned int> renumber(q_dofs_per_cell);
  //   const FiniteElementData<dim> fe(get_dpo_vector(this->degree),1,
  //                                   this->degree);
  //   FETools::hierarchic_to_lexicographic_numbering (fe, renumber);
  //   if (this->dofs_per_cell > q_dofs_per_cell)
  //     renumber.push_back(q_dofs_per_cell);
  //   this->poly_space.set_numbering(renumber);
  // }

  // finally fill in support points on cell and face
  initialize_unit_support_points (points);
  initialize_unit_face_support_points (points);

  // reinit constraints
  // initialize_constraints (points);

  // do not initialize embedding and restriction here. these matrices are
  // initialized on demand in get_restriction_matrix and
  // get_prolongation_matrix

  this->initialize_quad_dof_index_permutation();
}



template <class POLY, int dim, int spacedim>
void
FE_S_NP_Base<POLY,dim,spacedim>::
get_interpolation_matrix (const FiniteElement<dim,spacedim> &x_source_fe,
                          FullMatrix<double>       &interpolation_matrix) const
{
  // go through the list of elements we can interpolate from
  if (const FE_S_NP_Base<POLY,dim,spacedim> *source_fe
      = dynamic_cast<const FE_S_NP_Base<POLY,dim,spacedim>*>(&x_source_fe))
    {
      // ok, source is a Q element, so we will be able to do the work
      Assert (interpolation_matrix.m() == this->dofs_per_cell,
              ExcDimensionMismatch (interpolation_matrix.m(),
                                    this->dofs_per_cell));
      Assert (interpolation_matrix.n() == x_source_fe.dofs_per_cell,
              ExcDimensionMismatch (interpolation_matrix.m(),
                                    x_source_fe.dofs_per_cell));

      // only evaluate Q dofs
      const unsigned int q_dofs_per_cell = Utilities::fixed_power<dim>(this->degree+1);
      const unsigned int source_q_dofs_per_cell = Utilities::fixed_power<dim>(source_fe->degree+1);

      // evaluation is simply done by evaluating the other FE's basis functions on
      // the unit support points (FE_Q has the property that the cell
      // interpolation matrix is a unit matrix, so no need to evaluate it and
      // invert it)
      for (unsigned int j=0; j<q_dofs_per_cell; ++j)
        {
          // read in a point on this cell and evaluate the shape functions there
          const Point<dim> p = this->unit_support_points[j];

          // FE_Q element evaluates to 1 in unit support point and to zero in all
          // other points by construction
          Assert(std::abs(this->poly_space.compute_value (j, p)-1.)<1e-13,
                 ExcInternalError());

          for (unsigned int i=0; i<source_q_dofs_per_cell; ++i)
            interpolation_matrix(j,i) = source_fe->poly_space.compute_value (i, p);
        }

      // for FE_Q_DG0, add one last row of identity
      if (q_dofs_per_cell < this->dofs_per_cell)
        {
          AssertDimension(source_q_dofs_per_cell+1, source_fe->dofs_per_cell);
          for (unsigned int i=0; i<source_q_dofs_per_cell; ++i)
            interpolation_matrix(q_dofs_per_cell, i) = 0.;
          for (unsigned int j=0; j<q_dofs_per_cell; ++j)
            interpolation_matrix(j, source_q_dofs_per_cell) = 0.;
          interpolation_matrix(q_dofs_per_cell, source_q_dofs_per_cell) = 1.;
        }

      // cut off very small values
      const double eps = 2e-13*this->degree*dim;
      for (unsigned int i=0; i<this->dofs_per_cell; ++i)
        for (unsigned int j=0; j<source_fe->dofs_per_cell; ++j)
          if (std::fabs(interpolation_matrix(i,j)) < eps)
            interpolation_matrix(i,j) = 0.;

      // make sure that the row sum of each of the matrices is 1 at this
      // point. this must be so since the shape functions sum up to 1
      for (unsigned int i=0; i<this->dofs_per_cell; ++i)
        {
          double sum = 0.;
          for (unsigned int j=0; j<source_fe->dofs_per_cell; ++j)
            sum += interpolation_matrix(i,j);

          Assert (std::fabs(sum-1) < eps, ExcInternalError());
        }
    }
  else if (dynamic_cast<const FE_Nothing<dim>*>(&x_source_fe))
    {
      // the element we want to interpolate from is an FE_Nothing. this
      // element represents a function that is constant zero and has no
      // degrees of freedom, so the interpolation is simply a multiplication
      // with a n_dofs x 0 matrix. there is nothing to do here

      // we would like to verify that the number of rows and columns of
      // the matrix equals this->dofs_per_cell and zero. unfortunately,
      // whenever we do FullMatrix::reinit(m,0), it sets both rows and
      // columns to zero, instead of m and zero. thus, only test the
      // number of columns
      Assert (interpolation_matrix.n() == x_source_fe.dofs_per_cell,
              ExcDimensionMismatch (interpolation_matrix.m(),
                                    x_source_fe.dofs_per_cell));

    }
  else
    AssertThrow (false,
                 (typename FiniteElement<dim,spacedim>::ExcInterpolationNotImplemented()));

}



template <class POLY, int dim, int spacedim>
void
FE_S_NP_Base<POLY,dim,spacedim>::
get_face_interpolation_matrix (const FiniteElement<dim,spacedim> &source_fe,
                               FullMatrix<double>       &interpolation_matrix) const
{
  Assert (dim > 1, ExcImpossibleInDim(1));
  get_subface_interpolation_matrix (source_fe, numbers::invalid_unsigned_int,
                                    interpolation_matrix);
}



template <class POLY, int dim, int spacedim>
void
FE_S_NP_Base<POLY,dim,spacedim>::
get_subface_interpolation_matrix (const FiniteElement<dim,spacedim> &x_source_fe,
                                  const unsigned int        subface,
                                  FullMatrix<double>       &interpolation_matrix) const
{
  Assert (interpolation_matrix.m() == x_source_fe.dofs_per_face,
          ExcDimensionMismatch (interpolation_matrix.m(),
                                x_source_fe.dofs_per_face));

  // see if source is a Q element
  if (const FE_S_NP_Base<POLY,dim,spacedim> *source_fe
      = dynamic_cast<const FE_S_NP_Base<POLY,dim,spacedim> *>(&x_source_fe))
    {
      // have this test in here since a table of size 2x0 reports its size as
      // 0x0
      Assert (interpolation_matrix.n() == this->dofs_per_face,
              ExcDimensionMismatch (interpolation_matrix.n(),
                                    this->dofs_per_face));

      // Make sure that the element for which the DoFs should be constrained
      // is the one with the higher polynomial degree.  Actually the procedure
      // will work also if this assertion is not satisfied. But the matrices
      // produced in that case might lead to problems in the hp procedures,
      // which use this method.
      Assert (this->dofs_per_face <= source_fe->dofs_per_face,
              (typename FiniteElement<dim,spacedim>::
               ExcInterpolationNotImplemented ()));

      // generate a point on this cell and evaluate the shape functions there
      const Quadrature<dim-1>
      quad_face_support (source_fe->get_unit_face_support_points ());

      // Rule of thumb for FP accuracy, that can be expected for a given
      // polynomial degree.  This value is used to cut off values close to
      // zero.
      double eps = 2e-13*this->degree*(dim-1);

      // compute the interpolation matrix by simply taking the value at the
      // support points.
//TODO: Verify that all faces are the same with respect to
// these support points. Furthermore, check if something has to
// be done for the face orientation flag in 3D.
      const Quadrature<dim> subface_quadrature
        = subface == numbers::invalid_unsigned_int
          ?
          QProjector<dim>::project_to_face (quad_face_support, 0)
          :
          QProjector<dim>::project_to_subface (quad_face_support, 0, subface);
      for (unsigned int i=0; i<source_fe->dofs_per_face; ++i)
        {
          const Point<dim> &p = subface_quadrature.point (i);

          for (unsigned int j=0; j<this->dofs_per_face; ++j)
            {
              double matrix_entry = this->shape_value (this->face_to_cell_index(j, 0), p);

              // Correct the interpolated value. I.e. if it is close to 1 or
              // 0, make it exactly 1 or 0. Unfortunately, this is required to
              // avoid problems with higher order elements.
              if (std::fabs (matrix_entry - 1.0) < eps)
                matrix_entry = 1.0;
              if (std::fabs (matrix_entry) < eps)
                matrix_entry = 0.0;

              interpolation_matrix(i,j) = matrix_entry;
            }
        }

      // make sure that the row sum of each of the matrices is 1 at this
      // point. this must be so since the shape functions sum up to 1
      for (unsigned int j=0; j<source_fe->dofs_per_face; ++j)
        {
          double sum = 0.;

          for (unsigned int i=0; i<this->dofs_per_face; ++i)
            sum += interpolation_matrix(j,i);

          Assert (std::fabs(sum-1) < eps, ExcInternalError());
        }
    }
  else if (dynamic_cast<const FE_Nothing<dim> *>(&x_source_fe) != 0)
    {
      // nothing to do here, the FE_Nothing has no degrees of freedom anyway
    }
  else
    AssertThrow (false,(typename FiniteElement<dim,spacedim>::
                        ExcInterpolationNotImplemented()));
}



template <class POLY, int dim, int spacedim>
bool
FE_S_NP_Base<POLY,dim,spacedim>::hp_constraints_are_implemented () const
{
  return false;
}




template <class POLY, int dim, int spacedim>
std::vector<std::pair<unsigned int, unsigned int> >
FE_S_NP_Base<POLY,dim,spacedim>::
hp_vertex_dof_identities (const FiniteElement<dim,spacedim> &fe_other) const
{
  Assert (false, ExcNotImplemented());
  return std::vector<std::pair<unsigned int, unsigned int> > ();
}



template <class POLY, int dim, int spacedim>
std::vector<std::pair<unsigned int, unsigned int> >
FE_S_NP_Base<POLY,dim,spacedim>::
hp_line_dof_identities (const FiniteElement<dim,spacedim> &fe_other) const
{
  Assert (false, ExcNotImplemented());
  return std::vector<std::pair<unsigned int, unsigned int> > ();
}



template <class POLY, int dim, int spacedim>
std::vector<std::pair<unsigned int, unsigned int> >
FE_S_NP_Base<POLY,dim,spacedim>::
hp_quad_dof_identities (const FiniteElement<dim,spacedim>        &fe_other) const
{
  Assert (false, ExcNotImplemented());
  return std::vector<std::pair<unsigned int, unsigned int> > ();
}



template <class POLY, int dim, int spacedim>
FiniteElementDomination::Domination
FE_S_NP_Base<POLY,dim,spacedim>::
compare_for_face_domination (const FiniteElement<dim,spacedim> &fe_other) const
{
  Assert (false, ExcNotImplemented());
  return FiniteElementDomination::neither_element_dominates;
}


//---------------------------------------------------------------------------
// Auxiliary functions
//---------------------------------------------------------------------------



template <class POLY, int dim, int spacedim>
void FE_S_NP_Base<POLY,dim,spacedim>::initialize_unit_support_points (const std::vector<Point<1> > &points)
{
  // const std::vector<unsigned int> &index_map_inverse=
  //   this->poly_space.get_numbering_inverse();

  // Quadrature<1> support_1d(points);
  // Quadrature<dim> support_quadrature(support_1d);
  // this->unit_support_points.resize(support_quadrature.size());

  // for (unsigned int k=0; k<support_quadrature.size(); k++)
  //   this->unit_support_points[index_map_inverse[k]] = support_quadrature.point(k);
}



template <class POLY, int dim, int spacedim>
void FE_S_NP_Base<POLY,dim,spacedim>::initialize_unit_face_support_points (const std::vector<Point<1> > &points)
{
  // no faces in 1d, so nothing to do
  // if (dim == 1)
  //   return;

  // const unsigned int codim = dim-1;
  // this->unit_face_support_points.resize(Utilities::fixed_power<codim>(this->degree+1));

  // // find renumbering of faces and assign from values of quadrature
  // std::vector<unsigned int> face_index_map =
  //   FE_S_NP_Helper::face_lexicographic_to_hierarchic_numbering<dim>(this->degree);
  // Quadrature<1> support_1d(points);
  // Quadrature<codim> support_quadrature(support_1d);
  // this->unit_face_support_points.resize(support_quadrature.size());

  // for (unsigned int k=0; k<support_quadrature.size(); k++)
  //   this->unit_face_support_points[face_index_map[k]] = support_quadrature.point(k);
}



template <class POLY, int dim, int spacedim>
void
FE_S_NP_Base<POLY,dim,spacedim>::initialize_quad_dof_index_permutation ()
{
  // for 1D and 2D, do nothing
  if (dim < 3)
    return;

  // // Assert (this->adjust_quad_dof_index_for_face_orientation_table.n_elements()==8*this->dofs_per_quad,
  //         ExcInternalError());

  // const unsigned int n=this->degree-1;
  // Assert(n*n==this->dofs_per_quad, ExcInternalError());

  // // alias for the table to fill
  // Table<2,int> &data=this->adjust_quad_dof_index_for_face_orientation_table;

  // // the dofs on a face are connected to a n x n matrix. for example, for
  // // degree==4 we have the following dofs on a quad

  // //  ___________
  // // |           |
  // // |  6  7  8  |
  // // |           |
  // // |  3  4  5  |
  // // |           |
  // // |  0  1  2  |
  // // |___________|
  // //
  // // we have dof_no=i+n*j with index i in x-direction and index j in
  // // y-direction running from 0 to n-1.  to extract i and j we can use
  // // i=dof_no%n and j=dof_no/n. i and j can then be used to construct the
  // // rotated and mirrored numbers.


  // for (unsigned int local=0; local<this->dofs_per_quad; ++local)
  //   // face support points are in lexicographic ordering with x running
  //   // fastest. invert that (y running fastest)
  //   {
  //     unsigned int i=local%n,
  //                  j=local/n;

  //     // face_orientation=false, face_flip=false, face_rotation=false
  //     data(local,0)=j       + i      *n - local;
  //     // face_orientation=false, face_flip=false, face_rotation=true
  //     data(local,1)=i       + (n-1-j)*n - local;
  //     // face_orientation=false, face_flip=true,  face_rotation=false
  //     data(local,2)=(n-1-j) + (n-1-i)*n - local;
  //     // face_orientation=false, face_flip=true,  face_rotation=true
  //     data(local,3)=(n-1-i) + j      *n - local;
  //     // face_orientation=true,  face_flip=false, face_rotation=false
  //     data(local,4)=0;
  //     // face_orientation=true,  face_flip=false, face_rotation=true
  //     data(local,5)=j       + (n-1-i)*n - local;
  //     // face_orientation=true,  face_flip=true,  face_rotation=false
  //     data(local,6)=(n-1-i) + (n-1-j)*n - local;
  //     // face_orientation=true,  face_flip=true,  face_rotation=true
  //     data(local,7)=(n-1-j) + i      *n - local;
  //   }

  // // aditionally initialize reordering of line dofs
  // for (unsigned int i=0; i<this->dofs_per_line; ++i)
  //   this->adjust_line_dof_index_for_line_orientation_table[i]=this->dofs_per_line-1-i - i;
}



template <class POLY, int dim, int spacedim>
unsigned int
FE_S_NP_Base<POLY,dim,spacedim>::
face_to_cell_index (const unsigned int face_index,
                    const unsigned int face,
                    const bool face_orientation,
                    const bool face_flip,
                    const bool face_rotation) const
{
  Assert (face_index < this->dofs_per_face,
          ExcIndexRange(face_index, 0, this->dofs_per_face));
  Assert (face < GeometryInfo<dim>::faces_per_cell,
          ExcIndexRange(face, 0, GeometryInfo<dim>::faces_per_cell));

//TODO: we could presumably solve the 3d case below using the
// adjust_quad_dof_index_for_face_orientation_table field. for the
// 2d case, we can't use adjust_line_dof_index_for_line_orientation_table
// since that array is empty (presumably because we thought that
// there are no flipped edges in 2d, but these can happen in
// DoFTools::make_periodicity_constraints, for example). so we
// would need to either fill this field, or rely on derived classes
// implementing this function, as we currently do

  // we need to distinguish between DoFs on vertices, lines and in 3d quads.
  // do so in a sequence of if-else statements
  if (face_index < this->first_face_line_index)
    // DoF is on a vertex
    {
      // get the number of the vertex on the face that corresponds to this DoF,
      // along with the number of the DoF on this vertex
      const unsigned int face_vertex         = face_index / this->dofs_per_vertex;
      const unsigned int dof_index_on_vertex = face_index % this->dofs_per_vertex;

      // then get the number of this vertex on the cell and translate
      // this to a DoF number on the cell
      return (GeometryInfo<dim>::face_to_cell_vertices(face, face_vertex,
                                                       face_orientation,
                                                       face_flip,
                                                       face_rotation)
              * this->dofs_per_vertex
              +
              dof_index_on_vertex);
    }
  else if (face_index < this->first_face_quad_index)
    // DoF is on a face
    {
      // do the same kind of translation as before. we need to only consider
      // DoFs on the lines, i.e., ignoring those on the vertices
      const unsigned int index = face_index - this->first_face_line_index;

      const unsigned int face_line         = index / this->dofs_per_line;
      const unsigned int dof_index_on_line = index % this->dofs_per_line;

      // we now also need to adjust the line index for the case of
      // face orientation, face flips, etc
      unsigned int adjusted_dof_index_on_line;
      switch (dim)
        {
        case 1:
          Assert (false, ExcInternalError());

        case 2:
          // in 2d, only face_flip has a meaning. if it is set, consider
          // dofs in reverse order
          if (face_flip == false)
            adjusted_dof_index_on_line = dof_index_on_line;
          else
            adjusted_dof_index_on_line = this->dofs_per_line - dof_index_on_line - 1;
          break;

        case 3:
          // in 3d, things are difficult. someone will have to think
          // about how this code here should look like, by drawing a bunch
          // of pictures of how all the faces can look like with the various
          // flips and rotations.
          //
          // that said, the Q2 case is easy enough to implement, as is the case
          // where everything is in standard orientation
          Assert ((this->dofs_per_line <= 1) ||
                  ((face_orientation == true) &&
                   (face_flip == false) &&
                   (face_rotation == false)),
                  ExcNotImplemented());
          adjusted_dof_index_on_line = dof_index_on_line;
          break;
        }

      return (this->first_line_index
              + GeometryInfo<dim>::face_to_cell_lines(face, face_line,
                                                      face_orientation,
                                                      face_flip,
                                                      face_rotation)
              * this->dofs_per_line
              +
              adjusted_dof_index_on_line);
    }
  else
    // DoF is on a quad
    {
      Assert (dim >= 3, ExcInternalError());

      // ignore vertex and line dofs
      const unsigned int index = face_index - this->first_face_quad_index;

      // the same is true here as above for the 3d case -- someone will
      // just have to draw a bunch of pictures. in the meantime,
      // we can implement the Q2 case in which it is simple
      Assert ((this->dofs_per_quad <= 1) ||
              ((face_orientation == true) &&
               (face_flip == false) &&
               (face_rotation == false)),
              ExcNotImplemented());
      return (this->first_quad_index
              + face * this->dofs_per_quad
              + index);
    }
}




template <class POLY, int dim, int spacedim>
std::vector<unsigned int>
FE_S_NP_Base<POLY,dim,spacedim>::get_dpo_vector(const unsigned int deg)
{
  AssertThrow(deg>0,ExcMessage("FE_Q needs to be of degree > 0."));
  std::vector<unsigned int> dpo(dim+1, 1U);
  for (unsigned int i=1; i<dpo.size(); ++i)
  {
    if(deg/2 >= i)
      dpo[i]=Utilities::n_choose_k(deg-i, i);
    else
      dpo[i]=0U;
  }
  return dpo;
}



template <class POLY, int dim, int spacedim>
void
FE_S_NP_Base<POLY,dim,spacedim>::initialize_constraints (const std::vector<Point<1> > &points)
{
  Implementation::initialize_constraints (points, *this);
}



template <class POLY, int dim, int spacedim>
const FullMatrix<double> &
FE_S_NP_Base<POLY,dim,spacedim>
::get_prolongation_matrix (const unsigned int child,
                           const RefinementCase<dim> &refinement_case) const
{
  Assert (refinement_case<RefinementCase<dim>::isotropic_refinement+1,
          ExcIndexRange(refinement_case,0,RefinementCase<dim>::isotropic_refinement+1));
  Assert (refinement_case!=RefinementCase<dim>::no_refinement,
          ExcMessage("Prolongation matrices are only available for refined cells!"));
  Assert (child<GeometryInfo<dim>::n_children(refinement_case),
          ExcIndexRange(child,0,GeometryInfo<dim>::n_children(refinement_case)));

  // initialization upon first request
  if (this->prolongation[refinement_case-1][child].n() == 0)
    {
      Threads::Mutex::ScopedLock lock(this->mutex);

      // if matrix got updated while waiting for the lock
      if (this->prolongation[refinement_case-1][child].n() ==
          this->dofs_per_cell)
        return this->prolongation[refinement_case-1][child];

      // distinguish q/q_dg0 case: only treat Q dofs first
      const unsigned int q_dofs_per_cell = Utilities::fixed_power<dim>(this->degree+1);

      // compute the interpolation matrices in much the same way as we do for
      // the constraints. it's actually simpler here, since we don't have this
      // weird renumbering stuff going on. The trick is again that we the
      // interpolation matrix is formed by a permutation of the indices of the
      // cell matrix. The value eps is used a threshold to decide when certain
      // evaluations of the Lagrange polynomials are zero or one.
      const double eps = 1e-15*this->degree*dim;

#ifdef DEBUG
      // in DEBUG mode, check that the evaluation of support points in the
      // current numbering gives the identity operation
      for (unsigned int i=0; i<q_dofs_per_cell; ++i)
        {
          Assert (std::fabs (1.-this->poly_space.compute_value
                             (i, this->unit_support_points[i])) < eps,
                  ExcInternalError());
          for (unsigned int j=0; j<q_dofs_per_cell; ++j)
            if (j!=i)
              Assert (std::fabs (this->poly_space.compute_value
                                 (i, this->unit_support_points[j])) < eps,
                      ExcInternalError());
        }
#endif

      // to efficiently evaluate the polynomial at the subcell, make use of
      // the tensor product structure of this element and only evaluate 1D
      // information from the polynomial. This makes the cost of this function
      // almost negligible also for high order elements
      const unsigned int dofs1d = this->degree+1;
      std::vector<Table<2,double> >
      subcell_evaluations (dim, Table<2,double>(dofs1d, dofs1d));
      const std::vector<unsigned int> &index_map_inverse =
        this->poly_space.get_numbering_inverse();

      // helper value: step size how to walk through diagonal and how many
      // points we have left apart from the first dimension
      unsigned int step_size_diag = 0;
      {
        unsigned int factor = 1;
        for (unsigned int d=0; d<dim; ++d)
          {
            step_size_diag += factor;
            factor *= dofs1d;
          }
      }

      FullMatrix<double> prolongate (this->dofs_per_cell, this->dofs_per_cell);

      // go through the points in diagonal to capture variation in all
      // directions simultaneously
      for (unsigned int j=0; j<dofs1d; ++j)
        {
          const unsigned int diag_comp = index_map_inverse[j*step_size_diag];
          const Point<dim> p_subcell = this->unit_support_points[diag_comp];
          const Point<dim> p_cell =
            GeometryInfo<dim>::child_to_cell_coordinates (p_subcell, child,
                                                          refinement_case);
          for (unsigned int i=0; i<dofs1d; ++i)
            for (unsigned int d=0; d<dim; ++d)
              {
                // evaluate along line where only x is different from zero
                Point<dim> point;
                point[0] = p_cell[d];
                const double cell_value =
                  this->poly_space.compute_value(index_map_inverse[i], point);

                // cut off values that are too small. note that we have here
                // Lagrange interpolation functions, so they should be zero at
                // almost all points, and one at the others, at least on the
                // subcells. so set them to their exact values
                //
                // the actual cut-off value is somewhat fuzzy, but it works
                // for 2e-13*degree*dim (see above), which is kind of
                // reasonable given that we compute the values of the
                // polynomials via an degree-step recursion and then multiply
                // the 1d-values. this gives us a linear growth in degree*dim,
                // times a small constant.
                //
                // the embedding matrix is given by applying the inverse of
                // the subcell matrix on the cell_interpolation matrix. since
                // the subcell matrix is actually only a permutation vector,
                // all we need to do is to switch the rows we write the data
                // into. moreover, cut off very small values here
                if (std::fabs(cell_value) < eps)
                  subcell_evaluations[d](j,i) = 0;
                else
                  subcell_evaluations[d](j,i) = cell_value;
              }
        }

      // now expand from 1D info. block innermost dimension (x_0) in order to
      // avoid difficult checks at innermost loop
      unsigned int j_indices[dim];
      FE_S_NP_Helper::zero_indices<dim> (j_indices);
      for (unsigned int j=0; j<q_dofs_per_cell; j+=dofs1d)
        {
          unsigned int i_indices[dim];
          FE_S_NP_Helper::zero_indices<dim> (i_indices);
          for (unsigned int i=0; i<q_dofs_per_cell; i+=dofs1d)
            {
              double val_extra_dim = 1.;
              for (unsigned int d=1; d<dim; ++d)
                val_extra_dim *= subcell_evaluations[d](j_indices[d-1],
                                                        i_indices[d-1]);

              // innermost sum where we actually compute. the same as
              // prolongate(j,i) = this->poly_space.compute_value (i, p_cell)
              for (unsigned int jj=0; jj<dofs1d; ++jj)
                {
                  const unsigned int j_ind = index_map_inverse[j+jj];
                  for (unsigned int ii=0; ii<dofs1d; ++ii)
                    prolongate(j_ind,index_map_inverse[i+ii])
                      = val_extra_dim * subcell_evaluations[0](jj,ii);
                }

              // update indices that denote the tensor product position. a bit
              // fuzzy and therefore not done for innermost x_0 direction
              FE_S_NP_Helper::increment_indices<dim> (i_indices, dofs1d);
            }
          Assert (i_indices[dim-1] == 1, ExcInternalError());
          FE_S_NP_Helper::increment_indices<dim> (j_indices, dofs1d);
        }

      // the discontinuous node is simply mapped on the discontinuous node on
      // the child element
      if (q_dofs_per_cell < this->dofs_per_cell)
        prolongate(q_dofs_per_cell,q_dofs_per_cell) = 1.;

      // and make sure that the row sum is 1. this must be so since for this
      // element, the shape functions add up to one
#ifdef DEBUG
      for (unsigned int row=0; row<this->dofs_per_cell; ++row)
        {
          double sum = 0;
          for (unsigned int col=0; col<this->dofs_per_cell; ++col)
            sum += prolongate(row,col);
          Assert (std::fabs(sum-1.) < eps, ExcInternalError());
        }
#endif

      // swap matrices
      prolongate.swap(const_cast<FullMatrix<double> &>
                      (this->prolongation[refinement_case-1][child]));
    }

  // finally return the matrix
  return this->prolongation[refinement_case-1][child];
}



template <class POLY, int dim, int spacedim>
const FullMatrix<double> &
FE_S_NP_Base<POLY,dim,spacedim>
::get_restriction_matrix (const unsigned int child,
                          const RefinementCase<dim> &refinement_case) const
{
  Assert (refinement_case<RefinementCase<dim>::isotropic_refinement+1,
          ExcIndexRange(refinement_case,0,RefinementCase<dim>::isotropic_refinement+1));
  Assert (refinement_case!=RefinementCase<dim>::no_refinement,
          ExcMessage("Restriction matrices are only available for refined cells!"));
  Assert (child<GeometryInfo<dim>::n_children(refinement_case),
          ExcIndexRange(child,0,GeometryInfo<dim>::n_children(refinement_case)));

  // initialization upon first request
  if (this->restriction[refinement_case-1][child].n() == 0)
    {
      Threads::Mutex::ScopedLock lock(this->mutex);

      // if matrix got updated while waiting for the lock...
      if (this->restriction[refinement_case-1][child].n() ==
          this->dofs_per_cell)
        return this->restriction[refinement_case-1][child];

      FullMatrix<double> restriction(this->dofs_per_cell, this->dofs_per_cell);
      // distinguish q/q_dg0 case
      const unsigned int q_dofs_per_cell = Utilities::fixed_power<dim>(this->degree+1);

      // for these Lagrange interpolation polynomials, construction of the
      // restriction matrices is relatively simple. the reason is that the
      // interpolation points on the mother cell are (except for the case with
      // arbitrary nonequidistant nodes) always also interpolation points for
      // some shape function on one or the other child, because we have chosen
      // equidistant Lagrange interpolation points for the polynomials
      //
      // so the only thing we have to find out is: for each shape function on
      // the mother cell, which is the child cell (possibly more than one) on
      // which it is located, and which is the corresponding shape function
      // there. rather than doing it for the shape functions on the mother
      // cell, we take the interpolation points there
      //
      // note that the interpolation point of a shape function can be on the
      // boundary between subcells. in that case, restriction from children to
      // mother may produce two or more entries for a dof on the mother
      // cell. however, this doesn't hurt: since the element is continuous,
      // the contribution from each child should yield the same result, and
      // since the element is non-additive we just overwrite one value
      // (compute on one child) by the same value (compute on a later child),
      // so we don't have to care about this

      const double eps = 1e-15*this->degree*dim;
      const std::vector<unsigned int> &index_map_inverse =
        this->poly_space.get_numbering_inverse();

      const unsigned int dofs1d = this->degree+1;
      std::vector<Tensor<1,dim> > evaluations1d (dofs1d);

      restriction.reinit(this->dofs_per_cell, this->dofs_per_cell);

      for (unsigned int i=0; i<q_dofs_per_cell; ++i)
        {
          unsigned int mother_dof = index_map_inverse[i];
          const Point<dim> p_cell = this->unit_support_points[mother_dof];

          // check whether this interpolation point is inside this child cell
          const Point<dim> p_subcell
            = GeometryInfo<dim>::cell_to_child_coordinates (p_cell, child,
                                                            refinement_case);
          if (GeometryInfo<dim>::is_inside_unit_cell (p_subcell))
            {
              // same logic as in initialize_embedding to evaluate the
              // polynomial faster than from the tensor product: since we
              // evaluate all polynomials, it is much faster to just compute
              // the 1D values for all polynomials before and then get the
              // dim-data.
              for (unsigned int j=0; j<dofs1d; ++j)
                for (unsigned int d=0; d<dim; ++d)
                  {
                    Point<dim> point;
                    point[0] = p_subcell[d];
                    evaluations1d[j][d] =
                      this->poly_space.compute_value(index_map_inverse[j], point);
                  }
              unsigned int j_indices[dim];
              FE_S_NP_Helper::zero_indices<dim> (j_indices);
              double sum_check = 0;
              for (unsigned int j = 0; j<q_dofs_per_cell; j += dofs1d)
                {
                  double val_extra_dim = 1.;
                  for (unsigned int d=1; d<dim; ++d)
                    val_extra_dim *= evaluations1d[j_indices[d-1]][d];
                  for (unsigned int jj=0; jj<dofs1d; ++jj)
                    {

                      // find the child shape function(s) corresponding to
                      // this point. Usually this is just one function;
                      // however, when we use FE_Q on arbitrary nodes a parent
                      // support point might not be a child support point, and
                      // then we will get more than one nonzero value per
                      // row. Still, the values should sum up to 1
                      const double val
                        = val_extra_dim * evaluations1d[jj][0];
                      const unsigned int child_dof =
                        index_map_inverse[j+jj];
                      if (std::fabs (val-1.) < eps)
                        restriction(mother_dof,child_dof)=1.;
                      else if (std::fabs(val) > eps)
                        restriction(mother_dof,child_dof)=val;
                      sum_check += val;
                    }
                  FE_S_NP_Helper::increment_indices<dim> (j_indices, dofs1d);
                }
              Assert (std::fabs(sum_check-1) < eps,
                      ExcInternalError());
            }

          // part for FE_Q_DG0
          if (q_dofs_per_cell < this->dofs_per_cell)
            restriction(this->dofs_per_cell-1,this->dofs_per_cell-1) =
              1./GeometryInfo<dim>::n_children(RefinementCase<dim>(refinement_case));
        }

      // swap matrices
      restriction.swap(const_cast<FullMatrix<double> &>
                       (this->restriction[refinement_case-1][child]));
    }

  return this->restriction[refinement_case-1][child];
}



//---------------------------------------------------------------------------
// Data field initialization
//---------------------------------------------------------------------------


template <class POLY, int dim, int spacedim>
bool
FE_S_NP_Base<POLY,dim,spacedim>::has_support_on_face (const unsigned int shape_index,
                                                   const unsigned int face_index) const
{
  Assert (shape_index < this->dofs_per_cell,
          ExcIndexRange (shape_index, 0, this->dofs_per_cell));
  Assert (face_index < GeometryInfo<dim>::faces_per_cell,
          ExcIndexRange (face_index, 0, GeometryInfo<dim>::faces_per_cell));

  // in 1d, things are simple. since there is only one degree of freedom per
  // vertex in this class, the first is on vertex 0 (==face 0 in some sense),
  // the second on face 1:
  if (dim == 1)
    return (((shape_index == 0) && (face_index == 0)) ||
            ((shape_index == 1) && (face_index == 1)));

  // first, special-case interior shape functions, since they have no support
  // no-where on the boundary
  if (((dim==2) && (shape_index>=this->first_quad_index))
      ||
      ((dim==3) && (shape_index>=this->first_hex_index)))
    return false;

  // let's see whether this is a vertex
  if (shape_index < this->first_line_index)
    {
      // for Q elements, there is one dof per vertex, so
      // shape_index==vertex_number. check whether this vertex is on the given
      // face. thus, for each face, give a list of vertices
      const unsigned int vertex_no = shape_index;
      Assert (vertex_no < GeometryInfo<dim>::vertices_per_cell,
              ExcInternalError());

      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
        if (GeometryInfo<dim>::face_to_cell_vertices(face_index, v) == vertex_no)
          return true;

      return false;
    }
  else if (shape_index < this->first_quad_index)
    // ok, dof is on a line
    {
      const unsigned int line_index
        = (shape_index - this->first_line_index) / this->dofs_per_line;
      Assert (line_index < GeometryInfo<dim>::lines_per_cell,
              ExcInternalError());

      // in 2d, the line is the face, so get the line index
      if (dim == 2)
        return (line_index == face_index);
      else if (dim == 3)
        {
          // silence compiler warning
          const unsigned int lines_per_face =
            dim == 3 ? GeometryInfo<dim>::lines_per_face : 1;
          // see whether the given line is on the given face.
          for (unsigned int l=0; l<lines_per_face; ++l)
            if (GeometryInfo<3>::face_to_cell_lines(face_index, l) == line_index)
              return true;

          return false;
        }
      else
        Assert (false, ExcNotImplemented());
    }
  else if (shape_index < this->first_hex_index)
    // dof is on a quad
    {
      const unsigned int quad_index
        = (shape_index - this->first_quad_index) / this->dofs_per_quad;
      Assert (static_cast<signed int>(quad_index) <
              static_cast<signed int>(GeometryInfo<dim>::quads_per_cell),
              ExcInternalError());

      // in 2d, cell bubble are zero on all faces. but we have treated this
      // case above already
      Assert (dim != 2, ExcInternalError());

      // in 3d, quad_index=face_index
      if (dim == 3)
        return (quad_index == face_index);
      else
        Assert (false, ExcNotImplemented());
    }
  else
    // dof on hex
    {
      // can only happen in 3d, but this case has already been covered above
      Assert (false, ExcNotImplemented());
      return false;
    }

  // we should not have gotten here
  Assert (false, ExcInternalError());
  return false;
}



template <typename POLY, int dim, int spacedim>
std::pair<Table<2,bool>, std::vector<unsigned int> >
FE_S_NP_Base<POLY,dim,spacedim>::get_constant_modes () const
{
  Table<2,bool> constant_modes(1, this->dofs_per_cell);
  // AssertDimension(this->dofs_per_cell, Utilities::fixed_power<dim>(this->degree+1));
  constant_modes.fill(true);
  return std::pair<Table<2,bool>, std::vector<unsigned int> >
         (constant_modes, std::vector<unsigned int>(1, 0));
}


// ------  add by Zhen Tao -------------

template <typename POLY, int dim, int spacedim>
UpdateFlags
FE_S_NP_Base<POLY,dim,spacedim>::update_once (const UpdateFlags) const
{
  return update_default;
}

template <typename POLY, int dim, int spacedim>
UpdateFlags
FE_S_NP_Base<POLY,dim,spacedim>::update_each (const UpdateFlags flags) const
{
  UpdateFlags out = flags;

  if (flags & (update_values | update_gradients))
    out |= update_quadrature_points | update_covariant_transformation;

  return out;
}

template <class POLY, int dim, int spacedim>
typename Mapping<dim,spacedim>::InternalDataBase *
FE_S_NP_Base<POLY,dim,spacedim>::get_data (
  const UpdateFlags      update_flags,
  const Mapping<dim,spacedim>    &mapping,
  const Quadrature<dim> &quadrature) const
{
  InternalData *data = new InternalData(n_shape_functions);

  data->update_once = update_once(update_flags);
  data->update_each = update_each(update_flags);
  data->update_flags = data->update_once | data->update_each;

  const UpdateFlags flags(data->update_flags);
  const unsigned int n_q_points = quadrature.size();  

  // some scratch arrays
  std::vector<double> values(0);
  std::vector<Tensor<1,dim> > grads(0);
  std::vector<Tensor<2,dim> > grad_grads(0);  

  // if (flags & update_values)
    {
      values.resize (this->dofs_per_cell);
      data->shape_values.resize (this->dofs_per_cell,
                                 std::vector<double> (n_q_points));
    }

  // if (flags & update_gradients)
    {
      grads.resize (this->dofs_per_cell);
      data->shape_gradients.resize (this->dofs_per_cell,
                                    std::vector<Tensor<1,dim> > (n_q_points));
    }

  // if (flags & (update_values | update_gradients))
  for (unsigned int i=0; i<n_q_points; ++i)
    {
      this->poly_space.compute(quadrature.point(i),
                         values, grads, grad_grads);

      // if (flags & update_values)
        for (unsigned int k=0; k<this->dofs_per_cell; ++k)
          data->shape_values[k][i] = values[k];

      // if (flags & update_gradients)
        for (unsigned int k=0; k<this->dofs_per_cell; ++k)
          data->shape_gradients[k][i] = grads[k];
    }  

  data->corner_derivatives.resize(data->n_shape_functions * GeometryInfo<dim>::vertices_per_cell);
  data->corner_values.resize(data->n_shape_functions * GeometryInfo<dim>::vertices_per_cell);
  if(dim==3)
  {
    data->center_derivatives.resize(data->n_shape_functions * 7);
    data->center_values.resize(data->n_shape_functions * 7);    
  }
  compute_shapes (quadrature.get_points(), *data);

  return data;
}

template <class POLY, int dim, int spacedim>
void
FE_S_NP_Base<POLY,dim,spacedim>::compute_shapes (const std::vector<Point<dim> > &unit_points,
                                         InternalData &data) const
{
    FE_S_NP_Base<POLY,dim,spacedim>::compute_shapes_virtual(unit_points, data);
}

namespace internal
{
  namespace FE_S_NP_Base
  {
    template <class POLY, int spacedim>
    void
    compute_shapes_virtual (const unsigned int            n_shape_functions,
                            const std::vector<Point<1> > &unit_points,
                            typename dealii::FE_S_NP_Base<POLY,1,spacedim>::InternalData &data)
    {
      Assert(false, ExcNotImplemented());
    }

    template <class POLY, int spacedim>
    void
    compute_shapes_virtual (const unsigned int            n_shape_functions,
                            const std::vector<Point<2> > &unit_points,
                            typename dealii::FE_S_NP_Base<POLY,2,spacedim>::InternalData &data)
    {
      (void)n_shape_functions;

      for(unsigned int cr = 0 ; cr < 4 ; ++cr )
        {
          int x = cr % 2;
          int y = cr / 2;

          Assert(data.corner_values.size() == n_shape_functions * 4,
            ExcInternalError());

          data.corner_value(cr,0) = (1.-x)*(1.-y);
          data.corner_value(cr,1) = x*(1.-y);
          data.corner_value(cr,2) = (1.-x)*y;
          data.corner_value(cr,3) = x*y;

          Assert(data.corner_derivatives.size() == n_shape_functions * 4,
            ExcInternalError());

          data.corner_derivative(cr,0)[0] = (y-1.);
          data.corner_derivative(cr,1)[0] = (1.-y);
          data.corner_derivative(cr,2)[0] = -y;
          data.corner_derivative(cr,3)[0] = y;

          data.corner_derivative(cr,0)[1] = (x-1.);
          data.corner_derivative(cr,1)[1] = -x;
          data.corner_derivative(cr,2)[1] = (1.-x);
          data.corner_derivative(cr,3)[1] = x;
        }

    } // compute_shapes_virtual 2D end

    template <class POLY, int spacedim>
    void
    compute_shapes_virtual (const unsigned int            n_shape_functions,
                            const std::vector<Point<3> > &unit_points,
                            typename dealii::FE_S_NP_Base<POLY,3,spacedim>::InternalData &data)
    {
      (void)n_shape_functions;      

      for(unsigned int cr = 0 ; cr < 8 ; ++cr )
        {
          int x = cr % 2;
          int y = ((cr - x)/2) % 2;
          int z = (cr < 4) ? 0 : 1;

          Assert(data.corner_derivatives.size() == n_shape_functions * 8,
            ExcInternalError());

          data.corner_derivative(cr,0)[0] = (y-1.)*(1.-z);
          data.corner_derivative(cr,1)[0] = (1.-y)*(1.-z);
          data.corner_derivative(cr,2)[0] = -y*(1.-z);
          data.corner_derivative(cr,3)[0] = y*(1.-z);
          data.corner_derivative(cr,4)[0] = (y-1.)*z;
          data.corner_derivative(cr,5)[0] = (1.-y)*z;
          data.corner_derivative(cr,6)[0] = -y*z;
          data.corner_derivative(cr,7)[0] = y*z;
          data.corner_derivative(cr,0)[1] = (x-1.)*(1.-z);
          data.corner_derivative(cr,1)[1] = -x*(1.-z);
          data.corner_derivative(cr,2)[1] = (1.-x)*(1.-z);
          data.corner_derivative(cr,3)[1] = x*(1.-z);
          data.corner_derivative(cr,4)[1] = (x-1.)*z;
          data.corner_derivative(cr,5)[1] = -x*z;
          data.corner_derivative(cr,6)[1] = (1.-x)*z;
          data.corner_derivative(cr,7)[1] = x*z;
          data.corner_derivative(cr,0)[2] = (x-1)*(1.-y);
          data.corner_derivative(cr,1)[2] = x*(y-1.);
          data.corner_derivative(cr,2)[2] = (x-1.)*y;
          data.corner_derivative(cr,3)[2] = -x*y;
          data.corner_derivative(cr,4)[2] = (1.-x)*(1.-y);
          data.corner_derivative(cr,5)[2] = x*(1.-y);
          data.corner_derivative(cr,6)[2] = (1.-x)*y;
          data.corner_derivative(cr,7)[2] = x*y;
        }

        for(unsigned int cr = 0 ; cr < 8 ; ++cr )
        {
          int x = cr % 2;
          int y = ((cr - x)/2) % 2;
          int z = (cr < 4) ? 0 : 1;

          Assert(data.corner_values.size() == n_shape_functions * 8,
            ExcInternalError());

          data.corner_value(cr,0) = (1.-x)*(1.-y)*(1.-z);
          data.corner_value(cr,1) = x*(1.-y)*(1.-z);
          data.corner_value(cr,2) = (1.-x)*y*(1.-z);
          data.corner_value(cr,3) = x*y*(1.-z);
          data.corner_value(cr,4) = (1.-x)*(1.-y)*z;
          data.corner_value(cr,5) = x*(1.-y)*z;
          data.corner_value(cr,6) = (1.-x)*y*z;
          data.corner_value(cr,7) = x*y*z;
        }

        for(unsigned int cr = 0; cr < 7; ++cr)
        {
          Assert(data.center_values.size() == n_shape_functions * 7,
            ExcInternalError());

          double x = 0.5, y = 0.5, z = 0.5;
          if((cr == 0) || (cr == 1))
            x = (cr==0)? 0.0 : 1.0;
          if ((cr ==2) || (cr==3))
            y = (cr==2)? 0.0 : 1.0;
          if((cr == 4) || (cr==5))
            z = (cr==4)? 0.0 : 1.0;

          data.center_value(cr,0) = (1.-x)*(1.-y)*(1.-z);
          data.center_value(cr,1) = x*(1.-y)*(1.-z);
          data.center_value(cr,2) = (1.-x)*y*(1.-z);
          data.center_value(cr,3) = x*y*(1.-z);
          data.center_value(cr,4) = (1.-x)*(1.-y)*z;
          data.center_value(cr,5) = x*(1.-y)*z;
          data.center_value(cr,6) = (1.-x)*y*z;
          data.center_value(cr,7) = x*y*z;

          data.center_derivative(cr,0)[0] = (y-1.)*(1.-z);
          data.center_derivative(cr,1)[0] = (1.-y)*(1.-z);
          data.center_derivative(cr,2)[0] = -y*(1.-z);
          data.center_derivative(cr,3)[0] = y*(1.-z);
          data.center_derivative(cr,4)[0] = (y-1.)*z;
          data.center_derivative(cr,5)[0] = (1.-y)*z;
          data.center_derivative(cr,6)[0] = -y*z;
          data.center_derivative(cr,7)[0] = y*z;
          data.center_derivative(cr,0)[1] = (x-1.)*(1.-z);
          data.center_derivative(cr,1)[1] = -x*(1.-z);
          data.center_derivative(cr,2)[1] = (1.-x)*(1.-z);
          data.center_derivative(cr,3)[1] = x*(1.-z);
          data.center_derivative(cr,4)[1] = (x-1.)*z;
          data.center_derivative(cr,5)[1] = -x*z;
          data.center_derivative(cr,6)[1] = (1.-x)*z;
          data.center_derivative(cr,7)[1] = x*z;
          data.center_derivative(cr,0)[2] = (x-1)*(1.-y);
          data.center_derivative(cr,1)[2] = x*(y-1.);
          data.center_derivative(cr,2)[2] = (x-1.)*y;
          data.center_derivative(cr,3)[2] = -x*y;
          data.center_derivative(cr,4)[2] = (1.-x)*(1.-y);
          data.center_derivative(cr,5)[2] = x*(1.-y);
          data.center_derivative(cr,6)[2] = (1.-x)*y;
          data.center_derivative(cr,7)[2] = x*y;
        }

    } //compute_shapes_virtual 3D end

  }
}

template <class POLY, int dim, int spacedim>
void
FE_S_NP_Base<POLY,dim,spacedim>::
compute_shapes_virtual (const std::vector<Point<dim> > &unit_points,
                        InternalData &data) const
{
  internal::FE_S_NP_Base::
  compute_shapes_virtual<POLY,spacedim> (n_shape_functions, unit_points, data);
}

template <class POLY, int dim, int spacedim>
void
FE_S_NP_Base<POLY,dim,spacedim>::compute_mapping_support_points(
  const typename Triangulation<dim,spacedim>::cell_iterator &cell,
  std::vector<Point<spacedim> > &a) const
{
  a.resize(GeometryInfo<dim>::vertices_per_cell);

  for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
    a[i] = cell->vertex(i);
}

//---------------------------------------------------------------------------
// Fill data of FEValues
//---------------------------------------------------------------------------

template <class POLY, int dim, int spacedim>
void
FE_S_NP_Base<POLY,dim,spacedim>::fill_fe_values (
  const Mapping<dim,spacedim>                      &mapping,
  const typename Triangulation<dim,spacedim>::cell_iterator &cell,
  const Quadrature<dim>                            &quadrature,
  typename Mapping<dim,spacedim>::InternalDataBase &mapping_data,
  typename Mapping<dim,spacedim>::InternalDataBase &fedata,
  FEValuesData<dim,spacedim>                       &data,
  CellSimilarity::Similarity                  &cell_similarity) const
{
  Assert (dynamic_cast<InternalData *> (&fedata) != 0,
          ExcInternalError());
  InternalData &fe_data = static_cast<InternalData &> (fedata);

  const UpdateFlags flags(fe_data.current_update_flags());

  Assert (flags & update_quadrature_points, ExcInternalError());
  const unsigned int n_q_points = data.quadrature_points.size();

  unsigned int fe_degree = this->degree;
  if(dim==2)
  {
    Assert(fe_degree>1, ExcMessage("2D non parametric serendipity only for degree>1"));
  }else{
    Assert(fe_degree>2, ExcMessage("3D non parametric serendipity only for degree>2"));
  }

  // alternative formulation
  bool flag_new = true;

  // 1) Set face normals
  compute_mapping_support_points(cell, fe_data.mapping_support_points);
  const Tensor<1,spacedim> *supp_pts = &fe_data.mapping_support_points[0];
  std::vector<Tensor<1,dim> > Gamma;
  Gamma.resize(GeometryInfo<dim>::faces_per_cell);
  std::vector<Tensor<1,dim> > Gamma_mid;
  const double measure = cell->measure();
  const double h = std::pow(measure, 1.0/dim);

  if(dim==2)
  {
    std::vector<Tensor<1, dim> >tangential;
    tangential.resize(4);
    for(unsigned int k = 0; k<n_shape_functions; ++k)
      for(unsigned int d = 0; d<dim; ++d)
      { // face 0: J * [0 -1]^T
        tangential[0][d] += -1.*supp_pts[k][d]*fe_data.corner_derivative(0,k)[1];
        // face 1: J * [0 1]^T
        tangential[1][d] += supp_pts[k][d]*fe_data.corner_derivative(1,k)[1];
        // face 2: J * [1 0]^T
        tangential[2][d] += supp_pts[k][d]*fe_data.corner_derivative(0,k)[0];
        // face 3: J * [-1 0]^T
        tangential[3][d] += -1.*supp_pts[k][d]*fe_data.corner_derivative(2,k)[0];
      }

    // std::cout<<"**** cell "<<cell->index()<<" *******"<<std::endl;
    for(unsigned int face_no=0 ; face_no<4; ++face_no)
    {
      cross_product(Gamma[face_no], tangential[face_no]);
      Gamma[face_no] = -Gamma[face_no]/(Gamma[face_no].norm()*h);
      // std::cout<<"Gamma "<<face_no<<":"<< Gamma[face_no]<<std::endl;
    }
  } // end dim==2

  if(dim==3)
  {
    Gamma_mid.resize(dim);

    std::vector<Tensor<1, dim> >tangential;
    tangential.resize(15);

    for(unsigned int k = 0; k<n_shape_functions; ++k)
      for(unsigned int d = 0; d<dim; ++d)
      { 
        // face 0 tang 1: J * [0 -1  0]^T
        tangential[0][d] += -1.*supp_pts[k][d]*fe_data.corner_derivative(0,k)[1];
        // face 0 tang 2: J * [0 0 1]^T
        tangential[1][d] += supp_pts[k][d]*fe_data.corner_derivative(0,k)[2];
        // face 1 tang 1: J * [0 1  0]^T
        tangential[2][d] += supp_pts[k][d]*fe_data.corner_derivative(1,k)[1];
        // face 1 tang 2: J * [0 0 1]^T
        tangential[3][d] += supp_pts[k][d]*fe_data.corner_derivative(1,k)[2];
        // face 2 tang 1: J * [0 0  -1]^T
        tangential[4][d] += -1.*supp_pts[k][d]*fe_data.corner_derivative(0,k)[2];
        // face 2 tang 2: J * [1 0 0]^T
        tangential[5][d] += supp_pts[k][d]*fe_data.corner_derivative(0,k)[0];
        // face 3 tang 1: J * [0 0  1]^T
        tangential[6][d] += supp_pts[k][d]*fe_data.corner_derivative(2,k)[2];
        // face 3 tang 2: J * [1 0 0]^T
        tangential[7][d] += supp_pts[k][d]*fe_data.corner_derivative(2,k)[0];
        // face 4 tang 1: J * [-1 0 0]^T
        tangential[8][d] += -1.*supp_pts[k][d]*fe_data.corner_derivative(0,k)[0];
        // face 4 tang 2: J * [0 1 0]^T
        tangential[9][d] += supp_pts[k][d]*fe_data.corner_derivative(0,k)[1];
        // face 5 tang 1: J * [1 0 0]^T
        tangential[10][d] += supp_pts[k][d]*fe_data.corner_derivative(4,k)[0];
        // face 5 tang 2: J * [0 1 0]^T
        tangential[11][d] += supp_pts[k][d]*fe_data.corner_derivative(4,k)[1]; 

        //tang 1: J* [1 0 0]
        tangential[12][d] += supp_pts[k][d]*fe_data.center_derivative(6,k)[0];
        //tang 2: J* [0 1 0]
        tangential[13][d] += supp_pts[k][d]*fe_data.center_derivative(6,k)[1];
        //tang 3: J* [0 0 1]
        tangential[14][d] += supp_pts[k][d]*fe_data.center_derivative(6,k)[2];
      }

    for(unsigned int face_no=0 ; face_no<6; ++face_no)
    {
      cross_product(Gamma[face_no], tangential[2*face_no], tangential[2*face_no+1]);
      Gamma[face_no] = Gamma[face_no]/(Gamma[face_no].norm());
      // std::cout<<"Gamma "<<face_no<<":"<< Gamma[face_no]<<std::endl;
    }

    cross_product(Gamma_mid[0], tangential[13], tangential[14]);
    Gamma_mid[0] = Gamma_mid[0]/(Gamma_mid[0].norm());

    cross_product(Gamma_mid[1], tangential[14], tangential[12]);
    Gamma_mid[1] = Gamma_mid[1]/(Gamma_mid[1].norm());

    cross_product(Gamma_mid[2], tangential[12], tangential[13]);
    Gamma_mid[2] = Gamma_mid[2]/(Gamma_mid[2].norm());

  } //end dim==3

  FullMatrix<double> B((dim==3)?6:0);
  Tensor<1,dim> center_pt;

  if(dim==3)
  {

    for(unsigned int j=0; j<6; ++j)
      for(unsigned int k=0; k<6; ++k)
      {
        if((j==k)||((j-(j/2)*2)==0 && k == (j+1))||((j-(j/2)*2)==1 && k == (j-1)))
          B[j][k] = 0.0;
        else
        {
          Tensor<1,dim> pj;
          pj = Gamma[j] - (Gamma[j]*Gamma[k])*Gamma[k];
          Assert(pj.norm()>1e-13, ExcMessage("Geometry degenerate!"));
          B[j][k] = 1/pj.norm();
        }
      }

    // reset B[4][2],B[4][0],B[5][2],B[5][0]
    Tensor<1,dim> temp;
    temp = Gamma[4]-(Gamma[4]*Gamma[0])*Gamma[0] - (Gamma[4]*Gamma[2])*Gamma[2];
    B[4][2] = 1/temp.norm();
    B[4][0] = 1/temp.norm();
    B[4][1] = 1/temp.norm();
    B[4][3] = 1/temp.norm();
    temp = Gamma[5]-(Gamma[5]*Gamma[0])*Gamma[0] - (Gamma[5]*Gamma[2])*Gamma[2];
    B[5][2] = 1/temp.norm();
    B[5][0] = 1/temp.norm();
    B[5][1] = 1/temp.norm();
    B[5][3] = 1/temp.norm();
    //scale all lambdas
    for(unsigned int face_no=0 ; face_no<6; ++face_no)
      Gamma[face_no] = -Gamma[face_no]/h;
      
    Gamma_mid[0] = Gamma_mid[0]/h;
    Gamma_mid[1] = Gamma_mid[1]/h;
    Gamma_mid[2] = Gamma_mid[2]/h;    

    //calculate volume center point
    for(unsigned int k = 0; k<n_shape_functions; ++k)
      center_pt += supp_pts[k]*fe_data.center_value(6,k);          
  }

  // 2) Set A matrix
  unsigned int size_A=0;
  if(dim==2)
    size_A = 4*fe_degree;
  else if (dim==3)
    size_A = 8+12*(fe_degree-1);

  FullMatrix<double> A(size_A);
  std::vector<double> pre_pre_phi;
  std::vector<Tensor<1,dim> > pre_pre_phi_grad;
  const unsigned int r = fe_degree;

  double alpha_x, beta_x, alpha_y, beta_y, alpha_z, beta_z,eta_x,eta_y,eta_z;
  alpha_x = 0.0; beta_x = 0.0; eta_x = 1.0;
  alpha_y = 0.0; beta_y = 0.0; eta_y = 1.0;
  alpha_z = 0.0; beta_z = 0.0; eta_z = 1.0;

  double sigma_v,eta_v,sigma_h,eta_h;
  sigma_v = 1.0; eta_v = 1.0;
  sigma_h = 1.0; eta_h = 1.0;

  std::vector<double> pre_phi_int;
  std::vector<Tensor<1,dim> > pre_phi_int_grad;

  if(dim==2)
  {
    pre_pre_phi.resize(8);
    pre_pre_phi_grad.resize(8);

    // Tensor<1,dim> gamma_t = Gamma[2] - Gamma[3];
    // gamma_t = gamma_t/gamma_t.norm();
    // double cost = gamma_t*Gamma[0]/Gamma[0].norm();
    // sigma_v = 1.0/std::sqrt(1.0-cost*cost);
    // cost = gamma_t*Gamma[1]/Gamma[1].norm();
    // eta_v = 1.0/std::sqrt(1.0-cost*cost);

    // gamma_t = Gamma[0] - Gamma[1];
    // gamma_t = gamma_t/gamma_t.norm();
    // cost = gamma_t*Gamma[2]/Gamma[2].norm();
    // sigma_h = 1.0/std::sqrt(1.0-cost*cost);
    // cost = gamma_t*Gamma[3]/Gamma[3].norm();
    // eta_h = 1.0/std::sqrt(1.0-cost*cost);

    // std::cout<<"sigma_v:" <<sigma_v
    // <<"   eta_v:"<<eta_v
    // <<"   sigma_h:"<<sigma_h
    // <<"   eta_h:"<<eta_h
    // <<std::endl;

    unsigned int row_no = 0;
    for(unsigned int vertex_no = 0; vertex_no<4; ++vertex_no, ++row_no)
    {
      // set vertex dofs
      Tensor<1, dim> dof_pt;
      dof_pt = supp_pts[vertex_no];
      // set pre_pre_phi values
      pre_pre_phi[0] = (dof_pt - supp_pts[0])*Gamma[0]; //lambda_0
      pre_pre_phi[1] = (dof_pt - supp_pts[1])*Gamma[1]; //lambda_1
      pre_pre_phi[2] = (dof_pt - supp_pts[0])*Gamma[2]; //lambda_2
      pre_pre_phi[3] = (dof_pt - supp_pts[2])*Gamma[3]; //lambda_3
      Assert(pre_pre_phi[0]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[1]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[2]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[3]>=-1e-13, ExcMessage("dof point out of cell"));
      pre_pre_phi[4] = pre_pre_phi[0] - pre_pre_phi[1]; //lambda_V
      pre_pre_phi[5] = pre_pre_phi[2] - pre_pre_phi[3]; //lambda_H
      pre_pre_phi[6] = pre_pre_phi[4] / (sigma_v*pre_pre_phi[0] + eta_v*pre_pre_phi[1]); //R_V
      pre_pre_phi[7] = pre_pre_phi[5] / (sigma_h*pre_pre_phi[2] + eta_h*pre_pre_phi[3]); //R_H
      AssertIsFinite(pre_pre_phi[6]);
      AssertIsFinite(pre_pre_phi[7]);
      // set matrix A
      A[row_no][0] = pre_pre_phi[1]*pre_pre_phi[3]; //lambda_1*lambda_3
      A[row_no][1] = pre_pre_phi[0]*pre_pre_phi[3]; //lambda_0*lambda_3
      A[row_no][2] = pre_pre_phi[1]*pre_pre_phi[2]; //lambda_1*lambda_2
      A[row_no][3] = pre_pre_phi[0]*pre_pre_phi[2]; //lambda_0*lambda_2
    }

    Assert(row_no==4, ExcMessage("row number not correct"));

    for(unsigned int line_no=1; line_no<fe_degree; ++line_no, ++row_no)
    {
      // left line dofs
      Point<dim> dof_pt;
      dof_pt[0] = supp_pts[0][0]+(supp_pts[2][0]-supp_pts[0][0])*line_no/fe_degree;
      dof_pt[1] = supp_pts[0][1]+(supp_pts[2][1]-supp_pts[0][1])*line_no/fe_degree;

      // set pre_pre_phi values
      pre_pre_phi[0] = (dof_pt - supp_pts[0])*Gamma[0]; //lambda_0
      pre_pre_phi[1] = (dof_pt - supp_pts[1])*Gamma[1]; //lambda_1
      pre_pre_phi[2] = (dof_pt - supp_pts[0])*Gamma[2]; //lambda_2
      pre_pre_phi[3] = (dof_pt - supp_pts[2])*Gamma[3]; //lambda_3
      Assert(pre_pre_phi[0]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[1]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[2]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[3]>=-1e-13, ExcMessage("dof point out of cell"));
      pre_pre_phi[4] = pre_pre_phi[0] - pre_pre_phi[1]; //lambda_V
      pre_pre_phi[5] = pre_pre_phi[2] - pre_pre_phi[3]; //lambda_H
      pre_pre_phi[6] = pre_pre_phi[4] / (sigma_v*pre_pre_phi[0] + eta_v*pre_pre_phi[1]); //R_V
      pre_pre_phi[7] = pre_pre_phi[5] / (sigma_h*pre_pre_phi[2] + eta_h*pre_pre_phi[3]); //R_H
      AssertIsFinite(pre_pre_phi[6]);
      AssertIsFinite(pre_pre_phi[7]);

      // set matrix A
      A[row_no][0] = pre_pre_phi[1]*pre_pre_phi[3]; //lambda_1*lambda_3
      A[row_no][1] = pre_pre_phi[0]*pre_pre_phi[3]; //lambda_0*lambda_3
      A[row_no][2] = pre_pre_phi[1]*pre_pre_phi[2]; //lambda_1*lambda_2
      A[row_no][3] = pre_pre_phi[0]*pre_pre_phi[2]; //lambda_0*lambda_2

      for(unsigned int j=0; j< fe_degree-1; ++j)
      {
        A[row_no][4+j] = pre_pre_phi[2]*pre_pre_phi[3]*std::pow(pre_pre_phi[5],j);
        A[row_no][fe_degree+3+j] = 
        pre_pre_phi[2]*pre_pre_phi[3]*pre_pre_phi[4]*std::pow(pre_pre_phi[5],j);
      }
      A[row_no][2*fe_degree+1] = 
        pre_pre_phi[2]*pre_pre_phi[3]*pre_pre_phi[6]*std::pow(pre_pre_phi[5],fe_degree-2);

      if(flag_new)
      {
        Point<dim> dof_pt_hat;
        dof_pt_hat[0] = 0.0; dof_pt_hat[1] = (double)line_no/fe_degree;
        A[row_no][2*fe_degree+1] = 
        this->poly_space.compute_value(2*fe_degree+1, dof_pt_hat)*(-4.0);
      }
    }

    Assert(row_no==fe_degree+3, ExcMessage("row number not correct"));

    for(unsigned int line_no=1; line_no<fe_degree; ++line_no, ++row_no)
    {
      // right line dofs
      Point<dim> dof_pt;
      dof_pt[0] = supp_pts[1][0]+(supp_pts[3][0]-supp_pts[1][0])*line_no/fe_degree;
      dof_pt[1] = supp_pts[1][1]+(supp_pts[3][1]-supp_pts[1][1])*line_no/fe_degree;
      // set pre_pre_phi values
      pre_pre_phi[0] = (dof_pt - supp_pts[0])*Gamma[0]; //lambda_0
      pre_pre_phi[1] = (dof_pt - supp_pts[1])*Gamma[1]; //lambda_1
      pre_pre_phi[2] = (dof_pt - supp_pts[0])*Gamma[2]; //lambda_2
      pre_pre_phi[3] = (dof_pt - supp_pts[2])*Gamma[3]; //lambda_3
      Assert(pre_pre_phi[0]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[1]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[2]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[3]>=-1e-13, ExcMessage("dof point out of cell"));
      pre_pre_phi[4] = pre_pre_phi[0] - pre_pre_phi[1]; //lambda_V
      pre_pre_phi[5] = pre_pre_phi[2] - pre_pre_phi[3]; //lambda_H
      pre_pre_phi[6] = pre_pre_phi[4] / (sigma_v*pre_pre_phi[0] + eta_v*pre_pre_phi[1]); //R_V
      pre_pre_phi[7] = pre_pre_phi[5] / (sigma_h*pre_pre_phi[2] + eta_h*pre_pre_phi[3]); //R_H
      AssertIsFinite(pre_pre_phi[6]);
      AssertIsFinite(pre_pre_phi[7]);

      // set matrix A
      A[row_no][0] = pre_pre_phi[1]*pre_pre_phi[3]; //lambda_1*lambda_3
      A[row_no][1] = pre_pre_phi[0]*pre_pre_phi[3]; //lambda_0*lambda_3
      A[row_no][2] = pre_pre_phi[1]*pre_pre_phi[2]; //lambda_1*lambda_2
      A[row_no][3] = pre_pre_phi[0]*pre_pre_phi[2]; //lambda_0*lambda_2

      for(unsigned int j=0; j< fe_degree-1; ++j)
      {
        A[row_no][4+j] = pre_pre_phi[2]*pre_pre_phi[3]*std::pow(pre_pre_phi[5],j);
        A[row_no][fe_degree+3+j] = 
        pre_pre_phi[2]*pre_pre_phi[3]*pre_pre_phi[4]*std::pow(pre_pre_phi[5],j);
      }
      A[row_no][2*fe_degree+1] = 
        pre_pre_phi[2]*pre_pre_phi[3]*pre_pre_phi[6]*std::pow(pre_pre_phi[5],fe_degree-2);

      if(flag_new)
      {  
        Point<dim> dof_pt_hat;
        dof_pt_hat[0] = 1.0; dof_pt_hat[1] = (double)line_no/fe_degree;
        A[row_no][2*fe_degree+1] = 
        this->poly_space.compute_value(2*fe_degree+1, dof_pt_hat)*(-4.0);
      }
    }

    Assert(row_no==2*fe_degree+2, ExcMessage("row number not correct"));

    for(unsigned int line_no=1; line_no<fe_degree; ++line_no, ++row_no)
    {
      // bottom line dofs
      Point<dim> dof_pt;
      dof_pt[0] = supp_pts[0][0]+(supp_pts[1][0]-supp_pts[0][0])*line_no/fe_degree;
      dof_pt[1] = supp_pts[0][1]+(supp_pts[1][1]-supp_pts[0][1])*line_no/fe_degree;
      // set pre_pre_phi values
      pre_pre_phi[0] = (dof_pt - supp_pts[0])*Gamma[0]; //lambda_0
      pre_pre_phi[1] = (dof_pt - supp_pts[1])*Gamma[1]; //lambda_1
      pre_pre_phi[2] = (dof_pt - supp_pts[0])*Gamma[2]; //lambda_2
      pre_pre_phi[3] = (dof_pt - supp_pts[2])*Gamma[3]; //lambda_3
      Assert(pre_pre_phi[0]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[1]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[2]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[3]>=-1e-13, ExcMessage("dof point out of cell"));
      pre_pre_phi[4] = pre_pre_phi[0] - pre_pre_phi[1]; //lambda_V
      pre_pre_phi[5] = pre_pre_phi[2] - pre_pre_phi[3]; //lambda_H
      pre_pre_phi[6] = pre_pre_phi[4] / (sigma_v*pre_pre_phi[0] + eta_v*pre_pre_phi[1]); //R_V
      pre_pre_phi[7] = pre_pre_phi[5] / (sigma_h*pre_pre_phi[2] + eta_h*pre_pre_phi[3]); //R_H
      AssertIsFinite(pre_pre_phi[6]);
      AssertIsFinite(pre_pre_phi[7]);

      // set matrix A
      A[row_no][0] = pre_pre_phi[1]*pre_pre_phi[3]; //lambda_1*lambda_3
      A[row_no][1] = pre_pre_phi[0]*pre_pre_phi[3]; //lambda_0*lambda_3
      A[row_no][2] = pre_pre_phi[1]*pre_pre_phi[2]; //lambda_1*lambda_2
      A[row_no][3] = pre_pre_phi[0]*pre_pre_phi[2]; //lambda_0*lambda_2

      for(unsigned int j=0; j< fe_degree-1; ++j)
      {
        A[row_no][2*fe_degree+2+j] = pre_pre_phi[0]*pre_pre_phi[1]*std::pow(pre_pre_phi[4],j);
        A[row_no][3*fe_degree+1+j] = 
        pre_pre_phi[0]*pre_pre_phi[1]*pre_pre_phi[5]*std::pow(pre_pre_phi[4],j);
      }
      A[row_no][4*fe_degree-1] = 
        pre_pre_phi[0]*pre_pre_phi[1]*pre_pre_phi[7]*std::pow(pre_pre_phi[4],fe_degree-2);

      if(flag_new)
      {
        Point<dim> dof_pt_hat;
        dof_pt_hat[0] = (double)line_no/fe_degree; dof_pt_hat[1] = 0.0;
        A[row_no][4*fe_degree-1] = 
        this->poly_space.compute_value(4*fe_degree-1, dof_pt_hat)*(-4.0);
      }
    }

    Assert(row_no==3*fe_degree+1, ExcMessage("row number not correct"));

    for(unsigned int line_no=1; line_no<fe_degree; ++line_no, ++row_no)
    {
      // top line dofs
      Point<dim> dof_pt;
      dof_pt[0] = supp_pts[2][0]+(supp_pts[3][0]-supp_pts[2][0])*line_no/fe_degree;
      dof_pt[1] = supp_pts[2][1]+(supp_pts[3][1]-supp_pts[2][1])*line_no/fe_degree;
      // set pre_pre_phi values
      pre_pre_phi[0] = (dof_pt - supp_pts[0])*Gamma[0]; //lambda_0
      pre_pre_phi[1] = (dof_pt - supp_pts[1])*Gamma[1]; //lambda_1
      pre_pre_phi[2] = (dof_pt - supp_pts[0])*Gamma[2]; //lambda_2
      pre_pre_phi[3] = (dof_pt - supp_pts[2])*Gamma[3]; //lambda_3
      Assert(pre_pre_phi[0]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[1]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[2]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[3]>=-1e-13, ExcMessage("dof point out of cell"));
      pre_pre_phi[4] = pre_pre_phi[0] - pre_pre_phi[1]; //lambda_V
      pre_pre_phi[5] = pre_pre_phi[2] - pre_pre_phi[3]; //lambda_H
      pre_pre_phi[6] = pre_pre_phi[4] / (sigma_v*pre_pre_phi[0] + eta_v*pre_pre_phi[1]); //R_V
      pre_pre_phi[7] = pre_pre_phi[5] / (sigma_h*pre_pre_phi[2] + eta_h*pre_pre_phi[3]); //R_H
      AssertIsFinite(pre_pre_phi[6]);
      AssertIsFinite(pre_pre_phi[7]);

      // set matrix A
      A[row_no][0] = pre_pre_phi[1]*pre_pre_phi[3]; //lambda_1*lambda_3
      A[row_no][1] = pre_pre_phi[0]*pre_pre_phi[3]; //lambda_0*lambda_3
      A[row_no][2] = pre_pre_phi[1]*pre_pre_phi[2]; //lambda_1*lambda_2
      A[row_no][3] = pre_pre_phi[0]*pre_pre_phi[2]; //lambda_0*lambda_2

      for(unsigned int j=0; j< fe_degree-1; ++j)
      {
        A[row_no][2*fe_degree+2+j] = pre_pre_phi[0]*pre_pre_phi[1]*std::pow(pre_pre_phi[4],j);
        A[row_no][3*fe_degree+1+j] = 
        pre_pre_phi[0]*pre_pre_phi[1]*pre_pre_phi[5]*std::pow(pre_pre_phi[4],j);
      }
      A[row_no][4*fe_degree-1] = 
        pre_pre_phi[0]*pre_pre_phi[1]*pre_pre_phi[7]*std::pow(pre_pre_phi[4],fe_degree-2);

      if(flag_new)
      {
        Point<dim> dof_pt_hat;
        dof_pt_hat[0] = (double)line_no/fe_degree; dof_pt_hat[1] = 1.0;
        A[row_no][4*fe_degree-1] = 
        this->poly_space.compute_value(4*fe_degree-1, dof_pt_hat)*(-4.0);
      }
    }
  } //end (dim==2)

  if(dim==3)
  {
    pre_pre_phi.resize(12);
    pre_pre_phi_grad.resize(12);

    Assert(r>2,ExcMessage("3D FE_S_NP only works for degree>=3"));

    for(unsigned int row_no=0; row_no<12*r-4; ++row_no)
    {
      // set support point
      Tensor<1,dim> dof_pt;
      if(row_no<8)
      {
        dof_pt = supp_pts[row_no];
      }
      else
      {
        if(row_no>= 8 && row_no<7+r)
          dof_pt = supp_pts[0] + (supp_pts[2]-supp_pts[0])*(row_no-7)/r;
        if(row_no>= 7+r && row_no<6+2*r)
          dof_pt = supp_pts[1] + (supp_pts[3]-supp_pts[1])*(row_no-6-r)/r;
        if(row_no>= 6+2*r && row_no<5+3*r)
          dof_pt = supp_pts[0] + (supp_pts[1]-supp_pts[0])*(row_no-5-2*r)/r;
        if(row_no>= 5+3*r && row_no<4+4*r)
          dof_pt = supp_pts[2] + (supp_pts[3]-supp_pts[2])*(row_no-4-3*r)/r;
        if(row_no>= 4+4*r && row_no<3+5*r)
          dof_pt = supp_pts[4] + (supp_pts[6]-supp_pts[4])*(row_no-3-4*r)/r;
        if(row_no>= 3+5*r && row_no<2+6*r)
          dof_pt = supp_pts[5] + (supp_pts[7]-supp_pts[5])*(row_no-2-5*r)/r;
        if(row_no>=2+6*r && row_no<1+7*r)
          dof_pt = supp_pts[4] + (supp_pts[5]-supp_pts[4])*(row_no-1-6*r)/r;
        if(row_no>=1+7*r && row_no<8*r)
          dof_pt = supp_pts[6] + (supp_pts[7]-supp_pts[6])*(row_no-7*r)/r;
        if(row_no>=8*r && row_no<9*r-1)
          dof_pt = supp_pts[0] + (supp_pts[4]-supp_pts[0])*(row_no+1-8*r)/r;
        if(row_no>=9*r-1 && row_no<10*r-2)
          dof_pt = supp_pts[1] + (supp_pts[5]-supp_pts[1])*(row_no+2-9*r)/r;
        if(row_no>=10*r-2 && row_no<11*r-3)
          dof_pt = supp_pts[2] + (supp_pts[6]-supp_pts[2])*(row_no+3-10*r)/r;
        if(row_no>=11*r-3 && row_no<12*r-4)
          dof_pt = supp_pts[3] + (supp_pts[7]-supp_pts[3])*(row_no+4-11*r)/r;
      }
      // set pre_pre_phi values
      pre_pre_phi[0] = (dof_pt - supp_pts[0])*Gamma[0]; //lambda_0
      pre_pre_phi[1] = (dof_pt - supp_pts[1])*Gamma[1]; //lambda_1
      pre_pre_phi[2] = (dof_pt - supp_pts[0])*Gamma[2]; //lambda_2
      pre_pre_phi[3] = (dof_pt - supp_pts[2])*Gamma[3]; //lambda_3
      pre_pre_phi[4] = (dof_pt - supp_pts[0])*Gamma[4]; //lambda_4
      pre_pre_phi[5] = (dof_pt - supp_pts[4])*Gamma[5]; //lambda_5
      Assert(pre_pre_phi[0]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[1]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[2]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[3]>=-1e-13, ExcMessage("dof point out of cell")); 
      Assert(pre_pre_phi[4]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[5]>=-1e-13, ExcMessage("dof point out of cell"));          

      // set matrix A entries
      A[row_no][0] = pre_pre_phi[1]*pre_pre_phi[3]*pre_pre_phi[5];
      A[row_no][1] = pre_pre_phi[0]*pre_pre_phi[3]*pre_pre_phi[5];
      A[row_no][2] = pre_pre_phi[1]*pre_pre_phi[2]*pre_pre_phi[5];
      A[row_no][3] = pre_pre_phi[0]*pre_pre_phi[2]*pre_pre_phi[5];
      A[row_no][4] = pre_pre_phi[1]*pre_pre_phi[3]*pre_pre_phi[4];
      A[row_no][5] = pre_pre_phi[0]*pre_pre_phi[3]*pre_pre_phi[4];
      A[row_no][6] = pre_pre_phi[1]*pre_pre_phi[2]*pre_pre_phi[4];
      A[row_no][7] = pre_pre_phi[0]*pre_pre_phi[2]*pre_pre_phi[4];

      if((row_no>=8 && row_no<6+2*r) || (row_no>=4+4*r && r<2+6*r))
      {
        // y dir
        // R_x
        pre_pre_phi[9] = (B[0][4]*pre_pre_phi[0] - B[1][4]*pre_pre_phi[1])
        /(B[0][4]*pre_pre_phi[0] + B[1][4]*pre_pre_phi[1]);
        // R_z
        pre_pre_phi[11] = (B[4][0]*pre_pre_phi[4] - B[5][0]*pre_pre_phi[5])
        /(B[4][0]*pre_pre_phi[4] + B[5][0]*pre_pre_phi[5]); 

        AssertIsFinite(pre_pre_phi[9]);
        AssertIsFinite(pre_pre_phi[11]);        

        pre_pre_phi[7] = (dof_pt - center_pt)*Gamma_mid[1];
        pre_pre_phi[6] = B[0][4]*pre_pre_phi[0] - B[1][4]*pre_pre_phi[1];
        pre_pre_phi[8] = B[4][0]*pre_pre_phi[4] - B[5][0]*pre_pre_phi[5];


        double temp = pre_pre_phi[2]*pre_pre_phi[3];
        A[row_no][8] = temp;
        A[row_no][7+r] = temp*pre_pre_phi[6];
        A[row_no][4+4*r] = temp*pre_pre_phi[8];
        A[row_no][3+5*r] = temp*(alpha_y*pre_pre_phi[8]*pre_pre_phi[9]
          +beta_y*pre_pre_phi[6]*pre_pre_phi[11]
          +eta_y*pre_pre_phi[9]*pre_pre_phi[11]);

        for(unsigned int j=1; j<r-2; ++j)
        {
          A[row_no][8+j] = temp*std::pow(pre_pre_phi[7],j);
          A[row_no][7+r+j] = temp*pre_pre_phi[6]*std::pow(pre_pre_phi[7],j);
          A[row_no][4+4*r+j] = temp*pre_pre_phi[8]*std::pow(pre_pre_phi[7],j);
          A[row_no][3+5*r+j] = temp*pre_pre_phi[9]*pre_pre_phi[11]*std::pow(pre_pre_phi[7],j);          
        }

        A[row_no][8+r-2] = temp*std::pow(pre_pre_phi[7],r-2);
        A[row_no][7+r+r-2] = temp*pre_pre_phi[9]*std::pow(pre_pre_phi[7],r-2);
        A[row_no][4+4*r+r-2] = temp*pre_pre_phi[11]*std::pow(pre_pre_phi[7],r-2);
        A[row_no][3+5*r+r-2] = temp*pre_pre_phi[11]*pre_pre_phi[9]*std::pow(pre_pre_phi[7],r-2);
      }

      if((row_no>=6+2*r && row_no<4+4*r) || (row_no>=2+6*r && r<8*r))
      {
        // x dir
        // R_y
        pre_pre_phi[10] = (B[2][4]*pre_pre_phi[2] - B[3][4]*pre_pre_phi[3])
        /(B[2][4]*pre_pre_phi[2] + B[3][4]*pre_pre_phi[3]);
        // R_z
        pre_pre_phi[11] = (B[4][2]*pre_pre_phi[4] - B[5][2]*pre_pre_phi[5])
        /(B[4][2]*pre_pre_phi[4] + B[5][2]*pre_pre_phi[5]); 
        AssertIsFinite(pre_pre_phi[10]);
        AssertIsFinite(pre_pre_phi[11]);      

        pre_pre_phi[6] = (dof_pt - center_pt)*Gamma_mid[0];
        pre_pre_phi[7] = B[2][4]*pre_pre_phi[2] - B[3][4]*pre_pre_phi[3];
        pre_pre_phi[8] = B[4][2]*pre_pre_phi[4] - B[5][2]*pre_pre_phi[5];

        double temp = pre_pre_phi[0]*pre_pre_phi[1];
        A[row_no][6+2*r] = temp;
        A[row_no][5+3*r] = temp*pre_pre_phi[7];
        A[row_no][2+6*r] = temp*pre_pre_phi[8];
        A[row_no][1+7*r] = temp*(alpha_x*pre_pre_phi[7]*pre_pre_phi[11]
          +beta_x*pre_pre_phi[8]*pre_pre_phi[10]
          +eta_x*pre_pre_phi[10]*pre_pre_phi[11]);

        for(unsigned int j=1; j<r-2; ++j)
        {
          A[row_no][6+2*r+j] = temp*std::pow(pre_pre_phi[6],j);
          A[row_no][5+3*r+j] = temp*pre_pre_phi[7]*std::pow(pre_pre_phi[6],j);
          A[row_no][2+6*r+j] = temp*pre_pre_phi[8]*std::pow(pre_pre_phi[6],j);
          A[row_no][1+7*r+j] = temp*pre_pre_phi[10]*pre_pre_phi[11]*std::pow(pre_pre_phi[6],j);          
        }

        A[row_no][6+2*r+r-2] = temp*std::pow(pre_pre_phi[6],r-2);
        A[row_no][5+3*r+r-2] = temp*pre_pre_phi[10]*std::pow(pre_pre_phi[6],r-2);
        A[row_no][2+6*r+r-2] = temp*pre_pre_phi[11]*std::pow(pre_pre_phi[6],r-2);
        A[row_no][1+7*r+r-2] = temp*pre_pre_phi[10]*pre_pre_phi[11]*std::pow(pre_pre_phi[6],r-2);
      }

      if((row_no>=8*r && row_no<12*r-4))
      {
        // z dir

        // R_x
        pre_pre_phi[9] = (B[0][2]*pre_pre_phi[0] - B[1][2]*pre_pre_phi[1])
        /(B[0][2]*pre_pre_phi[0] + B[1][2]*pre_pre_phi[1]);
        // R_y
        pre_pre_phi[10] = (B[2][0]*pre_pre_phi[2] - B[3][0]*pre_pre_phi[3])
        /(B[2][0]*pre_pre_phi[2] + B[3][0]*pre_pre_phi[3]); 
        AssertIsFinite(pre_pre_phi[9]);
        AssertIsFinite(pre_pre_phi[10]);      

        pre_pre_phi[8] = (dof_pt - center_pt)*Gamma_mid[2];
        pre_pre_phi[6] = B[0][2]*pre_pre_phi[0] - B[1][2]*pre_pre_phi[1];
        pre_pre_phi[7] = B[2][0]*pre_pre_phi[2] - B[3][0]*pre_pre_phi[3];

        double temp = pre_pre_phi[4]*pre_pre_phi[5];
        A[row_no][8*r] = temp;
        A[row_no][9*r-1] = temp*pre_pre_phi[6];
        A[row_no][10*r-2] = temp*pre_pre_phi[7];
        A[row_no][11*r-3] = temp*(alpha_z*pre_pre_phi[7]*pre_pre_phi[9]
          + beta_z*pre_pre_phi[6]*pre_pre_phi[10]
          + eta_z*pre_pre_phi[9]*pre_pre_phi[10]);


        for(unsigned int j=1; j<r-2; ++j)
        {
          A[row_no][8*r+j] = temp*std::pow(pre_pre_phi[8],j);
          A[row_no][9*r-1+j] = temp*pre_pre_phi[6]*std::pow(pre_pre_phi[8],j);
          A[row_no][10*r-2+j] = temp*pre_pre_phi[7]*std::pow(pre_pre_phi[8],j);
          A[row_no][11*r-3+j] = temp*pre_pre_phi[10]*pre_pre_phi[9]*std::pow(pre_pre_phi[8],j);
        }

        A[row_no][8*r+r-2] = temp*std::pow(pre_pre_phi[8],r-2);
        A[row_no][9*r-1+r-2] = temp*pre_pre_phi[9]*std::pow(pre_pre_phi[8],r-2);
        A[row_no][10*r-2+r-2] = temp*pre_pre_phi[10]*std::pow(pre_pre_phi[8],r-2);
        A[row_no][11*r-3+r-2] = temp*pre_pre_phi[9]*pre_pre_phi[10]*std::pow(pre_pre_phi[8],r-2);        
      }  
    }
  } //end (dim==3)

  // 3) invert A
  { // note for square matrix if AB=I, then BA=I
    // A.print_formatted(std::cout);
    LAPACKFullMatrix<double> ll_inverse(A.m(), A.n());
    ll_inverse = A;
    ll_inverse.invert();
    A = ll_inverse;
    // Assert(false,ExcInternalError());
  }

  // 4) set values and grads for each quadrature points

  std::vector<std::vector<Tensor<1,dim> > > temp_gradients;

  if(dim==2 && flag_new)
  {
    temp_gradients.resize(2,
                        std::vector<Tensor<1,dim> > (n_q_points));

    mapping.transform(fe_data.shape_gradients[2*fe_degree+1], 
                      temp_gradients[0],
                      mapping_data, mapping_covariant);

    mapping.transform(fe_data.shape_gradients[4*fe_degree-1], 
                      temp_gradients[1],
                      mapping_data, mapping_covariant);    
  }

  for(unsigned int k=0; k<n_q_points; ++k)
  {
    Point<dim> quad_pt = data.quadrature_points[k];
    // Point<dim> quad_pt_hat = quadrature.point(k);

    Vector<double> pre_phi;
    Vector<double> pre_phi_dx;
    Vector<double> pre_phi_dy;
    Vector<double> pre_phi_dz;

    if(dim==2)
    {
      // set pre_pre_phi values
      pre_pre_phi[0] = (quad_pt - supp_pts[0])*Gamma[0]; //lambda_0
      pre_pre_phi[1] = (quad_pt - supp_pts[1])*Gamma[1]; //lambda_1
      pre_pre_phi[2] = (quad_pt - supp_pts[0])*Gamma[2]; //lambda_2
      pre_pre_phi[3] = (quad_pt - supp_pts[2])*Gamma[3]; //lambda_3
      Assert(pre_pre_phi[0]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[1]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[2]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[3]>=-1e-13, ExcMessage("dof point out of cell"));
      pre_pre_phi[4] = pre_pre_phi[0] - pre_pre_phi[1]; //lambda_V
      pre_pre_phi[5] = pre_pre_phi[2] - pre_pre_phi[3]; //lambda_H
      pre_pre_phi[6] = pre_pre_phi[4] / (sigma_v*pre_pre_phi[0] + eta_v*pre_pre_phi[1]); //R_V
      pre_pre_phi[7] = pre_pre_phi[5] / (sigma_h*pre_pre_phi[2] + eta_h*pre_pre_phi[3]); //R_H
      AssertIsFinite(pre_pre_phi[6]);
      AssertIsFinite(pre_pre_phi[7]);

      pre_pre_phi_grad[0] = Gamma[0];
      pre_pre_phi_grad[1] = Gamma[1];
      pre_pre_phi_grad[2] = Gamma[2];
      pre_pre_phi_grad[3] = Gamma[3];
      pre_pre_phi_grad[4] = pre_pre_phi_grad[0] - pre_pre_phi_grad[1];
      pre_pre_phi_grad[5] = pre_pre_phi_grad[2] - pre_pre_phi_grad[3];
      pre_pre_phi_grad[6] = 
      (pre_pre_phi_grad[4] - pre_pre_phi[6]*(sigma_v*pre_pre_phi_grad[0]+eta_v*pre_pre_phi_grad[1]))/
      (sigma_v*pre_pre_phi[0] + eta_v*pre_pre_phi[1]);
      pre_pre_phi_grad[7] = 
      (pre_pre_phi_grad[5] - pre_pre_phi[7]*(sigma_h*pre_pre_phi_grad[2]+eta_h*pre_pre_phi_grad[3]))/
      (sigma_h*pre_pre_phi[2] + eta_h*pre_pre_phi[3]);


      // for(unsigned int ii=0; ii<8; ++ii)
      //   std::cout<<pre_pre_phi_grad[ii]<<std::endl;

      pre_phi.reinit(4*fe_degree);
      pre_phi_dx.reinit(4*fe_degree);
      pre_phi_dy.reinit(4*fe_degree);

      pre_phi[0] = pre_pre_phi[1]*pre_pre_phi[3]; //lambda_1*lambda_3
      pre_phi[1] = pre_pre_phi[0]*pre_pre_phi[3]; //lambda_0*lambda_3
      pre_phi[2] = pre_pre_phi[1]*pre_pre_phi[2]; //lambda_1*lambda_2
      pre_phi[3] = pre_pre_phi[0]*pre_pre_phi[2]; //lambda_0*lambda_2     

      pre_phi_dx[0] = pre_pre_phi_grad[1][0]*pre_pre_phi[3]
      + pre_pre_phi[1]*pre_pre_phi_grad[3][0];
      pre_phi_dx[1] = pre_pre_phi_grad[0][0]*pre_pre_phi[3]
      + pre_pre_phi[0]*pre_pre_phi_grad[3][0];
      pre_phi_dx[2] = pre_pre_phi_grad[1][0]*pre_pre_phi[2]
      + pre_pre_phi[1]*pre_pre_phi_grad[2][0];
      pre_phi_dx[3] = pre_pre_phi_grad[0][0]*pre_pre_phi[2]
      + pre_pre_phi[0]*pre_pre_phi_grad[2][0];

      pre_phi_dy[0] = pre_pre_phi_grad[1][1]*pre_pre_phi[3]
      + pre_pre_phi[1]*pre_pre_phi_grad[3][1];
      pre_phi_dy[1] = pre_pre_phi_grad[0][1]*pre_pre_phi[3]
      + pre_pre_phi[0]*pre_pre_phi_grad[3][1];
      pre_phi_dy[2] = pre_pre_phi_grad[1][1]*pre_pre_phi[2]
      + pre_pre_phi[1]*pre_pre_phi_grad[2][1];
      pre_phi_dy[3] = pre_pre_phi_grad[0][1]*pre_pre_phi[2]
      + pre_pre_phi[0]*pre_pre_phi_grad[2][1];

      for(unsigned int j=0; j<fe_degree-1;++j)
      {
        // -----------------------------------------------------------------
        pre_phi[4+j] = pre_pre_phi[2]*pre_pre_phi[3]*std::pow(pre_pre_phi[5],j);

        pre_phi[fe_degree+3+j] = 
        pre_pre_phi[2]*pre_pre_phi[3]*((j==(fe_degree-2))?pre_pre_phi[6]:pre_pre_phi[4])
        *std::pow(pre_pre_phi[5],j);

        pre_phi[2*fe_degree+2+j] = pre_pre_phi[0]*pre_pre_phi[1]*std::pow(pre_pre_phi[4],j);

        pre_phi[3*fe_degree+1+j] = 
        pre_pre_phi[0]*pre_pre_phi[1]*((j==(fe_degree-2))?pre_pre_phi[7]:pre_pre_phi[5])
        *std::pow(pre_pre_phi[4],j);
        // -----------------------------------------------------------------
        pre_phi_dx[4+j] = 
        pre_pre_phi_grad[2][0]*pre_pre_phi[3]*std::pow(pre_pre_phi[5],j)
        + pre_pre_phi[2]*pre_pre_phi_grad[3][0]*std::pow(pre_pre_phi[5],j)
        + ((j==0)?0.:j*pre_pre_phi[2]*pre_pre_phi[3]*std::pow(pre_pre_phi[5],j-1)*pre_pre_phi_grad[5][0]);

        pre_phi_dx[fe_degree+3+j] = 
        pre_phi_dx[4+j]*((j==(fe_degree-2))?pre_pre_phi[6]:pre_pre_phi[4])
        + pre_phi[4+j]*((j==(fe_degree-2))?pre_pre_phi_grad[6][0]:pre_pre_phi_grad[4][0]);

        pre_phi_dx[2*fe_degree+2+j] = 
        pre_pre_phi_grad[0][0]*pre_pre_phi[1]*std::pow(pre_pre_phi[4],j)
        + pre_pre_phi[0]*pre_pre_phi_grad[1][0]*std::pow(pre_pre_phi[4],j)
        + ((j==0)?0.:j*pre_pre_phi[0]*pre_pre_phi[1]*std::pow(pre_pre_phi[4],j-1)*pre_pre_phi_grad[4][0]);

        pre_phi_dx[3*fe_degree+1+j] = 
        pre_phi_dx[2*fe_degree+2+j]*((j==(fe_degree-2))?pre_pre_phi[7]:pre_pre_phi[5])
        + pre_phi[2*fe_degree+2+j]*((j==(fe_degree-2))?pre_pre_phi_grad[7][0]:pre_pre_phi_grad[5][0]);
        // -------------------------------------------------------------------
        pre_phi_dy[4+j] = 
        pre_pre_phi_grad[2][1]*pre_pre_phi[3]*std::pow(pre_pre_phi[5],j)
        + pre_pre_phi[2]*pre_pre_phi_grad[3][1]*std::pow(pre_pre_phi[5],j)
        + ((j==0)?0.:j*pre_pre_phi[2]*pre_pre_phi[3]*std::pow(pre_pre_phi[5],j-1)*pre_pre_phi_grad[5][1]);

        pre_phi_dy[fe_degree+3+j] = 
        pre_phi_dy[4+j]*((j==(fe_degree-2))?pre_pre_phi[6]:pre_pre_phi[4])
        + pre_phi[4+j]*((j==(fe_degree-2))?pre_pre_phi_grad[6][1]:pre_pre_phi_grad[4][1]);

        pre_phi_dy[2*fe_degree+2+j] = 
        pre_pre_phi_grad[0][1]*pre_pre_phi[1]*std::pow(pre_pre_phi[4],j)
        + pre_pre_phi[0]*pre_pre_phi_grad[1][1]*std::pow(pre_pre_phi[4],j)
        + ((j==0)?0.:j*pre_pre_phi[0]*pre_pre_phi[1]*std::pow(pre_pre_phi[4],j-1)*pre_pre_phi_grad[4][1]);

        pre_phi_dy[3*fe_degree+1+j] = 
        pre_phi_dy[2*fe_degree+2+j]*((j==(fe_degree-2))?pre_pre_phi[7]:pre_pre_phi[5])
        + pre_phi[2*fe_degree+2+j]*((j==(fe_degree-2))?pre_pre_phi_grad[7][1]:pre_pre_phi_grad[5][1]);
      }

      // std::cout<<"*** pre_phi:"<<std::endl;
      // std::cout<<pre_phi<<std::endl;

      // std::cout<<"*** pre_phi_dx:"<<std::endl;
      // std::cout<<pre_phi_dx<<std::endl;

      // std::cout<<"*** pre_phi_dy:"<<std::endl;
      // std::cout<<pre_phi_dy<<std::endl;

      if(flag_new)
      {
        pre_phi[2*fe_degree+1] = fe_data.shape_values[2*fe_degree+1][k]*(-4.0);
        pre_phi[4*fe_degree-1] = fe_data.shape_values[4*fe_degree-1][k]*(-4.0);

        pre_phi_dx[2*fe_degree+1] = temp_gradients[0][k][0]*(-4.0);
        pre_phi_dy[2*fe_degree+1] = temp_gradients[0][k][1]*(-4.0);

        pre_phi_dx[4*fe_degree-1] = temp_gradients[1][k][0]*(-4.0);
        pre_phi_dy[4*fe_degree-1] = temp_gradients[1][k][1]*(-4.0);
      }
    } //end dim==2

    if(dim==3)
    {
      // set pre_pre_phi values
      pre_pre_phi[0] = (quad_pt - supp_pts[0])*Gamma[0]; //lambda_0
      pre_pre_phi[1] = (quad_pt - supp_pts[1])*Gamma[1]; //lambda_1
      pre_pre_phi[2] = (quad_pt - supp_pts[0])*Gamma[2]; //lambda_2
      pre_pre_phi[3] = (quad_pt - supp_pts[2])*Gamma[3]; //lambda_3
      pre_pre_phi[4] = (quad_pt - supp_pts[0])*Gamma[4]; //lambda_4
      pre_pre_phi[5] = (quad_pt - supp_pts[4])*Gamma[5]; //lambda_5
      Assert(pre_pre_phi[0]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[1]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[2]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[3]>=-1e-13, ExcMessage("dof point out of cell")); 
      Assert(pre_pre_phi[4]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[5]>=-1e-13, ExcMessage("dof point out of cell"));          

      // set pre_pre_phi_grad values
      pre_pre_phi_grad[0] = Gamma[0];
      pre_pre_phi_grad[1] = Gamma[1];
      pre_pre_phi_grad[2] = Gamma[2];
      pre_pre_phi_grad[3] = Gamma[3];
      pre_pre_phi_grad[4] = Gamma[4];
      pre_pre_phi_grad[5] = Gamma[5];

      pre_phi.reinit(12*r-4);
      pre_phi_dx.reinit(12*r-4);
      pre_phi_dy.reinit(12*r-4);
      pre_phi_dz.reinit(12*r-4);

      if(r>3)
      {
        pre_phi_int.resize(6);
        pre_phi_int_grad.resize(6);
      }

      pre_phi[0] = pre_pre_phi[1]*pre_pre_phi[3]*pre_pre_phi[5];
      pre_phi[1] = pre_pre_phi[0]*pre_pre_phi[3]*pre_pre_phi[5];
      pre_phi[2] = pre_pre_phi[1]*pre_pre_phi[2]*pre_pre_phi[5];
      pre_phi[3] = pre_pre_phi[0]*pre_pre_phi[2]*pre_pre_phi[5];
      pre_phi[4] = pre_pre_phi[1]*pre_pre_phi[3]*pre_pre_phi[4];
      pre_phi[5] = pre_pre_phi[0]*pre_pre_phi[3]*pre_pre_phi[4];
      pre_phi[6] = pre_pre_phi[1]*pre_pre_phi[2]*pre_pre_phi[4];
      pre_phi[7] = pre_pre_phi[0]*pre_pre_phi[2]*pre_pre_phi[4];      

      Tensor<1,dim> ttemp;

      ttemp = pre_pre_phi_grad[1]*pre_pre_phi[3]*pre_pre_phi[5]
      + pre_pre_phi[1]*pre_pre_phi_grad[3]*pre_pre_phi[5]
      + pre_pre_phi[1]*pre_pre_phi[3]*pre_pre_phi_grad[5];
      pre_phi_dx[0] = ttemp[0]; pre_phi_dy[0] = ttemp[1]; pre_phi_dz[0] = ttemp[2];

      ttemp = pre_pre_phi_grad[0]*pre_pre_phi[3]*pre_pre_phi[5]
      + pre_pre_phi[0]*pre_pre_phi_grad[3]*pre_pre_phi[5]
      + pre_pre_phi[0]*pre_pre_phi[3]*pre_pre_phi_grad[5];
      pre_phi_dx[1] = ttemp[0]; pre_phi_dy[1] = ttemp[1]; pre_phi_dz[1] = ttemp[2];

      ttemp = pre_pre_phi_grad[1]*pre_pre_phi[2]*pre_pre_phi[5]
      + pre_pre_phi[1]*pre_pre_phi_grad[2]*pre_pre_phi[5]
      + pre_pre_phi[1]*pre_pre_phi[2]*pre_pre_phi_grad[5];
      pre_phi_dx[2] = ttemp[0]; pre_phi_dy[2] = ttemp[1]; pre_phi_dz[2] = ttemp[2];

      ttemp = pre_pre_phi_grad[0]*pre_pre_phi[2]*pre_pre_phi[5]
      + pre_pre_phi[0]*pre_pre_phi_grad[2]*pre_pre_phi[5]
      + pre_pre_phi[0]*pre_pre_phi[2]*pre_pre_phi_grad[5];
      pre_phi_dx[3] = ttemp[0]; pre_phi_dy[3] = ttemp[1]; pre_phi_dz[3] = ttemp[2];

      ttemp = pre_pre_phi_grad[1]*pre_pre_phi[3]*pre_pre_phi[4]
      + pre_pre_phi[1]*pre_pre_phi_grad[3]*pre_pre_phi[4]
      + pre_pre_phi[1]*pre_pre_phi[3]*pre_pre_phi_grad[4];
      pre_phi_dx[4] = ttemp[0]; pre_phi_dy[4] = ttemp[1]; pre_phi_dz[4] = ttemp[2];      

      ttemp = pre_pre_phi_grad[0]*pre_pre_phi[3]*pre_pre_phi[4]
      + pre_pre_phi[0]*pre_pre_phi_grad[3]*pre_pre_phi[4]
      + pre_pre_phi[0]*pre_pre_phi[3]*pre_pre_phi_grad[4];
      pre_phi_dx[5] = ttemp[0]; pre_phi_dy[5] = ttemp[1]; pre_phi_dz[5] = ttemp[2];    

      ttemp = pre_pre_phi_grad[1]*pre_pre_phi[2]*pre_pre_phi[4]
      + pre_pre_phi[1]*pre_pre_phi_grad[2]*pre_pre_phi[4]
      + pre_pre_phi[1]*pre_pre_phi[2]*pre_pre_phi_grad[4];
      pre_phi_dx[6] = ttemp[0]; pre_phi_dy[6] = ttemp[1]; pre_phi_dz[6] = ttemp[2];    

      ttemp = pre_pre_phi_grad[0]*pre_pre_phi[2]*pre_pre_phi[4]
      + pre_pre_phi[0]*pre_pre_phi_grad[2]*pre_pre_phi[4]
      + pre_pre_phi[0]*pre_pre_phi[2]*pre_pre_phi_grad[4];   
      pre_phi_dx[7] = ttemp[0]; pre_phi_dy[7] = ttemp[1]; pre_phi_dz[7] = ttemp[2];   

 
      // y dir

      Tensor<1,dim> ytemp;
      double yvalue;

      pre_pre_phi[7] = (quad_pt - center_pt)*Gamma_mid[1];
      pre_pre_phi[6] = B[0][4]*pre_pre_phi[0] - B[1][4]*pre_pre_phi[1];
      pre_pre_phi[8] = B[4][0]*pre_pre_phi[4] - B[5][0]*pre_pre_phi[5];

      pre_pre_phi_grad[7] = Gamma_mid[1];
      pre_pre_phi_grad[6] = B[0][4]*pre_pre_phi_grad[0] - B[1][4]*pre_pre_phi_grad[1];
      pre_pre_phi_grad[8] = B[4][0]*pre_pre_phi_grad[4] - B[5][0]*pre_pre_phi_grad[5];

      // R_x
      pre_pre_phi[9] = (B[0][4]*pre_pre_phi[0] - B[1][4]*pre_pre_phi[1])
      /(B[0][4]*pre_pre_phi[0] + B[1][4]*pre_pre_phi[1]);
      // R_z
      pre_pre_phi[11] = (B[4][0]*pre_pre_phi[4] - B[5][0]*pre_pre_phi[5])
      /(B[4][0]*pre_pre_phi[4] + B[5][0]*pre_pre_phi[5]); 
      AssertIsFinite(pre_pre_phi[9]);
      AssertIsFinite(pre_pre_phi[11]);        

      pre_pre_phi_grad[9] = 
      (pre_pre_phi_grad[6] - 
        pre_pre_phi[9]*(B[0][4]*pre_pre_phi_grad[0] + B[1][4]*pre_pre_phi_grad[1])
        )/(B[0][4]*pre_pre_phi[0] + B[1][4]*pre_pre_phi[1]);
     
      pre_pre_phi_grad[11] =
      (pre_pre_phi_grad[8] -
        pre_pre_phi[11]*(B[4][0]*pre_pre_phi_grad[4] + B[5][0]*pre_pre_phi_grad[5])
        )/(B[4][0]*pre_pre_phi[4] + B[5][0]*pre_pre_phi[5]);
      
      double yBubble = pre_pre_phi[2]*pre_pre_phi[3];
      Tensor<1,dim> dyBubble = 
      pre_pre_phi_grad[2]*pre_pre_phi[3] + pre_pre_phi[2]*pre_pre_phi_grad[3];

      pre_phi[8] = yBubble;
      pre_phi[7+r] = yBubble*pre_pre_phi[6];
      pre_phi[4+4*r] = yBubble*pre_pre_phi[8];
      pre_phi[3+5*r] = yBubble*(alpha_y*pre_pre_phi[8]*pre_pre_phi[9]
        + beta_y*pre_pre_phi[6]*pre_pre_phi[11]
        + eta_y*pre_pre_phi[9]*pre_pre_phi[11]);
      

      pre_phi_dx[8] = dyBubble[0]; 
      pre_phi_dy[8] = dyBubble[1]; 
      pre_phi_dz[8] = dyBubble[2];
      pre_phi_dx[7+r] = dyBubble[0]*pre_pre_phi[6] + yBubble * pre_pre_phi_grad[6][0];
      pre_phi_dy[7+r] = dyBubble[1]*pre_pre_phi[6] + yBubble * pre_pre_phi_grad[6][1];
      pre_phi_dz[7+r] = dyBubble[2]*pre_pre_phi[6] + yBubble * pre_pre_phi_grad[6][2];
      pre_phi_dx[4+4*r] = dyBubble[0]*pre_pre_phi[8] + yBubble * pre_pre_phi_grad[8][0];
      pre_phi_dy[4+4*r] = dyBubble[1]*pre_pre_phi[8] + yBubble * pre_pre_phi_grad[8][1];
      pre_phi_dz[4+4*r] = dyBubble[2]*pre_pre_phi[8] + yBubble * pre_pre_phi_grad[8][2];

      yvalue = alpha_y*pre_pre_phi[8]*pre_pre_phi[9]
      +beta_y*pre_pre_phi[6]*pre_pre_phi[11]
      +eta_y*pre_pre_phi[9]*pre_pre_phi[11];

      ytemp = alpha_y*pre_pre_phi_grad[8]*pre_pre_phi[9]
      + alpha_y*pre_pre_phi[8]*pre_pre_phi_grad[9]
      + beta_y*pre_pre_phi_grad[6]*pre_pre_phi[11]
      + beta_y*pre_pre_phi[6]*pre_pre_phi_grad[11]
      + eta_y*pre_pre_phi_grad[9]*pre_pre_phi[11]
      + eta_y*pre_pre_phi[9]*pre_pre_phi_grad[11];

      pre_phi_dx[3+5*r] = dyBubble[0]*yvalue + yBubble*ytemp[0];
      pre_phi_dy[3+5*r] = dyBubble[1]*yvalue + yBubble*ytemp[1];
      pre_phi_dz[3+5*r] = dyBubble[2]*yvalue + yBubble*ytemp[2];

      for(unsigned int j=1; j<r-2; ++j)
      {
        pre_phi[8+j] = yBubble*std::pow(pre_pre_phi[7],j);
        pre_phi[7+r+j] = yBubble*pre_pre_phi[6]*std::pow(pre_pre_phi[7],j);
        pre_phi[4+4*r+j] = yBubble*pre_pre_phi[8]*std::pow(pre_pre_phi[7],j);
        pre_phi[3+5*r+j] = yBubble*pre_pre_phi[11]*pre_pre_phi[9]*std::pow(pre_pre_phi[7],j);     

        ytemp = j*std::pow(pre_pre_phi[7],j-1)*pre_pre_phi_grad[7];
        yvalue = std::pow(pre_pre_phi[7],j);
        pre_phi_dx[8+j] = dyBubble[0]*yvalue+yBubble*ytemp[0];
        pre_phi_dy[8+j] = dyBubble[1]*yvalue+yBubble*ytemp[1];
        pre_phi_dz[8+j] = dyBubble[2]*yvalue+yBubble*ytemp[2];                

        ytemp = pre_pre_phi[6] * j*std::pow(pre_pre_phi[7],j-1)*pre_pre_phi_grad[7]
        + pre_pre_phi_grad[6] * std::pow(pre_pre_phi[7],j);
        yvalue = pre_pre_phi[6]*std::pow(pre_pre_phi[7],j);
        pre_phi_dx[7+r+j] = dyBubble[0]*yvalue+yBubble*ytemp[0];
        pre_phi_dy[7+r+j] = dyBubble[1]*yvalue+yBubble*ytemp[1];
        pre_phi_dz[7+r+j] = dyBubble[2]*yvalue+yBubble*ytemp[2];

        ytemp = pre_pre_phi[8] * j*std::pow(pre_pre_phi[7],j-1)*pre_pre_phi_grad[7]
        + pre_pre_phi_grad[8] * std::pow(pre_pre_phi[7],j);
        yvalue = pre_pre_phi[8]*std::pow(pre_pre_phi[7],j);
        pre_phi_dx[4+4*r+j] = dyBubble[0]*yvalue+yBubble*ytemp[0];
        pre_phi_dy[4+4*r+j] = dyBubble[1]*yvalue+yBubble*ytemp[1];
        pre_phi_dz[4+4*r+j] = dyBubble[2]*yvalue+yBubble*ytemp[2];                

        ytemp = pre_pre_phi[11]*pre_pre_phi[9]*j*std::pow(pre_pre_phi[7],j-1)*pre_pre_phi_grad[7]
        + pre_pre_phi_grad[11]*pre_pre_phi[9]*std::pow(pre_pre_phi[7],j)
        + pre_pre_phi[11]*pre_pre_phi_grad[9]*std::pow(pre_pre_phi[7],j); 
        yvalue = pre_pre_phi[11]*pre_pre_phi[9]*std::pow(pre_pre_phi[7],j);
        pre_phi_dx[3+5*r+j] = dyBubble[0]*yvalue+yBubble*ytemp[0];
        pre_phi_dy[3+5*r+j] = dyBubble[1]*yvalue+yBubble*ytemp[1];
        pre_phi_dz[3+5*r+j] = dyBubble[2]*yvalue+yBubble*ytemp[2];                
      }

      pre_phi[8+r-2] = yBubble*std::pow(pre_pre_phi[7],r-2);
      pre_phi[7+r+r-2] = yBubble*pre_pre_phi[9]*std::pow(pre_pre_phi[7],r-2);
      pre_phi[4+4*r+r-2] = yBubble*pre_pre_phi[11]*std::pow(pre_pre_phi[7],r-2);
      pre_phi[3+5*r+r-2] = yBubble*pre_pre_phi[9]*pre_pre_phi[11]*std::pow(pre_pre_phi[7],r-2);

      ytemp = (r-2)*std::pow(pre_pre_phi[7],r-3)*pre_pre_phi_grad[7];
      yvalue = std::pow(pre_pre_phi[7],r-2);
      pre_phi_dx[8+r-2] = dyBubble[0]*yvalue + yBubble*ytemp[0];
      pre_phi_dy[8+r-2] = dyBubble[1]*yvalue + yBubble*ytemp[1];
      pre_phi_dz[8+r-2] = dyBubble[2]*yvalue + yBubble*ytemp[2];            

      yvalue = pre_pre_phi[9]*std::pow(pre_pre_phi[7],r-2);
      ytemp = pre_pre_phi[9]*(r-2)*std::pow(pre_pre_phi[7],r-3)*pre_pre_phi_grad[7]
      + pre_pre_phi_grad[9]*std::pow(pre_pre_phi[7],r-2);
      pre_phi_dx[7+r+r-2] = dyBubble[0]*yvalue + yBubble*ytemp[0];
      pre_phi_dy[7+r+r-2] = dyBubble[1]*yvalue + yBubble*ytemp[1];
      pre_phi_dz[7+r+r-2] = dyBubble[2]*yvalue + yBubble*ytemp[2];

      yvalue = pre_pre_phi[11]*std::pow(pre_pre_phi[7],r-2);
      ytemp = pre_pre_phi[11]*(r-2)*std::pow(pre_pre_phi[7],r-3)*pre_pre_phi_grad[7]
      + pre_pre_phi_grad[11]*std::pow(pre_pre_phi[7],r-2);
      pre_phi_dx[4+4*r+r-2] = dyBubble[0]*yvalue + yBubble*ytemp[0];
      pre_phi_dy[4+4*r+r-2] = dyBubble[1]*yvalue + yBubble*ytemp[1];
      pre_phi_dz[4+4*r+r-2] = dyBubble[2]*yvalue + yBubble*ytemp[2];      

      yvalue = pre_pre_phi[9]*pre_pre_phi[11]*std::pow(pre_pre_phi[7],r-2);
      ytemp = pre_pre_phi[9]*pre_pre_phi[11]*(r-2)*std::pow(pre_pre_phi[7],r-3)*pre_pre_phi_grad[7]
      + pre_pre_phi_grad[9]*pre_pre_phi[11]*std::pow(pre_pre_phi[7],r-2)
      + pre_pre_phi[9]*pre_pre_phi_grad[11]*std::pow(pre_pre_phi[7],r-2);
      pre_phi_dx[3+5*r+r-2] = dyBubble[0]*yvalue + yBubble*ytemp[0];
      pre_phi_dy[3+5*r+r-2] = dyBubble[1]*yvalue + yBubble*ytemp[1];
      pre_phi_dz[3+5*r+r-2] = dyBubble[2]*yvalue + yBubble*ytemp[2];

      if(r>3)
      {
        pre_phi_int[0] = (1. - pre_pre_phi[9])
        * B[2][0]*pre_pre_phi[2]
        * B[3][0]*pre_pre_phi[3]
        * B[4][0]*pre_pre_phi[4]
        * B[5][0]*pre_pre_phi[5];

        pre_phi_int[1] = (1. + pre_pre_phi[9])
        * B[2][1]*pre_pre_phi[2]
        * B[3][1]*pre_pre_phi[3]
        * B[4][1]*pre_pre_phi[4]
        * B[5][1]*pre_pre_phi[5];

        pre_phi_int[4] = (1. - pre_pre_phi[11])
        * B[0][4]*pre_pre_phi[0]
        * B[1][4]*pre_pre_phi[1]
        * B[2][4]*pre_pre_phi[2]
        * B[3][4]*pre_pre_phi[3];

        pre_phi_int[5] = (1. + pre_pre_phi[11])
        * B[0][5]*pre_pre_phi[0]
        * B[1][5]*pre_pre_phi[1]
        * B[2][5]*pre_pre_phi[2]
        * B[3][5]*pre_pre_phi[3];

        pre_phi_int_grad[0] = (-pre_pre_phi_grad[9])
        * B[2][0]*pre_pre_phi[2] * B[3][0]*pre_pre_phi[3]
        * B[4][0]*pre_pre_phi[4] * B[5][0]*pre_pre_phi[5]
        + (1. - pre_pre_phi[9])
        * (B[2][0]*pre_pre_phi_grad[2] * B[3][0]*pre_pre_phi[3]+B[2][0]*pre_pre_phi[2] * B[3][0]*pre_pre_phi_grad[3])
        * B[4][0]*pre_pre_phi[4] * B[5][0]*pre_pre_phi[5]
        + (1. - pre_pre_phi[9])
        * B[2][0]*pre_pre_phi[2] * B[3][0]*pre_pre_phi[3]
        * (B[4][0]*pre_pre_phi_grad[4] * B[5][0]*pre_pre_phi[5]+B[4][0]*pre_pre_phi[4] * B[5][0]*pre_pre_phi_grad[5]);

        pre_phi_int_grad[1] = (pre_pre_phi_grad[9])
        * B[2][1]*pre_pre_phi[2] * B[3][1]*pre_pre_phi[3]
        * B[4][1]*pre_pre_phi[4] * B[5][1]*pre_pre_phi[5]
        + (1. + pre_pre_phi[9])
        * (B[2][1]*pre_pre_phi_grad[2] * B[3][1]*pre_pre_phi[3]+B[2][1]*pre_pre_phi[2] * B[3][1]*pre_pre_phi_grad[3])
        * B[4][1]*pre_pre_phi[4] * B[5][1]*pre_pre_phi[5]
        + (1. + pre_pre_phi[9])
        * B[2][1]*pre_pre_phi[2] * B[3][1]*pre_pre_phi[3]
        * (B[4][1]*pre_pre_phi_grad[4] * B[5][1]*pre_pre_phi[5]+B[4][1]*pre_pre_phi[4] * B[5][1]*pre_pre_phi_grad[5]);

        pre_phi_int_grad[4] = (-pre_pre_phi_grad[11])
        * B[0][4]*pre_pre_phi[0] * B[1][4]*pre_pre_phi[1]
        * B[2][4]*pre_pre_phi[2] * B[3][4]*pre_pre_phi[3]
        + (1. - pre_pre_phi[11])
        * (B[0][4]*pre_pre_phi_grad[0] * B[1][4]*pre_pre_phi[1]+B[0][4]*pre_pre_phi[0] * B[1][4]*pre_pre_phi_grad[1])
        * B[2][4]*pre_pre_phi[2] * B[3][4]*pre_pre_phi[3]
        + (1. - pre_pre_phi[11])
        * B[0][4]*pre_pre_phi[0] * B[1][4]*pre_pre_phi[1]
        * (B[2][4]*pre_pre_phi_grad[2] * B[3][4]*pre_pre_phi[3]+B[2][4]*pre_pre_phi[2] * B[3][4]*pre_pre_phi_grad[3]);

        pre_phi_int_grad[5] = (pre_pre_phi_grad[11])
        * B[0][5]*pre_pre_phi[0] * B[1][5]*pre_pre_phi[1]
        * B[2][5]*pre_pre_phi[2] * B[3][5]*pre_pre_phi[3]
        +  (1. + pre_pre_phi[11])
        * (B[0][5]*pre_pre_phi_grad[0] * B[1][5]*pre_pre_phi[1]+B[0][5]*pre_pre_phi[0] * B[1][5]*pre_pre_phi_grad[1])
        * B[2][5]*pre_pre_phi[2] * B[3][5]*pre_pre_phi[3]
        +  (1. + pre_pre_phi[11])
        * B[0][5]*pre_pre_phi[0] * B[1][5]*pre_pre_phi[1]
        * (B[2][5]*pre_pre_phi_grad[2] * B[3][5]*pre_pre_phi[3]+B[2][5]*pre_pre_phi[2] * B[3][5]*pre_pre_phi_grad[3]);
      }
      // x dir

      pre_pre_phi[6] = (quad_pt - center_pt)*Gamma_mid[0];
      pre_pre_phi[7] = B[2][4]*pre_pre_phi[2] - B[3][4]*pre_pre_phi[3];
      pre_pre_phi[8] = B[4][2]*pre_pre_phi[4] - B[5][2]*pre_pre_phi[5];

      pre_pre_phi_grad[6] = Gamma_mid[0];
      pre_pre_phi_grad[7] = B[2][4]*pre_pre_phi_grad[2] - B[3][4]*pre_pre_phi_grad[3];
      pre_pre_phi_grad[8] = B[4][2]*pre_pre_phi_grad[4] - B[5][2]*pre_pre_phi_grad[5];

      // R_y
      pre_pre_phi[10] = (B[2][4]*pre_pre_phi[2] - B[3][4]*pre_pre_phi[3])
      /(B[2][4]*pre_pre_phi[2] + B[3][4]*pre_pre_phi[3]);
      // R_z
      pre_pre_phi[11] = (B[4][2]*pre_pre_phi[4] - B[5][2]*pre_pre_phi[5])
      /(B[4][2]*pre_pre_phi[4] + B[5][2]*pre_pre_phi[5]); 
      AssertIsFinite(pre_pre_phi[10]);
      AssertIsFinite(pre_pre_phi[11]);         
      
      pre_pre_phi_grad[10] = 
      (pre_pre_phi_grad[7] - 
        pre_pre_phi[10]*(B[2][4]*pre_pre_phi_grad[2] + B[3][4]*pre_pre_phi_grad[3])
        )/(B[2][4]*pre_pre_phi[2] + B[3][4]*pre_pre_phi[3]);
   
      pre_pre_phi_grad[11] = 
      (pre_pre_phi_grad[8] -
        pre_pre_phi[11]*(B[4][2]*pre_pre_phi_grad[4] + B[5][2]*pre_pre_phi_grad[5])
        )/(B[4][2]*pre_pre_phi[4] + B[5][2]*pre_pre_phi[5]);
    
      double xBubble = pre_pre_phi[0]*pre_pre_phi[1];
      Tensor<1,dim> dxBubble = 
      pre_pre_phi_grad[0]*pre_pre_phi[1] + pre_pre_phi[0]*pre_pre_phi_grad[1];

      Tensor<1,dim> xtemp;
      double xvalue;

      pre_phi[6+2*r] = xBubble;
      pre_phi[5+3*r] = xBubble*pre_pre_phi[7];
      pre_phi[2+6*r] = xBubble*pre_pre_phi[8];
      pre_phi[1+7*r] = xBubble*(alpha_x*pre_pre_phi[7]*pre_pre_phi[11]
        + beta_x*pre_pre_phi[8]*pre_pre_phi[10]
        + eta_x*pre_pre_phi[10]*pre_pre_phi[11]);


      pre_phi_dx[6+2*r] = dxBubble[0];
      pre_phi_dy[6+2*r] = dxBubble[1];
      pre_phi_dz[6+2*r] = dxBubble[2];
      pre_phi_dx[5+3*r] = dxBubble[0]*pre_pre_phi[7]+xBubble*pre_pre_phi_grad[7][0]; 
      pre_phi_dy[5+3*r] = dxBubble[1]*pre_pre_phi[7]+xBubble*pre_pre_phi_grad[7][1];
      pre_phi_dz[5+3*r] = dxBubble[2]*pre_pre_phi[7]+xBubble*pre_pre_phi_grad[7][2];
      pre_phi_dx[2+6*r] = dxBubble[0]*pre_pre_phi[8]+xBubble*pre_pre_phi_grad[8][0];     
      pre_phi_dy[2+6*r] = dxBubble[1]*pre_pre_phi[8]+xBubble*pre_pre_phi_grad[8][1];     
      pre_phi_dz[2+6*r] = dxBubble[2]*pre_pre_phi[8]+xBubble*pre_pre_phi_grad[8][2];   

      xvalue = alpha_x*pre_pre_phi[7]*pre_pre_phi[11]
      +beta_x*pre_pre_phi[8]*pre_pre_phi[10]
      +eta_x*pre_pre_phi[10]*pre_pre_phi[11];

      xtemp = alpha_x*pre_pre_phi_grad[7]*pre_pre_phi[11]
      + alpha_x*pre_pre_phi[7]*pre_pre_phi_grad[11]
      + beta_x*pre_pre_phi_grad[8]*pre_pre_phi[10]
      + beta_x*pre_pre_phi[8]*pre_pre_phi_grad[10]
      + eta_x*pre_pre_phi_grad[10]*pre_pre_phi[11]
      + eta_x*pre_pre_phi[10]*pre_pre_phi_grad[11];

      pre_phi_dx[1+7*r] = dxBubble[0]*xvalue + xBubble*xtemp[0];
      pre_phi_dy[1+7*r] = dxBubble[1]*xvalue + xBubble*xtemp[1];
      pre_phi_dz[1+7*r] = dxBubble[2]*xvalue + xBubble*xtemp[2];

      for(unsigned int j=1; j<r-2; ++j)
      {
        pre_phi[6+2*r+j] = xBubble*std::pow(pre_pre_phi[6],j);
        pre_phi[5+3*r+j] = xBubble*pre_pre_phi[7]*std::pow(pre_pre_phi[6],j);
        pre_phi[2+6*r+j] = xBubble*pre_pre_phi[8]*std::pow(pre_pre_phi[6],j);
        pre_phi[1+7*r+j] = xBubble*pre_pre_phi[10]*pre_pre_phi[11]*std::pow(pre_pre_phi[6],j);

        xvalue = std::pow(pre_pre_phi[6],j);
        xtemp = j*std::pow(pre_pre_phi[6],j-1)*pre_pre_phi_grad[6];
        pre_phi_dx[6+2*r+j] = dxBubble[0]*xvalue+xBubble*xtemp[0];
        pre_phi_dy[6+2*r+j] = dxBubble[1]*xvalue+xBubble*xtemp[1];
        pre_phi_dz[6+2*r+j] = dxBubble[2]*xvalue+xBubble*xtemp[2];

        xvalue = pre_pre_phi[7]*std::pow(pre_pre_phi[6],j);
        xtemp = pre_pre_phi[7]*j*std::pow(pre_pre_phi[6],j-1)*pre_pre_phi_grad[6]
        + pre_pre_phi_grad[7]*std::pow(pre_pre_phi[6],j);
        pre_phi_dx[5+3*r+j] = dxBubble[0]*xvalue+xBubble*xtemp[0];
        pre_phi_dy[5+3*r+j] = dxBubble[1]*xvalue+xBubble*xtemp[1];
        pre_phi_dz[5+3*r+j] = dxBubble[2]*xvalue+xBubble*xtemp[2];                

        xvalue = pre_pre_phi[8]*std::pow(pre_pre_phi[6],j);
        xtemp = pre_pre_phi[8]*j*std::pow(pre_pre_phi[6],j-1)*pre_pre_phi_grad[6]
        + pre_pre_phi_grad[8]*std::pow(pre_pre_phi[6],j);
        pre_phi_dx[2+6*r+j] = dxBubble[0]*xvalue+xBubble*xtemp[0];
        pre_phi_dy[2+6*r+j] = dxBubble[1]*xvalue+xBubble*xtemp[1];        
        pre_phi_dz[2+6*r+j] = dxBubble[2]*xvalue+xBubble*xtemp[2];

        xvalue = pre_pre_phi[10]*pre_pre_phi[11]*std::pow(pre_pre_phi[6],j);
        xtemp = pre_pre_phi[10]*pre_pre_phi[11]*j*std::pow(pre_pre_phi[6],j-1)*pre_pre_phi_grad[6]
        + pre_pre_phi_grad[10]*pre_pre_phi[11]*std::pow(pre_pre_phi[6],j)
        + pre_pre_phi[10]*pre_pre_phi_grad[11]*std::pow(pre_pre_phi[6],j);
        pre_phi_dx[1+7*r+j] = dxBubble[0]*xvalue+xBubble*xtemp[0];
        pre_phi_dy[1+7*r+j] = dxBubble[1]*xvalue+xBubble*xtemp[1];
        pre_phi_dz[1+7*r+j] = dxBubble[2]*xvalue+xBubble*xtemp[2];
      }

      pre_phi[6+2*r+r-2] = xBubble*std::pow(pre_pre_phi[6],r-2);
      pre_phi[5+3*r+r-2] = xBubble*pre_pre_phi[10]*std::pow(pre_pre_phi[6],r-2);
      pre_phi[2+6*r+r-2] = xBubble*pre_pre_phi[11]*std::pow(pre_pre_phi[6],r-2);
      pre_phi[1+7*r+r-2] = xBubble*pre_pre_phi[10]*pre_pre_phi[11]*std::pow(pre_pre_phi[6],r-2);

      xvalue = std::pow(pre_pre_phi[6],r-2);
      xtemp = (r-2)*std::pow(pre_pre_phi[6],r-3)*pre_pre_phi_grad[6];
      pre_phi_dx[6+2*r+r-2] = dxBubble[0]*xvalue+xBubble*xtemp[0];
      pre_phi_dy[6+2*r+r-2] = dxBubble[1]*xvalue+xBubble*xtemp[1];
      pre_phi_dz[6+2*r+r-2] = dxBubble[2]*xvalue+xBubble*xtemp[2];            

      xvalue = pre_pre_phi[10]*std::pow(pre_pre_phi[6],r-2);
      xtemp = pre_pre_phi[10]*(r-2)*std::pow(pre_pre_phi[6],r-3)*pre_pre_phi_grad[6]
      + pre_pre_phi_grad[10]*std::pow(pre_pre_phi[6],r-2);
      pre_phi_dx[5+3*r+r-2] = dxBubble[0]*xvalue+xBubble*xtemp[0];
      pre_phi_dy[5+3*r+r-2] = dxBubble[1]*xvalue+xBubble*xtemp[1];
      pre_phi_dz[5+3*r+r-2] = dxBubble[2]*xvalue+xBubble*xtemp[2];

      xvalue = pre_pre_phi[11]*std::pow(pre_pre_phi[6],r-2);
      xtemp = pre_pre_phi[11]*(r-2)*std::pow(pre_pre_phi[6],r-3)*pre_pre_phi_grad[6]
      + pre_pre_phi_grad[11]*std::pow(pre_pre_phi[6],r-2);
      pre_phi_dx[2+6*r+r-2] = dxBubble[0]*xvalue+xBubble*xtemp[0];
      pre_phi_dy[2+6*r+r-2] = dxBubble[1]*xvalue+xBubble*xtemp[1];
      pre_phi_dz[2+6*r+r-2] = dxBubble[2]*xvalue+xBubble*xtemp[2];

      xvalue = pre_pre_phi[10]*pre_pre_phi[11]*std::pow(pre_pre_phi[6],r-2);
      xtemp = pre_pre_phi[10]*pre_pre_phi[11]*(r-2)*std::pow(pre_pre_phi[6],r-3)*pre_pre_phi_grad[6]
      + pre_pre_phi_grad[10]*pre_pre_phi[11]*std::pow(pre_pre_phi[6],r-2)
      + pre_pre_phi[10]*pre_pre_phi_grad[11]*std::pow(pre_pre_phi[6],r-2);
      pre_phi_dx[1+7*r+r-2] = dxBubble[0]*xvalue+xBubble*xtemp[0];
      pre_phi_dy[1+7*r+r-2] = dxBubble[1]*xvalue+xBubble*xtemp[1];
      pre_phi_dz[1+7*r+r-2] = dxBubble[2]*xvalue+xBubble*xtemp[2];

      if(r>3)
      {
        pre_phi_int[2] = (1. - pre_pre_phi[10])
        * B[0][2]*pre_pre_phi[0]
        * B[1][2]*pre_pre_phi[1]
        * B[4][2]*pre_pre_phi[4]
        * B[5][2]*pre_pre_phi[5];

        pre_phi_int[3] = (1. + pre_pre_phi[10])
        * B[0][3]*pre_pre_phi[0]
        * B[1][3]*pre_pre_phi[1]
        * B[4][3]*pre_pre_phi[4]
        * B[5][3]*pre_pre_phi[5];

        pre_phi_int_grad[2] = (-pre_pre_phi_grad[10])
        * B[0][2]*pre_pre_phi[0] * B[1][2]*pre_pre_phi[1]
        * B[4][2]*pre_pre_phi[4] * B[5][2]*pre_pre_phi[5]
        + (1. - pre_pre_phi[10])
        * (B[0][2]*pre_pre_phi_grad[0] * B[1][2]*pre_pre_phi[1]+B[0][2]*pre_pre_phi[0] * B[1][2]*pre_pre_phi_grad[1])
        * B[4][2]*pre_pre_phi[4] * B[5][2]*pre_pre_phi[5]
        + (1. - pre_pre_phi[10])
        * B[0][2]*pre_pre_phi[0] * B[1][2]*pre_pre_phi[1]
        * (B[4][2]*pre_pre_phi_grad[4] * B[5][2]*pre_pre_phi[5]+B[4][2]*pre_pre_phi[4] * B[5][2]*pre_pre_phi_grad[5]);

        pre_phi_int_grad[3] = (pre_pre_phi_grad[10])
        * B[0][3]*pre_pre_phi[0] * B[1][3]*pre_pre_phi[1]
        * B[4][3]*pre_pre_phi[4] * B[5][3]*pre_pre_phi[5]
        + (1. + pre_pre_phi[10])
        * (B[0][3]*pre_pre_phi_grad[0] * B[1][3]*pre_pre_phi[1]+B[0][3]*pre_pre_phi[0] * B[1][3]*pre_pre_phi_grad[1])
        * B[4][3]*pre_pre_phi[4] * B[5][3]*pre_pre_phi[5]
        + (1. + pre_pre_phi[10])
        * B[0][3]*pre_pre_phi[0] * B[1][3]*pre_pre_phi[1]
        * (B[4][3]*pre_pre_phi_grad[4] * B[5][3]*pre_pre_phi[5]+B[4][3]*pre_pre_phi[4] * B[5][3]*pre_pre_phi_grad[5]);
      }
      // z dir

      pre_pre_phi[8] = (quad_pt - center_pt)*Gamma_mid[2];
      pre_pre_phi[6] = B[0][2]*pre_pre_phi[0] - B[1][2]*pre_pre_phi[1];
      pre_pre_phi[7] = B[2][0]*pre_pre_phi[2] - B[3][0]*pre_pre_phi[3];

      pre_pre_phi_grad[8] = Gamma_mid[2];
      pre_pre_phi_grad[6] = B[0][2]*pre_pre_phi_grad[0] - B[1][2]*pre_pre_phi_grad[1];
      pre_pre_phi_grad[7] = B[2][0]*pre_pre_phi_grad[2] - B[3][0]*pre_pre_phi_grad[3];

      // R_x
      pre_pre_phi[9] = (B[0][2]*pre_pre_phi[0] - B[1][2]*pre_pre_phi[1])
      /(B[0][2]*pre_pre_phi[0] + B[1][2]*pre_pre_phi[1]);
      // R_y
      pre_pre_phi[10] = (B[2][0]*pre_pre_phi[2] - B[3][0]*pre_pre_phi[3])
      /(B[2][0]*pre_pre_phi[2] + B[3][0]*pre_pre_phi[3]); 
      AssertIsFinite(pre_pre_phi[9]);
      AssertIsFinite(pre_pre_phi[10]);      

      pre_pre_phi_grad[9] = 
      (pre_pre_phi_grad[6] -
        pre_pre_phi[9]*(B[0][2]*pre_pre_phi_grad[0] + B[1][2]*pre_pre_phi_grad[1])
        )/(B[0][2]*pre_pre_phi[0] + B[1][2]*pre_pre_phi[1]);

      pre_pre_phi_grad[10] = 
      (pre_pre_phi_grad[7] -
        pre_pre_phi[10]*(B[2][0]*pre_pre_phi_grad[2] + B[3][0]*pre_pre_phi_grad[3])
        )/(B[2][0]*pre_pre_phi[2] + B[3][0]*pre_pre_phi[3]);

      double zBubble = pre_pre_phi[4]*pre_pre_phi[5];
      Tensor<1,dim> dzBubble = pre_pre_phi_grad[4]*pre_pre_phi[5]
      + pre_pre_phi[4]*pre_pre_phi_grad[5];

      double zvalue;
      Tensor<1,dim> ztemp;

      pre_phi[8*r] = zBubble;
      pre_phi[9*r-1] = zBubble*pre_pre_phi[6];
      pre_phi[10*r-2] = zBubble*pre_pre_phi[7];
      pre_phi[11*r-3] = zBubble*(alpha_z*pre_pre_phi[7]*pre_pre_phi[9]
        + beta_z*pre_pre_phi[6]*pre_pre_phi[10]
        + eta_z*pre_pre_phi[9]*pre_pre_phi[10]);


      pre_phi_dx[8*r] = dzBubble[0];
      pre_phi_dy[8*r] = dzBubble[1];
      pre_phi_dz[8*r] = dzBubble[2];
      pre_phi_dx[9*r-1] = dzBubble[0]*pre_pre_phi[6]+zBubble*pre_pre_phi_grad[6][0];
      pre_phi_dy[9*r-1] = dzBubble[1]*pre_pre_phi[6]+zBubble*pre_pre_phi_grad[6][1];
      pre_phi_dz[9*r-1] = dzBubble[2]*pre_pre_phi[6]+zBubble*pre_pre_phi_grad[6][2];
      pre_phi_dx[10*r-2] = dzBubble[0]*pre_pre_phi[7]+zBubble*pre_pre_phi_grad[7][0];
      pre_phi_dy[10*r-2] = dzBubble[1]*pre_pre_phi[7]+zBubble*pre_pre_phi_grad[7][1];
      pre_phi_dz[10*r-2] = dzBubble[2]*pre_pre_phi[7]+zBubble*pre_pre_phi_grad[7][2];

      zvalue = alpha_z*pre_pre_phi[7]*pre_pre_phi[9]
        + beta_z*pre_pre_phi[6]*pre_pre_phi[10]
        + eta_z*pre_pre_phi[9]*pre_pre_phi[10];

      ztemp = alpha_z*pre_pre_phi_grad[7]*pre_pre_phi[9]
        + alpha_z*pre_pre_phi[7]*pre_pre_phi_grad[9]
        + beta_z*pre_pre_phi_grad[6]*pre_pre_phi[10]
        + beta_z*pre_pre_phi[6]*pre_pre_phi_grad[10]
        + eta_z*pre_pre_phi_grad[9]*pre_pre_phi[10]
        + eta_z*pre_pre_phi[9]*pre_pre_phi_grad[10];

      pre_phi_dx[11*r-3] = dzBubble[0]*zvalue+zBubble*ztemp[0];
      pre_phi_dy[11*r-3] = dzBubble[1]*zvalue+zBubble*ztemp[1];
      pre_phi_dz[11*r-3] = dzBubble[2]*zvalue+zBubble*ztemp[2];

      for(unsigned int j=1; j<r-2; ++j)
      {
        pre_phi[8*r+j] = zBubble*std::pow(pre_pre_phi[8],j);
        pre_phi[9*r-1+j] = zBubble*pre_pre_phi[6]*std::pow(pre_pre_phi[8],j);
        pre_phi[10*r-2+j] = zBubble*pre_pre_phi[7]*std::pow(pre_pre_phi[8],j);
        pre_phi[11*r-3+j] = zBubble*pre_pre_phi[10]*pre_pre_phi[9]*std::pow(pre_pre_phi[8],j);

        zvalue = std::pow(pre_pre_phi[8],j);
        ztemp = j*std::pow(pre_pre_phi[8],j-1)*pre_pre_phi_grad[8];
        pre_phi_dx[8*r+j] = dzBubble[0]*zvalue+zBubble*ztemp[0];
        pre_phi_dy[8*r+j] = dzBubble[1]*zvalue+zBubble*ztemp[1];
        pre_phi_dz[8*r+j] = dzBubble[2]*zvalue+zBubble*ztemp[2];

        zvalue = pre_pre_phi[6]*std::pow(pre_pre_phi[8],j);
        ztemp = pre_pre_phi[6]*j*std::pow(pre_pre_phi[8],j-1)*pre_pre_phi_grad[8]
        + pre_pre_phi_grad[6]*std::pow(pre_pre_phi[8],j);
        pre_phi_dx[9*r-1+j] = dzBubble[0]*zvalue+zBubble*ztemp[0];
        pre_phi_dy[9*r-1+j] = dzBubble[1]*zvalue+zBubble*ztemp[1];
        pre_phi_dz[9*r-1+j] = dzBubble[2]*zvalue+zBubble*ztemp[2];

        zvalue = pre_pre_phi[7]*std::pow(pre_pre_phi[8],j);
        ztemp =  pre_pre_phi[7]*j*std::pow(pre_pre_phi[8],j-1)*pre_pre_phi_grad[8]
        + pre_pre_phi_grad[7]*std::pow(pre_pre_phi[8],j);
        pre_phi_dx[10*r-2+j] = dzBubble[0]*zvalue+zBubble*ztemp[0];
        pre_phi_dy[10*r-2+j] = dzBubble[1]*zvalue+zBubble*ztemp[1];
        pre_phi_dz[10*r-2+j] = dzBubble[2]*zvalue+zBubble*ztemp[2];

        zvalue = pre_pre_phi[10]*pre_pre_phi[9]*std::pow(pre_pre_phi[8],j);
        ztemp = pre_pre_phi[10]*pre_pre_phi[9]*j*std::pow(pre_pre_phi[8],j-1)*pre_pre_phi_grad[8]
        + pre_pre_phi_grad[10]*pre_pre_phi[9]*std::pow(pre_pre_phi[8],j)
        + pre_pre_phi[10]*pre_pre_phi_grad[9]*std::pow(pre_pre_phi[8],j);
        pre_phi_dx[11*r-3+j] = dzBubble[0]*zvalue+zBubble*ztemp[0];
        pre_phi_dy[11*r-3+j] = dzBubble[1]*zvalue+zBubble*ztemp[1];
        pre_phi_dz[11*r-3+j] = dzBubble[2]*zvalue+zBubble*ztemp[2];
      }

      pre_phi[8*r+r-2] = zBubble*std::pow(pre_pre_phi[8],r-2);
      pre_phi[9*r-1+r-2] = zBubble*pre_pre_phi[9]*std::pow(pre_pre_phi[8],r-2);
      pre_phi[10*r-2+r-2] = zBubble*pre_pre_phi[10]*std::pow(pre_pre_phi[8],r-2);
      pre_phi[11*r-3+r-2] = zBubble*pre_pre_phi[9]*pre_pre_phi[10]*std::pow(pre_pre_phi[8],r-2); 

      zvalue = std::pow(pre_pre_phi[8],r-2);
      ztemp = (r-2)*std::pow(pre_pre_phi[8],r-3)*pre_pre_phi_grad[8];
      pre_phi_dx[8*r+r-2] = dzBubble[0]*zvalue+zBubble*ztemp[0];
      pre_phi_dy[8*r+r-2] = dzBubble[1]*zvalue+zBubble*ztemp[1];
      pre_phi_dz[8*r+r-2] = dzBubble[2]*zvalue+zBubble*ztemp[2];

      zvalue = pre_pre_phi[9]*std::pow(pre_pre_phi[8],r-2);
      ztemp = pre_pre_phi[9]*(r-2)*std::pow(pre_pre_phi[8],r-3)*pre_pre_phi_grad[8]
      + pre_pre_phi_grad[9]*std::pow(pre_pre_phi[8],r-2);
      pre_phi_dx[9*r-1+r-2] = dzBubble[0]*zvalue+zBubble*ztemp[0];
      pre_phi_dy[9*r-1+r-2] = dzBubble[1]*zvalue+zBubble*ztemp[1];
      pre_phi_dz[9*r-1+r-2] = dzBubble[2]*zvalue+zBubble*ztemp[2];

      zvalue = pre_pre_phi[10]*std::pow(pre_pre_phi[8],r-2);
      ztemp = pre_pre_phi[10]*(r-2)*std::pow(pre_pre_phi[8],r-3)*pre_pre_phi_grad[8]
      + pre_pre_phi_grad[10]*std::pow(pre_pre_phi[8],r-2);
      pre_phi_dx[10*r-2+r-2] = dzBubble[0]*zvalue+zBubble*ztemp[0];
      pre_phi_dy[10*r-2+r-2] = dzBubble[1]*zvalue+zBubble*ztemp[1];
      pre_phi_dz[10*r-2+r-2] = dzBubble[2]*zvalue+zBubble*ztemp[2];

      zvalue = pre_pre_phi[9]*pre_pre_phi[10]*std::pow(pre_pre_phi[8],r-2); 
      ztemp = pre_pre_phi[9]*pre_pre_phi[10]*(r-2)*std::pow(pre_pre_phi[8],r-3)*pre_pre_phi_grad[8]
      + pre_pre_phi_grad[9]*pre_pre_phi[10]*std::pow(pre_pre_phi[8],r-2)
      + pre_pre_phi[9]*pre_pre_phi_grad[10]*std::pow(pre_pre_phi[8],r-2);
      pre_phi_dx[11*r-3+r-2] = dzBubble[0]*zvalue+zBubble*ztemp[0];
      pre_phi_dy[11*r-3+r-2] = dzBubble[1]*zvalue+zBubble*ztemp[1];
      pre_phi_dz[11*r-3+r-2] = dzBubble[2]*zvalue+zBubble*ztemp[2];

    } //end dim==3

    Vector<double> soln;
    Vector<double> soln_dx;
    Vector<double> soln_dy;
    Vector<double> soln_dz;
    double pre_phi_int_2d=0.;
    double pre_phi_int_dx=0.;
    double pre_phi_int_dy=0.;
    double pre_phi_int_dz=0.;

    if(dim==2)
    {
      soln.reinit(4*fe_degree);
      soln_dx.reinit(4*fe_degree);
      soln_dy.reinit(4*fe_degree);
      A.Tvmult(soln, pre_phi);
      A.Tvmult(soln_dx, pre_phi_dx);
      A.Tvmult(soln_dy, pre_phi_dy);

      pre_phi_int_2d = pre_pre_phi[0]*pre_pre_phi[1]
      *pre_pre_phi[2]*pre_pre_phi[3];
      // lambda_0*lambda_1*lambda_2*lambda_3

      pre_phi_int_dx = 
      pre_pre_phi_grad[0][0]*pre_pre_phi[1]
      *pre_pre_phi[2]*pre_pre_phi[3]
      + pre_pre_phi[0]*pre_pre_phi_grad[1][0]
      *pre_pre_phi[2]*pre_pre_phi[3]
      + pre_pre_phi[0]*pre_pre_phi[1]
      *pre_pre_phi_grad[2][0]*pre_pre_phi[3]
      + pre_pre_phi[0]*pre_pre_phi[1]
      *pre_pre_phi[2]*pre_pre_phi_grad[3][0];

      pre_phi_int_dy = 
      pre_pre_phi_grad[0][1]*pre_pre_phi[1]
      *pre_pre_phi[2]*pre_pre_phi[3]
      + pre_pre_phi[0]*pre_pre_phi_grad[1][1]
      *pre_pre_phi[2]*pre_pre_phi[3]
      + pre_pre_phi[0]*pre_pre_phi[1]
      *pre_pre_phi_grad[2][1]*pre_pre_phi[3]
      + pre_pre_phi[0]*pre_pre_phi[1]
      *pre_pre_phi[2]*pre_pre_phi_grad[3][1];
    }

    if(dim==3)
    {
      soln.reinit(12*fe_degree-4);
      soln_dx.reinit(12*fe_degree-4);
      soln_dy.reinit(12*fe_degree-4);
      soln_dz.reinit(12*fe_degree-4);      
      A.Tvmult(soln, pre_phi);
      A.Tvmult(soln_dx, pre_phi_dx);
      A.Tvmult(soln_dy, pre_phi_dy);
      A.Tvmult(soln_dz, pre_phi_dz);
    }

    for(unsigned int i=0; i<this->dofs_per_cell; ++i)
    {
      if(dim==2)
      {
         if(i<4*fe_degree)
         {
          //vertex and edge dofs
          if (flags & update_values)
            data.shape_values(i,k) = soln[i];

          if (flags & update_gradients)
          {
            data.shape_gradients[i][k][0] = soln_dx[i];
            data.shape_gradients[i][k][1] = soln_dy[i];
          }
         }
         else
         {
          Assert(fe_degree>=4, ExcMessage("if r<4, should see no interior dof"));
          Assert(i-4*fe_degree>=0, ExcMessage("if r<4, should see no interior dof"));
          int p=0, q=0;

          switch(i-4*fe_degree)
          {
            case 0:
              p=0;q=0; break;
            case 1:
              p=1;q=0; break;
            case 2:
              p=0;q=1; break;
            default:
              Assert(false, ExcNotImplemented());
          }

          // interior dofs
          if(flags & update_values)
          {
            data.shape_values(i,k) = pre_phi_int_2d* std::pow(pre_pre_phi[4],p)
            * std::pow(pre_pre_phi[5],q);
          }
          if (flags & update_gradients)
          { 
            data.shape_gradients[i][k][0] = 
            pre_phi_int_dx * std::pow(pre_pre_phi[4],p)
            * std::pow(pre_pre_phi[5],q)
            + pre_phi_int_2d * ((p==0)?0.: p*std::pow(pre_pre_phi[4],p-1)*pre_pre_phi_grad[4][0])
            * std::pow(pre_pre_phi[5],q)
            + pre_phi_int_2d * std::pow(pre_pre_phi[4],p)
            * ((q==0)? 0.: q*std::pow(pre_pre_phi[5],q-1)*pre_pre_phi_grad[5][0]);

            data.shape_gradients[i][k][1] = 
            pre_phi_int_dy * std::pow(pre_pre_phi[4],p)
            * std::pow(pre_pre_phi[5],q)
            + pre_phi_int_2d * ((p==0)?0.: p*std::pow(pre_pre_phi[4],p-1)*pre_pre_phi_grad[4][1])
            * std::pow(pre_pre_phi[5],q)
            + pre_phi_int_2d * std::pow(pre_pre_phi[4],p)
            * ((q==0)? 0.: q*std::pow(pre_pre_phi[5],q-1)*pre_pre_phi_grad[5][1]);
          }
         }
      } //end if dim==2 
      else if(dim==3)
      {
        if(i<12*fe_degree-4)
         {
          //vertex and edge dofs
          if (flags & update_values)
            data.shape_values(i,k) = soln[i];

          if (flags & update_gradients)
          {
            data.shape_gradients[i][k][0] = soln_dx[i];
            data.shape_gradients[i][k][1] = soln_dy[i];
            data.shape_gradients[i][k][2] = soln_dz[i];
          }
         }
         else
         {
          Assert(r>3,ExcMessage("should not go here if degree<=3"));
          Assert(r<5,ExcMessage("higer order not available yet"));
          int ii = i-(12*fe_degree-4);

          if (flags & update_values)
            data.shape_values(i,k) = pre_phi_int[ii];

          if (flags & update_gradients)
            data.shape_gradients[i][k]= pre_phi_int_grad[ii];
         }
      }
    }
  } //end of step 4)
}

template <class POLY, int dim, int spacedim>
void
FE_S_NP_Base<POLY,dim,spacedim>::fill_fe_face_values (
  const Mapping<dim,spacedim>                      &mapping,
  const typename Triangulation<dim,spacedim>::cell_iterator &cell,
  const unsigned int                                  face,
  const Quadrature<dim-1>                            &quadrature,
  typename Mapping<dim,spacedim>::InternalDataBase &mapping_data,
  typename Mapping<dim,spacedim>::InternalDataBase &fedata,
  FEValuesData<dim,spacedim>                       &data) const
{
  Assert (dynamic_cast<InternalData *> (&fedata) != 0,
          ExcInternalError());
  InternalData &fe_data = static_cast<InternalData &> (fedata);

  const typename QProjector<dim>::DataSetDescriptor offset
    = QProjector<dim>::DataSetDescriptor::face (face,
                                                cell->face_orientation(face),
                                                cell->face_flip(face),
                                                cell->face_rotation(face),
                                                quadrature.size());

  const UpdateFlags flags(fe_data.current_update_flags());

  Assert (flags & update_quadrature_points, ExcInternalError());
  const unsigned int n_q_points = data.quadrature_points.size();

  unsigned int fe_degree = this->degree;
  if(dim==2)
  {
    Assert(fe_degree>1, ExcMessage("2D non parametric serendipity only for degree>1"));
  }else{
    Assert(fe_degree>2, ExcMessage("3D non parametric serendipity only for degree>2"));
  }

  // alternative formulation
  bool flag_new = true;

  // 1) Set face normals
  compute_mapping_support_points(cell, fe_data.mapping_support_points);
  const Tensor<1,spacedim> *supp_pts = &fe_data.mapping_support_points[0];
  std::vector<Tensor<1,dim> > Gamma;
  Gamma.resize(GeometryInfo<dim>::faces_per_cell);
  std::vector<Tensor<1,dim> > Gamma_mid;
  const double measure = cell->measure();
  const double h = std::pow(measure, 1.0/dim);

  if(dim==2)
  {
    std::vector<Tensor<1, dim> >tangential;
    tangential.resize(4);
    for(unsigned int k = 0; k<n_shape_functions; ++k)
      for(unsigned int d = 0; d<dim; ++d)
      { // face 0: J * [0 -1]^T
        tangential[0][d] += -1.*supp_pts[k][d]*fe_data.corner_derivative(0,k)[1];
        // face 1: J * [0 1]^T
        tangential[1][d] += supp_pts[k][d]*fe_data.corner_derivative(1,k)[1];
        // face 2: J * [1 0]^T
        tangential[2][d] += supp_pts[k][d]*fe_data.corner_derivative(0,k)[0];
        // face 3: J * [-1 0]^T
        tangential[3][d] += -1.*supp_pts[k][d]*fe_data.corner_derivative(2,k)[0];
      }

    // std::cout<<"**** cell "<<cell->index()<<" *******"<<std::endl;
    for(unsigned int face_no=0 ; face_no<4; ++face_no)
    {
      cross_product(Gamma[face_no], tangential[face_no]);
      Gamma[face_no] = -Gamma[face_no]/(Gamma[face_no].norm()*h);
      // std::cout<<"Gamma "<<face_no<<":"<< Gamma[face_no]<<std::endl;
    }
  } // end dim==2

  if(dim==3)
  {
    Gamma_mid.resize(dim);

    std::vector<Tensor<1, dim> >tangential;
    tangential.resize(15);

    for(unsigned int k = 0; k<n_shape_functions; ++k)
      for(unsigned int d = 0; d<dim; ++d)
      { 
        // face 0 tang 1: J * [0 -1  0]^T
        tangential[0][d] += -1.*supp_pts[k][d]*fe_data.corner_derivative(0,k)[1];
        // face 0 tang 2: J * [0 0 1]^T
        tangential[1][d] += supp_pts[k][d]*fe_data.corner_derivative(0,k)[2];
        // face 1 tang 1: J * [0 1  0]^T
        tangential[2][d] += supp_pts[k][d]*fe_data.corner_derivative(1,k)[1];
        // face 1 tang 2: J * [0 0 1]^T
        tangential[3][d] += supp_pts[k][d]*fe_data.corner_derivative(1,k)[2];
        // face 2 tang 1: J * [0 0  -1]^T
        tangential[4][d] += -1.*supp_pts[k][d]*fe_data.corner_derivative(0,k)[2];
        // face 2 tang 2: J * [1 0 0]^T
        tangential[5][d] += supp_pts[k][d]*fe_data.corner_derivative(0,k)[0];
        // face 3 tang 1: J * [0 0  1]^T
        tangential[6][d] += supp_pts[k][d]*fe_data.corner_derivative(2,k)[2];
        // face 3 tang 2: J * [1 0 0]^T
        tangential[7][d] += supp_pts[k][d]*fe_data.corner_derivative(2,k)[0];
        // face 4 tang 1: J * [-1 0 0]^T
        tangential[8][d] += -1.*supp_pts[k][d]*fe_data.corner_derivative(0,k)[0];
        // face 4 tang 2: J * [0 1 0]^T
        tangential[9][d] += supp_pts[k][d]*fe_data.corner_derivative(0,k)[1];
        // face 5 tang 1: J * [1 0 0]^T
        tangential[10][d] += supp_pts[k][d]*fe_data.corner_derivative(4,k)[0];
        // face 5 tang 2: J * [0 1 0]^T
        tangential[11][d] += supp_pts[k][d]*fe_data.corner_derivative(4,k)[1]; 

        //tang 1: J* [1 0 0]
        tangential[12][d] += supp_pts[k][d]*fe_data.center_derivative(6,k)[0];
        //tang 2: J* [0 1 0]
        tangential[13][d] += supp_pts[k][d]*fe_data.center_derivative(6,k)[1];
        //tang 3: J* [0 0 1]
        tangential[14][d] += supp_pts[k][d]*fe_data.center_derivative(6,k)[2];
      }

    for(unsigned int face_no=0 ; face_no<6; ++face_no)
    {
      cross_product(Gamma[face_no], tangential[2*face_no], tangential[2*face_no+1]);
      Gamma[face_no] = Gamma[face_no]/(Gamma[face_no].norm());
      // std::cout<<"Gamma "<<face_no<<":"<< Gamma[face_no]<<std::endl;
    }

    cross_product(Gamma_mid[0], tangential[13], tangential[14]);
    Gamma_mid[0] = Gamma_mid[0]/(Gamma_mid[0].norm());

    cross_product(Gamma_mid[1], tangential[14], tangential[12]);
    Gamma_mid[1] = Gamma_mid[1]/(Gamma_mid[1].norm());

    cross_product(Gamma_mid[2], tangential[12], tangential[13]);
    Gamma_mid[2] = Gamma_mid[2]/(Gamma_mid[2].norm());

  } //end dim==3

  FullMatrix<double> B((dim==3)?6:0);
  Tensor<1,dim> center_pt;

  if(dim==3)
  {

    for(unsigned int j=0; j<6; ++j)
      for(unsigned int k=0; k<6; ++k)
      {
        if((j==k)||((j-(j/2)*2)==0 && k == (j+1))||((j-(j/2)*2)==1 && k == (j-1)))
          B[j][k] = 0.0;
        else
        {
          Tensor<1,dim> pj;
          pj = Gamma[j] - (Gamma[j]*Gamma[k])*Gamma[k];
          Assert(pj.norm()>1e-13, ExcMessage("Geometry degenerate!"));
          B[j][k] = 1/pj.norm();
        }
      }

    // reset B[4][2],B[4][0],B[5][2],B[5][0]
    Tensor<1,dim> temp;
    temp = Gamma[4]-(Gamma[4]*Gamma[0])*Gamma[0] - (Gamma[4]*Gamma[2])*Gamma[2];
    B[4][2] = 1/temp.norm();
    B[4][0] = 1/temp.norm();
    B[4][1] = 1/temp.norm();
    B[4][3] = 1/temp.norm();
    temp = Gamma[5]-(Gamma[5]*Gamma[0])*Gamma[0] - (Gamma[5]*Gamma[2])*Gamma[2];
    B[5][2] = 1/temp.norm();
    B[5][0] = 1/temp.norm();
    B[5][1] = 1/temp.norm();
    B[5][3] = 1/temp.norm();
    //scale all lambdas
    for(unsigned int face_no=0 ; face_no<6; ++face_no)
      Gamma[face_no] = -Gamma[face_no]/h;
      
    Gamma_mid[0] = Gamma_mid[0]/h;
    Gamma_mid[1] = Gamma_mid[1]/h;
    Gamma_mid[2] = Gamma_mid[2]/h;    

    //calculate volume center point
    for(unsigned int k = 0; k<n_shape_functions; ++k)
      center_pt += supp_pts[k]*fe_data.center_value(6,k);          
  }

  // 2) Set A matrix
  unsigned int size_A=0;
  if(dim==2)
    size_A = 4*fe_degree;
  else if (dim==3)
    size_A = 8+12*(fe_degree-1);

  FullMatrix<double> A(size_A);
  std::vector<double> pre_pre_phi;
  std::vector<Tensor<1,dim> > pre_pre_phi_grad;
  const unsigned int r = fe_degree;

  double alpha_x, beta_x, alpha_y, beta_y, alpha_z, beta_z,eta_x,eta_y,eta_z;
  alpha_x = 0.0; beta_x = 0.0; eta_x = 1.0;
  alpha_y = 0.0; beta_y = 0.0; eta_y = 1.0;
  alpha_z = 0.0; beta_z = 0.0; eta_z = 1.0;

  double sigma_v,eta_v,sigma_h,eta_h;
  sigma_v = 1.0; eta_v = 1.0;
  sigma_h = 1.0; eta_h = 1.0;

  std::vector<double> pre_phi_int;
  std::vector<Tensor<1,dim> > pre_phi_int_grad;



  if(dim==2)
  {
    pre_pre_phi.resize(8);
    pre_pre_phi_grad.resize(8);

    // Tensor<1,dim> gamma_t = Gamma[2] - Gamma[3];
    // gamma_t = gamma_t/gamma_t.norm();
    // double cost = gamma_t*Gamma[0]/Gamma[0].norm();
    // sigma_v = 1.0/std::sqrt(1.0-cost*cost);
    // cost = gamma_t*Gamma[1]/Gamma[1].norm();
    // eta_v = 1.0/std::sqrt(1.0-cost*cost);

    // gamma_t = Gamma[0] - Gamma[1];
    // gamma_t = gamma_t/gamma_t.norm();
    // cost = gamma_t*Gamma[2]/Gamma[2].norm();
    // sigma_h = 1.0/std::sqrt(1.0-cost*cost);
    // cost = gamma_t*Gamma[3]/Gamma[3].norm();
    // eta_h = 1.0/std::sqrt(1.0-cost*cost);

    // std::cout<<"sigma_v:" <<sigma_v
    // <<"   eta_v:"<<eta_v
    // <<"   sigma_h:"<<sigma_h
    // <<"   eta_h:"<<eta_h
    // <<std::endl;

    unsigned int row_no = 0;
    for(unsigned int vertex_no = 0; vertex_no<4; ++vertex_no, ++row_no)
    {
      // set vertex dofs
      Tensor<1, dim> dof_pt;
      dof_pt = supp_pts[vertex_no];
      // set pre_pre_phi values
      pre_pre_phi[0] = (dof_pt - supp_pts[0])*Gamma[0]; //lambda_0
      pre_pre_phi[1] = (dof_pt - supp_pts[1])*Gamma[1]; //lambda_1
      pre_pre_phi[2] = (dof_pt - supp_pts[0])*Gamma[2]; //lambda_2
      pre_pre_phi[3] = (dof_pt - supp_pts[2])*Gamma[3]; //lambda_3
      Assert(pre_pre_phi[0]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[1]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[2]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[3]>=-1e-13, ExcMessage("dof point out of cell"));
      pre_pre_phi[4] = pre_pre_phi[0] - pre_pre_phi[1]; //lambda_V
      pre_pre_phi[5] = pre_pre_phi[2] - pre_pre_phi[3]; //lambda_H
      pre_pre_phi[6] = pre_pre_phi[4] / (sigma_v*pre_pre_phi[0] + eta_v*pre_pre_phi[1]); //R_V
      pre_pre_phi[7] = pre_pre_phi[5] / (sigma_h*pre_pre_phi[2] + eta_h*pre_pre_phi[3]); //R_H
      AssertIsFinite(pre_pre_phi[6]);
      AssertIsFinite(pre_pre_phi[7]);
      // set matrix A
      A[row_no][0] = pre_pre_phi[1]*pre_pre_phi[3]; //lambda_1*lambda_3
      A[row_no][1] = pre_pre_phi[0]*pre_pre_phi[3]; //lambda_0*lambda_3
      A[row_no][2] = pre_pre_phi[1]*pre_pre_phi[2]; //lambda_1*lambda_2
      A[row_no][3] = pre_pre_phi[0]*pre_pre_phi[2]; //lambda_0*lambda_2
    }

    Assert(row_no==4, ExcMessage("row number not correct"));

    for(unsigned int line_no=1; line_no<fe_degree; ++line_no, ++row_no)
    {
      // left line dofs
      Point<dim> dof_pt;
      dof_pt[0] = supp_pts[0][0]+(supp_pts[2][0]-supp_pts[0][0])*line_no/fe_degree;
      dof_pt[1] = supp_pts[0][1]+(supp_pts[2][1]-supp_pts[0][1])*line_no/fe_degree;

      // set pre_pre_phi values
      pre_pre_phi[0] = (dof_pt - supp_pts[0])*Gamma[0]; //lambda_0
      pre_pre_phi[1] = (dof_pt - supp_pts[1])*Gamma[1]; //lambda_1
      pre_pre_phi[2] = (dof_pt - supp_pts[0])*Gamma[2]; //lambda_2
      pre_pre_phi[3] = (dof_pt - supp_pts[2])*Gamma[3]; //lambda_3
      Assert(pre_pre_phi[0]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[1]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[2]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[3]>=-1e-13, ExcMessage("dof point out of cell"));
      pre_pre_phi[4] = pre_pre_phi[0] - pre_pre_phi[1]; //lambda_V
      pre_pre_phi[5] = pre_pre_phi[2] - pre_pre_phi[3]; //lambda_H
      pre_pre_phi[6] = pre_pre_phi[4] / (sigma_v*pre_pre_phi[0] + eta_v*pre_pre_phi[1]); //R_V
      pre_pre_phi[7] = pre_pre_phi[5] / (sigma_h*pre_pre_phi[2] + eta_h*pre_pre_phi[3]); //R_H
      AssertIsFinite(pre_pre_phi[6]);
      AssertIsFinite(pre_pre_phi[7]);

      // set matrix A
      A[row_no][0] = pre_pre_phi[1]*pre_pre_phi[3]; //lambda_1*lambda_3
      A[row_no][1] = pre_pre_phi[0]*pre_pre_phi[3]; //lambda_0*lambda_3
      A[row_no][2] = pre_pre_phi[1]*pre_pre_phi[2]; //lambda_1*lambda_2
      A[row_no][3] = pre_pre_phi[0]*pre_pre_phi[2]; //lambda_0*lambda_2

      for(unsigned int j=0; j< fe_degree-1; ++j)
      {
        A[row_no][4+j] = pre_pre_phi[2]*pre_pre_phi[3]*std::pow(pre_pre_phi[5],j);
        A[row_no][fe_degree+3+j] = 
        pre_pre_phi[2]*pre_pre_phi[3]*pre_pre_phi[4]*std::pow(pre_pre_phi[5],j);
      }
      A[row_no][2*fe_degree+1] = 
        pre_pre_phi[2]*pre_pre_phi[3]*pre_pre_phi[6]*std::pow(pre_pre_phi[5],fe_degree-2);

      if(flag_new)
      {
        Point<dim> dof_pt_hat;
        dof_pt_hat[0] = 0.0; dof_pt_hat[1] = (double)line_no/fe_degree;
        A[row_no][2*fe_degree+1] = 
        this->poly_space.compute_value(2*fe_degree+1, dof_pt_hat)*(-4.0);
      }
    }

    Assert(row_no==fe_degree+3, ExcMessage("row number not correct"));

    for(unsigned int line_no=1; line_no<fe_degree; ++line_no, ++row_no)
    {
      // right line dofs
      Point<dim> dof_pt;
      dof_pt[0] = supp_pts[1][0]+(supp_pts[3][0]-supp_pts[1][0])*line_no/fe_degree;
      dof_pt[1] = supp_pts[1][1]+(supp_pts[3][1]-supp_pts[1][1])*line_no/fe_degree;
      // set pre_pre_phi values
      pre_pre_phi[0] = (dof_pt - supp_pts[0])*Gamma[0]; //lambda_0
      pre_pre_phi[1] = (dof_pt - supp_pts[1])*Gamma[1]; //lambda_1
      pre_pre_phi[2] = (dof_pt - supp_pts[0])*Gamma[2]; //lambda_2
      pre_pre_phi[3] = (dof_pt - supp_pts[2])*Gamma[3]; //lambda_3
      Assert(pre_pre_phi[0]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[1]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[2]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[3]>=-1e-13, ExcMessage("dof point out of cell"));
      pre_pre_phi[4] = pre_pre_phi[0] - pre_pre_phi[1]; //lambda_V
      pre_pre_phi[5] = pre_pre_phi[2] - pre_pre_phi[3]; //lambda_H
      pre_pre_phi[6] = pre_pre_phi[4] / (sigma_v*pre_pre_phi[0] + eta_v*pre_pre_phi[1]); //R_V
      pre_pre_phi[7] = pre_pre_phi[5] / (sigma_h*pre_pre_phi[2] + eta_h*pre_pre_phi[3]); //R_H
      AssertIsFinite(pre_pre_phi[6]);
      AssertIsFinite(pre_pre_phi[7]);

      // set matrix A
      A[row_no][0] = pre_pre_phi[1]*pre_pre_phi[3]; //lambda_1*lambda_3
      A[row_no][1] = pre_pre_phi[0]*pre_pre_phi[3]; //lambda_0*lambda_3
      A[row_no][2] = pre_pre_phi[1]*pre_pre_phi[2]; //lambda_1*lambda_2
      A[row_no][3] = pre_pre_phi[0]*pre_pre_phi[2]; //lambda_0*lambda_2

      for(unsigned int j=0; j< fe_degree-1; ++j)
      {
        A[row_no][4+j] = pre_pre_phi[2]*pre_pre_phi[3]*std::pow(pre_pre_phi[5],j);
        A[row_no][fe_degree+3+j] = 
        pre_pre_phi[2]*pre_pre_phi[3]*pre_pre_phi[4]*std::pow(pre_pre_phi[5],j);
      }
      A[row_no][2*fe_degree+1] = 
        pre_pre_phi[2]*pre_pre_phi[3]*pre_pre_phi[6]*std::pow(pre_pre_phi[5],fe_degree-2);

      if(flag_new)
      {
        Point<dim> dof_pt_hat;
        dof_pt_hat[0] = 1.0; dof_pt_hat[1] = (double)line_no/fe_degree;
        A[row_no][2*fe_degree+1] = 
        this->poly_space.compute_value(2*fe_degree+1, dof_pt_hat)*(-4.0);
      }
    }

    Assert(row_no==2*fe_degree+2, ExcMessage("row number not correct"));

    for(unsigned int line_no=1; line_no<fe_degree; ++line_no, ++row_no)
    {
      // bottom line dofs
      Point<dim> dof_pt;
      dof_pt[0] = supp_pts[0][0]+(supp_pts[1][0]-supp_pts[0][0])*line_no/fe_degree;
      dof_pt[1] = supp_pts[0][1]+(supp_pts[1][1]-supp_pts[0][1])*line_no/fe_degree;
      // set pre_pre_phi values
      pre_pre_phi[0] = (dof_pt - supp_pts[0])*Gamma[0]; //lambda_0
      pre_pre_phi[1] = (dof_pt - supp_pts[1])*Gamma[1]; //lambda_1
      pre_pre_phi[2] = (dof_pt - supp_pts[0])*Gamma[2]; //lambda_2
      pre_pre_phi[3] = (dof_pt - supp_pts[2])*Gamma[3]; //lambda_3
      Assert(pre_pre_phi[0]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[1]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[2]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[3]>=-1e-13, ExcMessage("dof point out of cell"));
      pre_pre_phi[4] = pre_pre_phi[0] - pre_pre_phi[1]; //lambda_V
      pre_pre_phi[5] = pre_pre_phi[2] - pre_pre_phi[3]; //lambda_H
      pre_pre_phi[6] = pre_pre_phi[4] / (sigma_v*pre_pre_phi[0] + eta_v*pre_pre_phi[1]); //R_V
      pre_pre_phi[7] = pre_pre_phi[5] / (sigma_h*pre_pre_phi[2] + eta_h*pre_pre_phi[3]); //R_H
      AssertIsFinite(pre_pre_phi[6]);
      AssertIsFinite(pre_pre_phi[7]);

      // set matrix A
      A[row_no][0] = pre_pre_phi[1]*pre_pre_phi[3]; //lambda_1*lambda_3
      A[row_no][1] = pre_pre_phi[0]*pre_pre_phi[3]; //lambda_0*lambda_3
      A[row_no][2] = pre_pre_phi[1]*pre_pre_phi[2]; //lambda_1*lambda_2
      A[row_no][3] = pre_pre_phi[0]*pre_pre_phi[2]; //lambda_0*lambda_2

      for(unsigned int j=0; j< fe_degree-1; ++j)
      {
        A[row_no][2*fe_degree+2+j] = pre_pre_phi[0]*pre_pre_phi[1]*std::pow(pre_pre_phi[4],j);
        A[row_no][3*fe_degree+1+j] = 
        pre_pre_phi[0]*pre_pre_phi[1]*pre_pre_phi[5]*std::pow(pre_pre_phi[4],j);
      }
      A[row_no][4*fe_degree-1] = 
        pre_pre_phi[0]*pre_pre_phi[1]*pre_pre_phi[7]*std::pow(pre_pre_phi[4],fe_degree-2);

      if(flag_new){
        Point<dim> dof_pt_hat;
        dof_pt_hat[0] = (double)line_no/fe_degree; dof_pt_hat[1] = 0.0;
        A[row_no][4*fe_degree-1] = 
        this->poly_space.compute_value(4*fe_degree-1, dof_pt_hat)*(-4.0);
      }
    }

    Assert(row_no==3*fe_degree+1, ExcMessage("row number not correct"));

    for(unsigned int line_no=1; line_no<fe_degree; ++line_no, ++row_no)
    {
      // top line dofs
      Point<dim> dof_pt;
      dof_pt[0] = supp_pts[2][0]+(supp_pts[3][0]-supp_pts[2][0])*line_no/fe_degree;
      dof_pt[1] = supp_pts[2][1]+(supp_pts[3][1]-supp_pts[2][1])*line_no/fe_degree;
      // set pre_pre_phi values
      pre_pre_phi[0] = (dof_pt - supp_pts[0])*Gamma[0]; //lambda_0
      pre_pre_phi[1] = (dof_pt - supp_pts[1])*Gamma[1]; //lambda_1
      pre_pre_phi[2] = (dof_pt - supp_pts[0])*Gamma[2]; //lambda_2
      pre_pre_phi[3] = (dof_pt - supp_pts[2])*Gamma[3]; //lambda_3
      Assert(pre_pre_phi[0]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[1]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[2]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[3]>=-1e-13, ExcMessage("dof point out of cell"));
      pre_pre_phi[4] = pre_pre_phi[0] - pre_pre_phi[1]; //lambda_V
      pre_pre_phi[5] = pre_pre_phi[2] - pre_pre_phi[3]; //lambda_H
      pre_pre_phi[6] = pre_pre_phi[4] / (sigma_v*pre_pre_phi[0] + eta_v*pre_pre_phi[1]); //R_V
      pre_pre_phi[7] = pre_pre_phi[5] / (sigma_h*pre_pre_phi[2] + eta_h*pre_pre_phi[3]); //R_H
      AssertIsFinite(pre_pre_phi[6]);
      AssertIsFinite(pre_pre_phi[7]);

      // set matrix A
      A[row_no][0] = pre_pre_phi[1]*pre_pre_phi[3]; //lambda_1*lambda_3
      A[row_no][1] = pre_pre_phi[0]*pre_pre_phi[3]; //lambda_0*lambda_3
      A[row_no][2] = pre_pre_phi[1]*pre_pre_phi[2]; //lambda_1*lambda_2
      A[row_no][3] = pre_pre_phi[0]*pre_pre_phi[2]; //lambda_0*lambda_2

      for(unsigned int j=0; j< fe_degree-1; ++j)
      {
        A[row_no][2*fe_degree+2+j] = pre_pre_phi[0]*pre_pre_phi[1]*std::pow(pre_pre_phi[4],j);
        A[row_no][3*fe_degree+1+j] = 
        pre_pre_phi[0]*pre_pre_phi[1]*pre_pre_phi[5]*std::pow(pre_pre_phi[4],j);
      }
      A[row_no][4*fe_degree-1] = 
        pre_pre_phi[0]*pre_pre_phi[1]*pre_pre_phi[7]*std::pow(pre_pre_phi[4],fe_degree-2);

      if(flag_new)
      {
        Point<dim> dof_pt_hat;
        dof_pt_hat[0] = (double)line_no/fe_degree; dof_pt_hat[1] = 1.0;
        A[row_no][4*fe_degree-1] = 
        this->poly_space.compute_value(4*fe_degree-1, dof_pt_hat)*(-4.0);
      }
    }
  } //end (dim==2)

  if(dim==3)
  {
    pre_pre_phi.resize(12);
    pre_pre_phi_grad.resize(12);

    Assert(r>2,ExcMessage("3D FE_S_NP only works for degree>=3"));

    for(unsigned int row_no=0; row_no<12*r-4; ++row_no)
    {
      // set support point
      Tensor<1,dim> dof_pt;
      if(row_no<8)
      {
        dof_pt = supp_pts[row_no];
      }
      else
      {
        if(row_no>= 8 && row_no<7+r)
          dof_pt = supp_pts[0] + (supp_pts[2]-supp_pts[0])*(row_no-7)/r;
        if(row_no>= 7+r && row_no<6+2*r)
          dof_pt = supp_pts[1] + (supp_pts[3]-supp_pts[1])*(row_no-6-r)/r;
        if(row_no>= 6+2*r && row_no<5+3*r)
          dof_pt = supp_pts[0] + (supp_pts[1]-supp_pts[0])*(row_no-5-2*r)/r;
        if(row_no>= 5+3*r && row_no<4+4*r)
          dof_pt = supp_pts[2] + (supp_pts[3]-supp_pts[2])*(row_no-4-3*r)/r;
        if(row_no>= 4+4*r && row_no<3+5*r)
          dof_pt = supp_pts[4] + (supp_pts[6]-supp_pts[4])*(row_no-3-4*r)/r;
        if(row_no>= 3+5*r && row_no<2+6*r)
          dof_pt = supp_pts[5] + (supp_pts[7]-supp_pts[5])*(row_no-2-5*r)/r;
        if(row_no>=2+6*r && row_no<1+7*r)
          dof_pt = supp_pts[4] + (supp_pts[5]-supp_pts[4])*(row_no-1-6*r)/r;
        if(row_no>=1+7*r && row_no<8*r)
          dof_pt = supp_pts[6] + (supp_pts[7]-supp_pts[6])*(row_no-7*r)/r;
        if(row_no>=8*r && row_no<9*r-1)
          dof_pt = supp_pts[0] + (supp_pts[4]-supp_pts[0])*(row_no+1-8*r)/r;
        if(row_no>=9*r-1 && row_no<10*r-2)
          dof_pt = supp_pts[1] + (supp_pts[5]-supp_pts[1])*(row_no+2-9*r)/r;
        if(row_no>=10*r-2 && row_no<11*r-3)
          dof_pt = supp_pts[2] + (supp_pts[6]-supp_pts[2])*(row_no+3-10*r)/r;
        if(row_no>=11*r-3 && row_no<12*r-4)
          dof_pt = supp_pts[3] + (supp_pts[7]-supp_pts[3])*(row_no+4-11*r)/r;
      }
      // set pre_pre_phi values
      pre_pre_phi[0] = (dof_pt - supp_pts[0])*Gamma[0]; //lambda_0
      pre_pre_phi[1] = (dof_pt - supp_pts[1])*Gamma[1]; //lambda_1
      pre_pre_phi[2] = (dof_pt - supp_pts[0])*Gamma[2]; //lambda_2
      pre_pre_phi[3] = (dof_pt - supp_pts[2])*Gamma[3]; //lambda_3
      pre_pre_phi[4] = (dof_pt - supp_pts[0])*Gamma[4]; //lambda_4
      pre_pre_phi[5] = (dof_pt - supp_pts[4])*Gamma[5]; //lambda_5
      Assert(pre_pre_phi[0]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[1]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[2]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[3]>=-1e-13, ExcMessage("dof point out of cell")); 
      Assert(pre_pre_phi[4]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[5]>=-1e-13, ExcMessage("dof point out of cell"));          

      // set matrix A entries
      A[row_no][0] = pre_pre_phi[1]*pre_pre_phi[3]*pre_pre_phi[5];
      A[row_no][1] = pre_pre_phi[0]*pre_pre_phi[3]*pre_pre_phi[5];
      A[row_no][2] = pre_pre_phi[1]*pre_pre_phi[2]*pre_pre_phi[5];
      A[row_no][3] = pre_pre_phi[0]*pre_pre_phi[2]*pre_pre_phi[5];
      A[row_no][4] = pre_pre_phi[1]*pre_pre_phi[3]*pre_pre_phi[4];
      A[row_no][5] = pre_pre_phi[0]*pre_pre_phi[3]*pre_pre_phi[4];
      A[row_no][6] = pre_pre_phi[1]*pre_pre_phi[2]*pre_pre_phi[4];
      A[row_no][7] = pre_pre_phi[0]*pre_pre_phi[2]*pre_pre_phi[4];

      if((row_no>=8 && row_no<6+2*r) || (row_no>=4+4*r && r<2+6*r))
      {
        // y dir
        // R_x
        pre_pre_phi[9] = (B[0][4]*pre_pre_phi[0] - B[1][4]*pre_pre_phi[1])
        /(B[0][4]*pre_pre_phi[0] + B[1][4]*pre_pre_phi[1]);
        // R_z
        pre_pre_phi[11] = (B[4][0]*pre_pre_phi[4] - B[5][0]*pre_pre_phi[5])
        /(B[4][0]*pre_pre_phi[4] + B[5][0]*pre_pre_phi[5]); 

        AssertIsFinite(pre_pre_phi[9]);
        AssertIsFinite(pre_pre_phi[11]);        

        pre_pre_phi[7] = (dof_pt - center_pt)*Gamma_mid[1];
        pre_pre_phi[6] = B[0][4]*pre_pre_phi[0] - B[1][4]*pre_pre_phi[1];
        pre_pre_phi[8] = B[4][0]*pre_pre_phi[4] - B[5][0]*pre_pre_phi[5];


        double temp = pre_pre_phi[2]*pre_pre_phi[3];
        A[row_no][8] = temp;
        A[row_no][7+r] = temp*pre_pre_phi[6];
        A[row_no][4+4*r] = temp*pre_pre_phi[8];
        A[row_no][3+5*r] = temp*(alpha_y*pre_pre_phi[8]*pre_pre_phi[9]
          +beta_y*pre_pre_phi[6]*pre_pre_phi[11]
          +eta_y*pre_pre_phi[9]*pre_pre_phi[11]);

        for(unsigned int j=1; j<r-2; ++j)
        {
          A[row_no][8+j] = temp*std::pow(pre_pre_phi[7],j);
          A[row_no][7+r+j] = temp*pre_pre_phi[6]*std::pow(pre_pre_phi[7],j);
          A[row_no][4+4*r+j] = temp*pre_pre_phi[8]*std::pow(pre_pre_phi[7],j);
          A[row_no][3+5*r+j] = temp*pre_pre_phi[9]*pre_pre_phi[11]*std::pow(pre_pre_phi[7],j);          
        }

        A[row_no][8+r-2] = temp*std::pow(pre_pre_phi[7],r-2);
        A[row_no][7+r+r-2] = temp*pre_pre_phi[9]*std::pow(pre_pre_phi[7],r-2);
        A[row_no][4+4*r+r-2] = temp*pre_pre_phi[11]*std::pow(pre_pre_phi[7],r-2);
        A[row_no][3+5*r+r-2] = temp*pre_pre_phi[11]*pre_pre_phi[9]*std::pow(pre_pre_phi[7],r-2);
      }

      if((row_no>=6+2*r && row_no<4+4*r) || (row_no>=2+6*r && r<8*r))
      {
        // x dir
        // R_y
        pre_pre_phi[10] = (B[2][4]*pre_pre_phi[2] - B[3][4]*pre_pre_phi[3])
        /(B[2][4]*pre_pre_phi[2] + B[3][4]*pre_pre_phi[3]);
        // R_z
        pre_pre_phi[11] = (B[4][2]*pre_pre_phi[4] - B[5][2]*pre_pre_phi[5])
        /(B[4][2]*pre_pre_phi[4] + B[5][2]*pre_pre_phi[5]); 
        AssertIsFinite(pre_pre_phi[10]);
        AssertIsFinite(pre_pre_phi[11]);      

        pre_pre_phi[6] = (dof_pt - center_pt)*Gamma_mid[0];
        pre_pre_phi[7] = B[2][4]*pre_pre_phi[2] - B[3][4]*pre_pre_phi[3];
        pre_pre_phi[8] = B[4][2]*pre_pre_phi[4] - B[5][2]*pre_pre_phi[5];

        double temp = pre_pre_phi[0]*pre_pre_phi[1];
        A[row_no][6+2*r] = temp;
        A[row_no][5+3*r] = temp*pre_pre_phi[7];
        A[row_no][2+6*r] = temp*pre_pre_phi[8];
        A[row_no][1+7*r] = temp*(alpha_x*pre_pre_phi[7]*pre_pre_phi[11]
          +beta_x*pre_pre_phi[8]*pre_pre_phi[10]
          +eta_x*pre_pre_phi[10]*pre_pre_phi[11]);

        for(unsigned int j=1; j<r-2; ++j)
        {
          A[row_no][6+2*r+j] = temp*std::pow(pre_pre_phi[6],j);
          A[row_no][5+3*r+j] = temp*pre_pre_phi[7]*std::pow(pre_pre_phi[6],j);
          A[row_no][2+6*r+j] = temp*pre_pre_phi[8]*std::pow(pre_pre_phi[6],j);
          A[row_no][1+7*r+j] = temp*pre_pre_phi[10]*pre_pre_phi[11]*std::pow(pre_pre_phi[6],j);          
        }

        A[row_no][6+2*r+r-2] = temp*std::pow(pre_pre_phi[6],r-2);
        A[row_no][5+3*r+r-2] = temp*pre_pre_phi[10]*std::pow(pre_pre_phi[6],r-2);
        A[row_no][2+6*r+r-2] = temp*pre_pre_phi[11]*std::pow(pre_pre_phi[6],r-2);
        A[row_no][1+7*r+r-2] = temp*pre_pre_phi[10]*pre_pre_phi[11]*std::pow(pre_pre_phi[6],r-2);
      }

      if((row_no>=8*r && row_no<12*r-4))
      {
        // z dir

        // R_x
        pre_pre_phi[9] = (B[0][2]*pre_pre_phi[0] - B[1][2]*pre_pre_phi[1])
        /(B[0][2]*pre_pre_phi[0] + B[1][2]*pre_pre_phi[1]);
        // R_y
        pre_pre_phi[10] = (B[2][0]*pre_pre_phi[2] - B[3][0]*pre_pre_phi[3])
        /(B[2][0]*pre_pre_phi[2] + B[3][0]*pre_pre_phi[3]); 
        AssertIsFinite(pre_pre_phi[9]);
        AssertIsFinite(pre_pre_phi[10]);      

        pre_pre_phi[8] = (dof_pt - center_pt)*Gamma_mid[2];
        pre_pre_phi[6] = B[0][2]*pre_pre_phi[0] - B[1][2]*pre_pre_phi[1];
        pre_pre_phi[7] = B[2][0]*pre_pre_phi[2] - B[3][0]*pre_pre_phi[3];

        double temp = pre_pre_phi[4]*pre_pre_phi[5];
        A[row_no][8*r] = temp;
        A[row_no][9*r-1] = temp*pre_pre_phi[6];
        A[row_no][10*r-2] = temp*pre_pre_phi[7];
        A[row_no][11*r-3] = temp*(alpha_z*pre_pre_phi[7]*pre_pre_phi[9]
          + beta_z*pre_pre_phi[6]*pre_pre_phi[10]
          + eta_z*pre_pre_phi[9]*pre_pre_phi[10]);


        for(unsigned int j=1; j<r-2; ++j)
        {
          A[row_no][8*r+j] = temp*std::pow(pre_pre_phi[8],j);
          A[row_no][9*r-1+j] = temp*pre_pre_phi[6]*std::pow(pre_pre_phi[8],j);
          A[row_no][10*r-2+j] = temp*pre_pre_phi[7]*std::pow(pre_pre_phi[8],j);
          A[row_no][11*r-3+j] = temp*pre_pre_phi[10]*pre_pre_phi[9]*std::pow(pre_pre_phi[8],j);
        }

        A[row_no][8*r+r-2] = temp*std::pow(pre_pre_phi[8],r-2);
        A[row_no][9*r-1+r-2] = temp*pre_pre_phi[9]*std::pow(pre_pre_phi[8],r-2);
        A[row_no][10*r-2+r-2] = temp*pre_pre_phi[10]*std::pow(pre_pre_phi[8],r-2);
        A[row_no][11*r-3+r-2] = temp*pre_pre_phi[9]*pre_pre_phi[10]*std::pow(pre_pre_phi[8],r-2);        
      }  
    }
  } //end (dim==3)

  // 3) invert A
  { // note for square matrix if AB=I, then BA=I
    // A.print_formatted(std::cout);
    LAPACKFullMatrix<double> ll_inverse(A.m(), A.n());
    ll_inverse = A;
    ll_inverse.invert();
    A = ll_inverse;
    // Assert(false,ExcInternalError());
  }

  // 4) set values and grads for each quadrature points

  std::vector<std::vector<Tensor<1,dim> > > temp_gradients;

  if(dim == 2 && flag_new)
  {
    temp_gradients.resize(2, std::vector<Tensor<1,dim> > (n_q_points));
    // std::cout<<"offset="<<offset<<", n_q_points="<<n_q_points<<std::endl;
    // std::cout<<fe_data.shape_gradients.size()<<"  "
    // <<fe_data.shape_gradients[0].size()<<std::endl;
    // Assert (flags & update_contravariant_transformation,
    //     typename FEValuesBase<dim>::ExcAccessToUninitializedField("update_contravariant_transformation"));
    // Assert (flags & update_covariant_transformation,
    //     typename FEValuesBase<dim>::ExcAccessToUninitializedField("update_covariant_transformation"));
    mapping.transform(make_slice(fe_data.shape_gradients[2*fe_degree+1],offset,n_q_points),
                      temp_gradients[0],
                      mapping_data, mapping_covariant);
    // Assert(false,ExcInternalError());
    mapping.transform(make_slice(fe_data.shape_gradients[4*fe_degree-1],offset,n_q_points),
                      temp_gradients[1],
                      mapping_data, mapping_covariant);    
  }

  for(unsigned int k=0; k<n_q_points; ++k)
  {
    Point<dim> quad_pt = data.quadrature_points[k];
    // Point<dim> quad_pt_hat = quadrature.point(k);

    Vector<double> pre_phi;
    Vector<double> pre_phi_dx;
    Vector<double> pre_phi_dy;
    Vector<double> pre_phi_dz;

    if(dim==2)
    {
      // set pre_pre_phi values
      pre_pre_phi[0] = (quad_pt - supp_pts[0])*Gamma[0]; //lambda_0
      pre_pre_phi[1] = (quad_pt - supp_pts[1])*Gamma[1]; //lambda_1
      pre_pre_phi[2] = (quad_pt - supp_pts[0])*Gamma[2]; //lambda_2
      pre_pre_phi[3] = (quad_pt - supp_pts[2])*Gamma[3]; //lambda_3
      Assert(pre_pre_phi[0]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[1]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[2]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[3]>=-1e-13, ExcMessage("dof point out of cell"));
      pre_pre_phi[4] = pre_pre_phi[0] - pre_pre_phi[1]; //lambda_V
      pre_pre_phi[5] = pre_pre_phi[2] - pre_pre_phi[3]; //lambda_H
      pre_pre_phi[6] = pre_pre_phi[4] / (sigma_v*pre_pre_phi[0] + eta_v*pre_pre_phi[1]); //R_V
      pre_pre_phi[7] = pre_pre_phi[5] / (sigma_h*pre_pre_phi[2] + eta_h*pre_pre_phi[3]); //R_H
      AssertIsFinite(pre_pre_phi[6]);
      AssertIsFinite(pre_pre_phi[7]);

      pre_pre_phi_grad[0] = Gamma[0];
      pre_pre_phi_grad[1] = Gamma[1];
      pre_pre_phi_grad[2] = Gamma[2];
      pre_pre_phi_grad[3] = Gamma[3];
      pre_pre_phi_grad[4] = pre_pre_phi_grad[0] - pre_pre_phi_grad[1];
      pre_pre_phi_grad[5] = pre_pre_phi_grad[2] - pre_pre_phi_grad[3];
      pre_pre_phi_grad[6] = 
      (pre_pre_phi_grad[4] - pre_pre_phi[6]*(sigma_v*pre_pre_phi_grad[0]+eta_v*pre_pre_phi_grad[1]))/
      (sigma_v*pre_pre_phi[0] + eta_v*pre_pre_phi[1]);
      pre_pre_phi_grad[7] = 
      (pre_pre_phi_grad[5] - pre_pre_phi[7]*(sigma_h*pre_pre_phi_grad[2]+eta_h*pre_pre_phi_grad[3]))/
      (sigma_h*pre_pre_phi[2] + eta_h*pre_pre_phi[3]);


      // for(unsigned int ii=0; ii<8; ++ii)
      //   std::cout<<pre_pre_phi_grad[ii]<<std::endl;

      pre_phi.reinit(4*fe_degree);
      pre_phi_dx.reinit(4*fe_degree);
      pre_phi_dy.reinit(4*fe_degree);

      pre_phi[0] = pre_pre_phi[1]*pre_pre_phi[3]; //lambda_1*lambda_3
      pre_phi[1] = pre_pre_phi[0]*pre_pre_phi[3]; //lambda_0*lambda_3
      pre_phi[2] = pre_pre_phi[1]*pre_pre_phi[2]; //lambda_1*lambda_2
      pre_phi[3] = pre_pre_phi[0]*pre_pre_phi[2]; //lambda_0*lambda_2     

      pre_phi_dx[0] = pre_pre_phi_grad[1][0]*pre_pre_phi[3]
      + pre_pre_phi[1]*pre_pre_phi_grad[3][0];
      pre_phi_dx[1] = pre_pre_phi_grad[0][0]*pre_pre_phi[3]
      + pre_pre_phi[0]*pre_pre_phi_grad[3][0];
      pre_phi_dx[2] = pre_pre_phi_grad[1][0]*pre_pre_phi[2]
      + pre_pre_phi[1]*pre_pre_phi_grad[2][0];
      pre_phi_dx[3] = pre_pre_phi_grad[0][0]*pre_pre_phi[2]
      + pre_pre_phi[0]*pre_pre_phi_grad[2][0];

      pre_phi_dy[0] = pre_pre_phi_grad[1][1]*pre_pre_phi[3]
      + pre_pre_phi[1]*pre_pre_phi_grad[3][1];
      pre_phi_dy[1] = pre_pre_phi_grad[0][1]*pre_pre_phi[3]
      + pre_pre_phi[0]*pre_pre_phi_grad[3][1];
      pre_phi_dy[2] = pre_pre_phi_grad[1][1]*pre_pre_phi[2]
      + pre_pre_phi[1]*pre_pre_phi_grad[2][1];
      pre_phi_dy[3] = pre_pre_phi_grad[0][1]*pre_pre_phi[2]
      + pre_pre_phi[0]*pre_pre_phi_grad[2][1];

      for(unsigned int j=0; j<fe_degree-1;++j)
      {
        // -----------------------------------------------------------------
        pre_phi[4+j] = pre_pre_phi[2]*pre_pre_phi[3]*std::pow(pre_pre_phi[5],j);

        pre_phi[fe_degree+3+j] = 
        pre_pre_phi[2]*pre_pre_phi[3]*((j==(fe_degree-2))?pre_pre_phi[6]:pre_pre_phi[4])
        *std::pow(pre_pre_phi[5],j);

        pre_phi[2*fe_degree+2+j] = pre_pre_phi[0]*pre_pre_phi[1]*std::pow(pre_pre_phi[4],j);

        pre_phi[3*fe_degree+1+j] = 
        pre_pre_phi[0]*pre_pre_phi[1]*((j==(fe_degree-2))?pre_pre_phi[7]:pre_pre_phi[5])
        *std::pow(pre_pre_phi[4],j);
        // -----------------------------------------------------------------
        pre_phi_dx[4+j] = 
        pre_pre_phi_grad[2][0]*pre_pre_phi[3]*std::pow(pre_pre_phi[5],j)
        + pre_pre_phi[2]*pre_pre_phi_grad[3][0]*std::pow(pre_pre_phi[5],j)
        + ((j==0)?0.:j*pre_pre_phi[2]*pre_pre_phi[3]*std::pow(pre_pre_phi[5],j-1)*pre_pre_phi_grad[5][0]);

        pre_phi_dx[fe_degree+3+j] = 
        pre_phi_dx[4+j]*((j==(fe_degree-2))?pre_pre_phi[6]:pre_pre_phi[4])
        + pre_phi[4+j]*((j==(fe_degree-2))?pre_pre_phi_grad[6][0]:pre_pre_phi_grad[4][0]);

        pre_phi_dx[2*fe_degree+2+j] = 
        pre_pre_phi_grad[0][0]*pre_pre_phi[1]*std::pow(pre_pre_phi[4],j)
        + pre_pre_phi[0]*pre_pre_phi_grad[1][0]*std::pow(pre_pre_phi[4],j)
        + ((j==0)?0.:j*pre_pre_phi[0]*pre_pre_phi[1]*std::pow(pre_pre_phi[4],j-1)*pre_pre_phi_grad[4][0]);

        pre_phi_dx[3*fe_degree+1+j] = 
        pre_phi_dx[2*fe_degree+2+j]*((j==(fe_degree-2))?pre_pre_phi[7]:pre_pre_phi[5])
        + pre_phi[2*fe_degree+2+j]*((j==(fe_degree-2))?pre_pre_phi_grad[7][0]:pre_pre_phi_grad[5][0]);
        // -------------------------------------------------------------------
        pre_phi_dy[4+j] = 
        pre_pre_phi_grad[2][1]*pre_pre_phi[3]*std::pow(pre_pre_phi[5],j)
        + pre_pre_phi[2]*pre_pre_phi_grad[3][1]*std::pow(pre_pre_phi[5],j)
        + ((j==0)?0.:j*pre_pre_phi[2]*pre_pre_phi[3]*std::pow(pre_pre_phi[5],j-1)*pre_pre_phi_grad[5][1]);

        pre_phi_dy[fe_degree+3+j] = 
        pre_phi_dy[4+j]*((j==(fe_degree-2))?pre_pre_phi[6]:pre_pre_phi[4])
        + pre_phi[4+j]*((j==(fe_degree-2))?pre_pre_phi_grad[6][1]:pre_pre_phi_grad[4][1]);

        pre_phi_dy[2*fe_degree+2+j] = 
        pre_pre_phi_grad[0][1]*pre_pre_phi[1]*std::pow(pre_pre_phi[4],j)
        + pre_pre_phi[0]*pre_pre_phi_grad[1][1]*std::pow(pre_pre_phi[4],j)
        + ((j==0)?0.:j*pre_pre_phi[0]*pre_pre_phi[1]*std::pow(pre_pre_phi[4],j-1)*pre_pre_phi_grad[4][1]);

        pre_phi_dy[3*fe_degree+1+j] = 
        pre_phi_dy[2*fe_degree+2+j]*((j==(fe_degree-2))?pre_pre_phi[7]:pre_pre_phi[5])
        + pre_phi[2*fe_degree+2+j]*((j==(fe_degree-2))?pre_pre_phi_grad[7][1]:pre_pre_phi_grad[5][1]);
      }

      // std::cout<<"*** pre_phi:"<<std::endl;
      // std::cout<<pre_phi<<std::endl;

      // std::cout<<"*** pre_phi_dx:"<<std::endl;
      // std::cout<<pre_phi_dx<<std::endl;

      // std::cout<<"*** pre_phi_dy:"<<std::endl;
      // std::cout<<pre_phi_dy<<std::endl;
      if(flag_new)
      {
        pre_phi[2*fe_degree+1] = fe_data.shape_values[2*fe_degree+1][k+offset]*(-4.0);
        pre_phi[4*fe_degree-1] = fe_data.shape_values[4*fe_degree-1][k+offset]*(-4.0);

        pre_phi_dx[2*fe_degree+1] = temp_gradients[0][k][0]*(-4.0);
        pre_phi_dy[2*fe_degree+1] = temp_gradients[0][k][1]*(-4.0);

        pre_phi_dx[4*fe_degree-1] = temp_gradients[1][k][0]*(-4.0);
        pre_phi_dy[4*fe_degree-1] = temp_gradients[1][k][1]*(-4.0);
      }
    } //end dim==2

    if(dim==3)
    {
      // set pre_pre_phi values
      pre_pre_phi[0] = (quad_pt - supp_pts[0])*Gamma[0]; //lambda_0
      pre_pre_phi[1] = (quad_pt - supp_pts[1])*Gamma[1]; //lambda_1
      pre_pre_phi[2] = (quad_pt - supp_pts[0])*Gamma[2]; //lambda_2
      pre_pre_phi[3] = (quad_pt - supp_pts[2])*Gamma[3]; //lambda_3
      pre_pre_phi[4] = (quad_pt - supp_pts[0])*Gamma[4]; //lambda_4
      pre_pre_phi[5] = (quad_pt - supp_pts[4])*Gamma[5]; //lambda_5
      Assert(pre_pre_phi[0]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[1]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[2]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[3]>=-1e-13, ExcMessage("dof point out of cell")); 
      Assert(pre_pre_phi[4]>=-1e-13, ExcMessage("dof point out of cell"));
      Assert(pre_pre_phi[5]>=-1e-13, ExcMessage("dof point out of cell"));          

      // set pre_pre_phi_grad values
      pre_pre_phi_grad[0] = Gamma[0];
      pre_pre_phi_grad[1] = Gamma[1];
      pre_pre_phi_grad[2] = Gamma[2];
      pre_pre_phi_grad[3] = Gamma[3];
      pre_pre_phi_grad[4] = Gamma[4];
      pre_pre_phi_grad[5] = Gamma[5];

      pre_phi.reinit(12*r-4);
      pre_phi_dx.reinit(12*r-4);
      pre_phi_dy.reinit(12*r-4);
      pre_phi_dz.reinit(12*r-4);

      if(r>3)
      {
        pre_phi_int.resize(6);
        pre_phi_int_grad.resize(6);
      }

      pre_phi[0] = pre_pre_phi[1]*pre_pre_phi[3]*pre_pre_phi[5];
      pre_phi[1] = pre_pre_phi[0]*pre_pre_phi[3]*pre_pre_phi[5];
      pre_phi[2] = pre_pre_phi[1]*pre_pre_phi[2]*pre_pre_phi[5];
      pre_phi[3] = pre_pre_phi[0]*pre_pre_phi[2]*pre_pre_phi[5];
      pre_phi[4] = pre_pre_phi[1]*pre_pre_phi[3]*pre_pre_phi[4];
      pre_phi[5] = pre_pre_phi[0]*pre_pre_phi[3]*pre_pre_phi[4];
      pre_phi[6] = pre_pre_phi[1]*pre_pre_phi[2]*pre_pre_phi[4];
      pre_phi[7] = pre_pre_phi[0]*pre_pre_phi[2]*pre_pre_phi[4];      

      Tensor<1,dim> ttemp;

      ttemp = pre_pre_phi_grad[1]*pre_pre_phi[3]*pre_pre_phi[5]
      + pre_pre_phi[1]*pre_pre_phi_grad[3]*pre_pre_phi[5]
      + pre_pre_phi[1]*pre_pre_phi[3]*pre_pre_phi_grad[5];
      pre_phi_dx[0] = ttemp[0]; pre_phi_dy[0] = ttemp[1]; pre_phi_dz[0] = ttemp[2];

      ttemp = pre_pre_phi_grad[0]*pre_pre_phi[3]*pre_pre_phi[5]
      + pre_pre_phi[0]*pre_pre_phi_grad[3]*pre_pre_phi[5]
      + pre_pre_phi[0]*pre_pre_phi[3]*pre_pre_phi_grad[5];
      pre_phi_dx[1] = ttemp[0]; pre_phi_dy[1] = ttemp[1]; pre_phi_dz[1] = ttemp[2];

      ttemp = pre_pre_phi_grad[1]*pre_pre_phi[2]*pre_pre_phi[5]
      + pre_pre_phi[1]*pre_pre_phi_grad[2]*pre_pre_phi[5]
      + pre_pre_phi[1]*pre_pre_phi[2]*pre_pre_phi_grad[5];
      pre_phi_dx[2] = ttemp[0]; pre_phi_dy[2] = ttemp[1]; pre_phi_dz[2] = ttemp[2];

      ttemp = pre_pre_phi_grad[0]*pre_pre_phi[2]*pre_pre_phi[5]
      + pre_pre_phi[0]*pre_pre_phi_grad[2]*pre_pre_phi[5]
      + pre_pre_phi[0]*pre_pre_phi[2]*pre_pre_phi_grad[5];
      pre_phi_dx[3] = ttemp[0]; pre_phi_dy[3] = ttemp[1]; pre_phi_dz[3] = ttemp[2];

      ttemp = pre_pre_phi_grad[1]*pre_pre_phi[3]*pre_pre_phi[4]
      + pre_pre_phi[1]*pre_pre_phi_grad[3]*pre_pre_phi[4]
      + pre_pre_phi[1]*pre_pre_phi[3]*pre_pre_phi_grad[4];
      pre_phi_dx[4] = ttemp[0]; pre_phi_dy[4] = ttemp[1]; pre_phi_dz[4] = ttemp[2];      

      ttemp = pre_pre_phi_grad[0]*pre_pre_phi[3]*pre_pre_phi[4]
      + pre_pre_phi[0]*pre_pre_phi_grad[3]*pre_pre_phi[4]
      + pre_pre_phi[0]*pre_pre_phi[3]*pre_pre_phi_grad[4];
      pre_phi_dx[5] = ttemp[0]; pre_phi_dy[5] = ttemp[1]; pre_phi_dz[5] = ttemp[2];    

      ttemp = pre_pre_phi_grad[1]*pre_pre_phi[2]*pre_pre_phi[4]
      + pre_pre_phi[1]*pre_pre_phi_grad[2]*pre_pre_phi[4]
      + pre_pre_phi[1]*pre_pre_phi[2]*pre_pre_phi_grad[4];
      pre_phi_dx[6] = ttemp[0]; pre_phi_dy[6] = ttemp[1]; pre_phi_dz[6] = ttemp[2];    

      ttemp = pre_pre_phi_grad[0]*pre_pre_phi[2]*pre_pre_phi[4]
      + pre_pre_phi[0]*pre_pre_phi_grad[2]*pre_pre_phi[4]
      + pre_pre_phi[0]*pre_pre_phi[2]*pre_pre_phi_grad[4];   
      pre_phi_dx[7] = ttemp[0]; pre_phi_dy[7] = ttemp[1]; pre_phi_dz[7] = ttemp[2];   

 
      // y dir

      Tensor<1,dim> ytemp;
      double yvalue;

      pre_pre_phi[7] = (quad_pt - center_pt)*Gamma_mid[1];
      pre_pre_phi[6] = B[0][4]*pre_pre_phi[0] - B[1][4]*pre_pre_phi[1];
      pre_pre_phi[8] = B[4][0]*pre_pre_phi[4] - B[5][0]*pre_pre_phi[5];

      pre_pre_phi_grad[7] = Gamma_mid[1];
      pre_pre_phi_grad[6] = B[0][4]*pre_pre_phi_grad[0] - B[1][4]*pre_pre_phi_grad[1];
      pre_pre_phi_grad[8] = B[4][0]*pre_pre_phi_grad[4] - B[5][0]*pre_pre_phi_grad[5];

      // R_x
      pre_pre_phi[9] = (B[0][4]*pre_pre_phi[0] - B[1][4]*pre_pre_phi[1])
      /(B[0][4]*pre_pre_phi[0] + B[1][4]*pre_pre_phi[1]);
      // R_z
      pre_pre_phi[11] = (B[4][0]*pre_pre_phi[4] - B[5][0]*pre_pre_phi[5])
      /(B[4][0]*pre_pre_phi[4] + B[5][0]*pre_pre_phi[5]); 
      AssertIsFinite(pre_pre_phi[9]);
      AssertIsFinite(pre_pre_phi[11]);        

      pre_pre_phi_grad[9] = 
      (pre_pre_phi_grad[6] - 
        pre_pre_phi[9]*(B[0][4]*pre_pre_phi_grad[0] + B[1][4]*pre_pre_phi_grad[1])
        )/(B[0][4]*pre_pre_phi[0] + B[1][4]*pre_pre_phi[1]);
     
      pre_pre_phi_grad[11] =
      (pre_pre_phi_grad[8] -
        pre_pre_phi[11]*(B[4][0]*pre_pre_phi_grad[4] + B[5][0]*pre_pre_phi_grad[5])
        )/(B[4][0]*pre_pre_phi[4] + B[5][0]*pre_pre_phi[5]);
      
      double yBubble = pre_pre_phi[2]*pre_pre_phi[3];
      Tensor<1,dim> dyBubble = 
      pre_pre_phi_grad[2]*pre_pre_phi[3] + pre_pre_phi[2]*pre_pre_phi_grad[3];

      pre_phi[8] = yBubble;
      pre_phi[7+r] = yBubble*pre_pre_phi[6];
      pre_phi[4+4*r] = yBubble*pre_pre_phi[8];
      pre_phi[3+5*r] = yBubble*(alpha_y*pre_pre_phi[8]*pre_pre_phi[9]
        + beta_y*pre_pre_phi[6]*pre_pre_phi[11]
        + eta_y*pre_pre_phi[9]*pre_pre_phi[11]);
      

      pre_phi_dx[8] = dyBubble[0]; 
      pre_phi_dy[8] = dyBubble[1]; 
      pre_phi_dz[8] = dyBubble[2];
      pre_phi_dx[7+r] = dyBubble[0]*pre_pre_phi[6] + yBubble * pre_pre_phi_grad[6][0];
      pre_phi_dy[7+r] = dyBubble[1]*pre_pre_phi[6] + yBubble * pre_pre_phi_grad[6][1];
      pre_phi_dz[7+r] = dyBubble[2]*pre_pre_phi[6] + yBubble * pre_pre_phi_grad[6][2];
      pre_phi_dx[4+4*r] = dyBubble[0]*pre_pre_phi[8] + yBubble * pre_pre_phi_grad[8][0];
      pre_phi_dy[4+4*r] = dyBubble[1]*pre_pre_phi[8] + yBubble * pre_pre_phi_grad[8][1];
      pre_phi_dz[4+4*r] = dyBubble[2]*pre_pre_phi[8] + yBubble * pre_pre_phi_grad[8][2];

      yvalue = alpha_y*pre_pre_phi[8]*pre_pre_phi[9]
      +beta_y*pre_pre_phi[6]*pre_pre_phi[11]
      +eta_y*pre_pre_phi[9]*pre_pre_phi[11];

      ytemp = alpha_y*pre_pre_phi_grad[8]*pre_pre_phi[9]
      + alpha_y*pre_pre_phi[8]*pre_pre_phi_grad[9]
      + beta_y*pre_pre_phi_grad[6]*pre_pre_phi[11]
      + beta_y*pre_pre_phi[6]*pre_pre_phi_grad[11]
      + eta_y*pre_pre_phi_grad[9]*pre_pre_phi[11]
      + eta_y*pre_pre_phi[9]*pre_pre_phi_grad[11];

      pre_phi_dx[3+5*r] = dyBubble[0]*yvalue + yBubble*ytemp[0];
      pre_phi_dy[3+5*r] = dyBubble[1]*yvalue + yBubble*ytemp[1];
      pre_phi_dz[3+5*r] = dyBubble[2]*yvalue + yBubble*ytemp[2];

      for(unsigned int j=1; j<r-2; ++j)
      {
        pre_phi[8+j] = yBubble*std::pow(pre_pre_phi[7],j);
        pre_phi[7+r+j] = yBubble*pre_pre_phi[6]*std::pow(pre_pre_phi[7],j);
        pre_phi[4+4*r+j] = yBubble*pre_pre_phi[8]*std::pow(pre_pre_phi[7],j);
        pre_phi[3+5*r+j] = yBubble*pre_pre_phi[11]*pre_pre_phi[9]*std::pow(pre_pre_phi[7],j);     

        ytemp = j*std::pow(pre_pre_phi[7],j-1)*pre_pre_phi_grad[7];
        yvalue = std::pow(pre_pre_phi[7],j);
        pre_phi_dx[8+j] = dyBubble[0]*yvalue+yBubble*ytemp[0];
        pre_phi_dy[8+j] = dyBubble[1]*yvalue+yBubble*ytemp[1];
        pre_phi_dz[8+j] = dyBubble[2]*yvalue+yBubble*ytemp[2];                

        ytemp = pre_pre_phi[6] * j*std::pow(pre_pre_phi[7],j-1)*pre_pre_phi_grad[7]
        + pre_pre_phi_grad[6] * std::pow(pre_pre_phi[7],j);
        yvalue = pre_pre_phi[6]*std::pow(pre_pre_phi[7],j);
        pre_phi_dx[7+r+j] = dyBubble[0]*yvalue+yBubble*ytemp[0];
        pre_phi_dy[7+r+j] = dyBubble[1]*yvalue+yBubble*ytemp[1];
        pre_phi_dz[7+r+j] = dyBubble[2]*yvalue+yBubble*ytemp[2];

        ytemp = pre_pre_phi[8] * j*std::pow(pre_pre_phi[7],j-1)*pre_pre_phi_grad[7]
        + pre_pre_phi_grad[8] * std::pow(pre_pre_phi[7],j);
        yvalue = pre_pre_phi[8]*std::pow(pre_pre_phi[7],j);
        pre_phi_dx[4+4*r+j] = dyBubble[0]*yvalue+yBubble*ytemp[0];
        pre_phi_dy[4+4*r+j] = dyBubble[1]*yvalue+yBubble*ytemp[1];
        pre_phi_dz[4+4*r+j] = dyBubble[2]*yvalue+yBubble*ytemp[2];                

        ytemp = pre_pre_phi[11]*pre_pre_phi[9]*j*std::pow(pre_pre_phi[7],j-1)*pre_pre_phi_grad[7]
        + pre_pre_phi_grad[11]*pre_pre_phi[9]*std::pow(pre_pre_phi[7],j)
        + pre_pre_phi[11]*pre_pre_phi_grad[9]*std::pow(pre_pre_phi[7],j); 
        yvalue = pre_pre_phi[11]*pre_pre_phi[9]*std::pow(pre_pre_phi[7],j);
        pre_phi_dx[3+5*r+j] = dyBubble[0]*yvalue+yBubble*ytemp[0];
        pre_phi_dy[3+5*r+j] = dyBubble[1]*yvalue+yBubble*ytemp[1];
        pre_phi_dz[3+5*r+j] = dyBubble[2]*yvalue+yBubble*ytemp[2];                
      }

      pre_phi[8+r-2] = yBubble*std::pow(pre_pre_phi[7],r-2);
      pre_phi[7+r+r-2] = yBubble*pre_pre_phi[9]*std::pow(pre_pre_phi[7],r-2);
      pre_phi[4+4*r+r-2] = yBubble*pre_pre_phi[11]*std::pow(pre_pre_phi[7],r-2);
      pre_phi[3+5*r+r-2] = yBubble*pre_pre_phi[9]*pre_pre_phi[11]*std::pow(pre_pre_phi[7],r-2);

      ytemp = (r-2)*std::pow(pre_pre_phi[7],r-3)*pre_pre_phi_grad[7];
      yvalue = std::pow(pre_pre_phi[7],r-2);
      pre_phi_dx[8+r-2] = dyBubble[0]*yvalue + yBubble*ytemp[0];
      pre_phi_dy[8+r-2] = dyBubble[1]*yvalue + yBubble*ytemp[1];
      pre_phi_dz[8+r-2] = dyBubble[2]*yvalue + yBubble*ytemp[2];            

      yvalue = pre_pre_phi[9]*std::pow(pre_pre_phi[7],r-2);
      ytemp = pre_pre_phi[9]*(r-2)*std::pow(pre_pre_phi[7],r-3)*pre_pre_phi_grad[7]
      + pre_pre_phi_grad[9]*std::pow(pre_pre_phi[7],r-2);
      pre_phi_dx[7+r+r-2] = dyBubble[0]*yvalue + yBubble*ytemp[0];
      pre_phi_dy[7+r+r-2] = dyBubble[1]*yvalue + yBubble*ytemp[1];
      pre_phi_dz[7+r+r-2] = dyBubble[2]*yvalue + yBubble*ytemp[2];

      yvalue = pre_pre_phi[11]*std::pow(pre_pre_phi[7],r-2);
      ytemp = pre_pre_phi[11]*(r-2)*std::pow(pre_pre_phi[7],r-3)*pre_pre_phi_grad[7]
      + pre_pre_phi_grad[11]*std::pow(pre_pre_phi[7],r-2);
      pre_phi_dx[4+4*r+r-2] = dyBubble[0]*yvalue + yBubble*ytemp[0];
      pre_phi_dy[4+4*r+r-2] = dyBubble[1]*yvalue + yBubble*ytemp[1];
      pre_phi_dz[4+4*r+r-2] = dyBubble[2]*yvalue + yBubble*ytemp[2];      

      yvalue = pre_pre_phi[9]*pre_pre_phi[11]*std::pow(pre_pre_phi[7],r-2);
      ytemp = pre_pre_phi[9]*pre_pre_phi[11]*(r-2)*std::pow(pre_pre_phi[7],r-3)*pre_pre_phi_grad[7]
      + pre_pre_phi_grad[9]*pre_pre_phi[11]*std::pow(pre_pre_phi[7],r-2)
      + pre_pre_phi[9]*pre_pre_phi_grad[11]*std::pow(pre_pre_phi[7],r-2);
      pre_phi_dx[3+5*r+r-2] = dyBubble[0]*yvalue + yBubble*ytemp[0];
      pre_phi_dy[3+5*r+r-2] = dyBubble[1]*yvalue + yBubble*ytemp[1];
      pre_phi_dz[3+5*r+r-2] = dyBubble[2]*yvalue + yBubble*ytemp[2];

      if(r>3)
      {
        pre_phi_int[0] = (1. - pre_pre_phi[9])
        * B[2][0]*pre_pre_phi[2]
        * B[3][0]*pre_pre_phi[3]
        * B[4][0]*pre_pre_phi[4]
        * B[5][0]*pre_pre_phi[5];

        pre_phi_int[1] = (1. + pre_pre_phi[9])
        * B[2][1]*pre_pre_phi[2]
        * B[3][1]*pre_pre_phi[3]
        * B[4][1]*pre_pre_phi[4]
        * B[5][1]*pre_pre_phi[5];

        pre_phi_int[4] = (1. - pre_pre_phi[11])
        * B[0][4]*pre_pre_phi[0]
        * B[1][4]*pre_pre_phi[1]
        * B[2][4]*pre_pre_phi[2]
        * B[3][4]*pre_pre_phi[3];

        pre_phi_int[5] = (1. + pre_pre_phi[11])
        * B[0][5]*pre_pre_phi[0]
        * B[1][5]*pre_pre_phi[1]
        * B[2][5]*pre_pre_phi[2]
        * B[3][5]*pre_pre_phi[3];

        pre_phi_int_grad[0] = (-pre_pre_phi_grad[9])
        * B[2][0]*pre_pre_phi[2] * B[3][0]*pre_pre_phi[3]
        * B[4][0]*pre_pre_phi[4] * B[5][0]*pre_pre_phi[5]
        + (1. - pre_pre_phi[9])
        * (B[2][0]*pre_pre_phi_grad[2] * B[3][0]*pre_pre_phi[3]+B[2][0]*pre_pre_phi[2] * B[3][0]*pre_pre_phi_grad[3])
        * B[4][0]*pre_pre_phi[4] * B[5][0]*pre_pre_phi[5]
        + (1. - pre_pre_phi[9])
        * B[2][0]*pre_pre_phi[2] * B[3][0]*pre_pre_phi[3]
        * (B[4][0]*pre_pre_phi_grad[4] * B[5][0]*pre_pre_phi[5]+B[4][0]*pre_pre_phi[4] * B[5][0]*pre_pre_phi_grad[5]);

        pre_phi_int_grad[1] = (pre_pre_phi_grad[9])
        * B[2][1]*pre_pre_phi[2] * B[3][1]*pre_pre_phi[3]
        * B[4][1]*pre_pre_phi[4] * B[5][1]*pre_pre_phi[5]
        + (1. + pre_pre_phi[9])
        * (B[2][1]*pre_pre_phi_grad[2] * B[3][1]*pre_pre_phi[3]+B[2][1]*pre_pre_phi[2] * B[3][1]*pre_pre_phi_grad[3])
        * B[4][1]*pre_pre_phi[4] * B[5][1]*pre_pre_phi[5]
        + (1. + pre_pre_phi[9])
        * B[2][1]*pre_pre_phi[2] * B[3][1]*pre_pre_phi[3]
        * (B[4][1]*pre_pre_phi_grad[4] * B[5][1]*pre_pre_phi[5]+B[4][1]*pre_pre_phi[4] * B[5][1]*pre_pre_phi_grad[5]);

        pre_phi_int_grad[4] = (-pre_pre_phi_grad[11])
        * B[0][4]*pre_pre_phi[0] * B[1][4]*pre_pre_phi[1]
        * B[2][4]*pre_pre_phi[2] * B[3][4]*pre_pre_phi[3]
        + (1. - pre_pre_phi[11])
        * (B[0][4]*pre_pre_phi_grad[0] * B[1][4]*pre_pre_phi[1]+B[0][4]*pre_pre_phi[0] * B[1][4]*pre_pre_phi_grad[1])
        * B[2][4]*pre_pre_phi[2] * B[3][4]*pre_pre_phi[3]
        + (1. - pre_pre_phi[11])
        * B[0][4]*pre_pre_phi[0] * B[1][4]*pre_pre_phi[1]
        * (B[2][4]*pre_pre_phi_grad[2] * B[3][4]*pre_pre_phi[3]+B[2][4]*pre_pre_phi[2] * B[3][4]*pre_pre_phi_grad[3]);

        pre_phi_int_grad[5] = (pre_pre_phi_grad[11])
        * B[0][5]*pre_pre_phi[0] * B[1][5]*pre_pre_phi[1]
        * B[2][5]*pre_pre_phi[2] * B[3][5]*pre_pre_phi[3]
        +  (1. + pre_pre_phi[11])
        * (B[0][5]*pre_pre_phi_grad[0] * B[1][5]*pre_pre_phi[1]+B[0][5]*pre_pre_phi[0] * B[1][5]*pre_pre_phi_grad[1])
        * B[2][5]*pre_pre_phi[2] * B[3][5]*pre_pre_phi[3]
        +  (1. + pre_pre_phi[11])
        * B[0][5]*pre_pre_phi[0] * B[1][5]*pre_pre_phi[1]
        * (B[2][5]*pre_pre_phi_grad[2] * B[3][5]*pre_pre_phi[3]+B[2][5]*pre_pre_phi[2] * B[3][5]*pre_pre_phi_grad[3]);
      }
      // x dir

      pre_pre_phi[6] = (quad_pt - center_pt)*Gamma_mid[0];
      pre_pre_phi[7] = B[2][4]*pre_pre_phi[2] - B[3][4]*pre_pre_phi[3];
      pre_pre_phi[8] = B[4][2]*pre_pre_phi[4] - B[5][2]*pre_pre_phi[5];

      pre_pre_phi_grad[6] = Gamma_mid[0];
      pre_pre_phi_grad[7] = B[2][4]*pre_pre_phi_grad[2] - B[3][4]*pre_pre_phi_grad[3];
      pre_pre_phi_grad[8] = B[4][2]*pre_pre_phi_grad[4] - B[5][2]*pre_pre_phi_grad[5];

      // R_y
      pre_pre_phi[10] = (B[2][4]*pre_pre_phi[2] - B[3][4]*pre_pre_phi[3])
      /(B[2][4]*pre_pre_phi[2] + B[3][4]*pre_pre_phi[3]);
      // R_z
      pre_pre_phi[11] = (B[4][2]*pre_pre_phi[4] - B[5][2]*pre_pre_phi[5])
      /(B[4][2]*pre_pre_phi[4] + B[5][2]*pre_pre_phi[5]); 
      AssertIsFinite(pre_pre_phi[10]);
      AssertIsFinite(pre_pre_phi[11]);         
      
      pre_pre_phi_grad[10] = 
      (pre_pre_phi_grad[7] - 
        pre_pre_phi[10]*(B[2][4]*pre_pre_phi_grad[2] + B[3][4]*pre_pre_phi_grad[3])
        )/(B[2][4]*pre_pre_phi[2] + B[3][4]*pre_pre_phi[3]);
   
      pre_pre_phi_grad[11] = 
      (pre_pre_phi_grad[8] -
        pre_pre_phi[11]*(B[4][2]*pre_pre_phi_grad[4] + B[5][2]*pre_pre_phi_grad[5])
        )/(B[4][2]*pre_pre_phi[4] + B[5][2]*pre_pre_phi[5]);
    
      double xBubble = pre_pre_phi[0]*pre_pre_phi[1];
      Tensor<1,dim> dxBubble = 
      pre_pre_phi_grad[0]*pre_pre_phi[1] + pre_pre_phi[0]*pre_pre_phi_grad[1];

      Tensor<1,dim> xtemp;
      double xvalue;

      pre_phi[6+2*r] = xBubble;
      pre_phi[5+3*r] = xBubble*pre_pre_phi[7];
      pre_phi[2+6*r] = xBubble*pre_pre_phi[8];
      pre_phi[1+7*r] = xBubble*(alpha_x*pre_pre_phi[7]*pre_pre_phi[11]
        + beta_x*pre_pre_phi[8]*pre_pre_phi[10]
        + eta_x*pre_pre_phi[10]*pre_pre_phi[11]);


      pre_phi_dx[6+2*r] = dxBubble[0];
      pre_phi_dy[6+2*r] = dxBubble[1];
      pre_phi_dz[6+2*r] = dxBubble[2];
      pre_phi_dx[5+3*r] = dxBubble[0]*pre_pre_phi[7]+xBubble*pre_pre_phi_grad[7][0]; 
      pre_phi_dy[5+3*r] = dxBubble[1]*pre_pre_phi[7]+xBubble*pre_pre_phi_grad[7][1];
      pre_phi_dz[5+3*r] = dxBubble[2]*pre_pre_phi[7]+xBubble*pre_pre_phi_grad[7][2];
      pre_phi_dx[2+6*r] = dxBubble[0]*pre_pre_phi[8]+xBubble*pre_pre_phi_grad[8][0];     
      pre_phi_dy[2+6*r] = dxBubble[1]*pre_pre_phi[8]+xBubble*pre_pre_phi_grad[8][1];     
      pre_phi_dz[2+6*r] = dxBubble[2]*pre_pre_phi[8]+xBubble*pre_pre_phi_grad[8][2];   

      xvalue = alpha_x*pre_pre_phi[7]*pre_pre_phi[11]
      +beta_x*pre_pre_phi[8]*pre_pre_phi[10]
      +eta_x*pre_pre_phi[10]*pre_pre_phi[11];

      xtemp = alpha_x*pre_pre_phi_grad[7]*pre_pre_phi[11]
      + alpha_x*pre_pre_phi[7]*pre_pre_phi_grad[11]
      + beta_x*pre_pre_phi_grad[8]*pre_pre_phi[10]
      + beta_x*pre_pre_phi[8]*pre_pre_phi_grad[10]
      + eta_x*pre_pre_phi_grad[10]*pre_pre_phi[11]
      + eta_x*pre_pre_phi[10]*pre_pre_phi_grad[11];

      pre_phi_dx[1+7*r] = dxBubble[0]*xvalue + xBubble*xtemp[0];
      pre_phi_dy[1+7*r] = dxBubble[1]*xvalue + xBubble*xtemp[1];
      pre_phi_dz[1+7*r] = dxBubble[2]*xvalue + xBubble*xtemp[2];

      for(unsigned int j=1; j<r-2; ++j)
      {
        pre_phi[6+2*r+j] = xBubble*std::pow(pre_pre_phi[6],j);
        pre_phi[5+3*r+j] = xBubble*pre_pre_phi[7]*std::pow(pre_pre_phi[6],j);
        pre_phi[2+6*r+j] = xBubble*pre_pre_phi[8]*std::pow(pre_pre_phi[6],j);
        pre_phi[1+7*r+j] = xBubble*pre_pre_phi[10]*pre_pre_phi[11]*std::pow(pre_pre_phi[6],j);

        xvalue = std::pow(pre_pre_phi[6],j);
        xtemp = j*std::pow(pre_pre_phi[6],j-1)*pre_pre_phi_grad[6];
        pre_phi_dx[6+2*r+j] = dxBubble[0]*xvalue+xBubble*xtemp[0];
        pre_phi_dy[6+2*r+j] = dxBubble[1]*xvalue+xBubble*xtemp[1];
        pre_phi_dz[6+2*r+j] = dxBubble[2]*xvalue+xBubble*xtemp[2];

        xvalue = pre_pre_phi[7]*std::pow(pre_pre_phi[6],j);
        xtemp = pre_pre_phi[7]*j*std::pow(pre_pre_phi[6],j-1)*pre_pre_phi_grad[6]
        + pre_pre_phi_grad[7]*std::pow(pre_pre_phi[6],j);
        pre_phi_dx[5+3*r+j] = dxBubble[0]*xvalue+xBubble*xtemp[0];
        pre_phi_dy[5+3*r+j] = dxBubble[1]*xvalue+xBubble*xtemp[1];
        pre_phi_dz[5+3*r+j] = dxBubble[2]*xvalue+xBubble*xtemp[2];                

        xvalue = pre_pre_phi[8]*std::pow(pre_pre_phi[6],j);
        xtemp = pre_pre_phi[8]*j*std::pow(pre_pre_phi[6],j-1)*pre_pre_phi_grad[6]
        + pre_pre_phi_grad[8]*std::pow(pre_pre_phi[6],j);
        pre_phi_dx[2+6*r+j] = dxBubble[0]*xvalue+xBubble*xtemp[0];
        pre_phi_dy[2+6*r+j] = dxBubble[1]*xvalue+xBubble*xtemp[1];        
        pre_phi_dz[2+6*r+j] = dxBubble[2]*xvalue+xBubble*xtemp[2];

        xvalue = pre_pre_phi[10]*pre_pre_phi[11]*std::pow(pre_pre_phi[6],j);
        xtemp = pre_pre_phi[10]*pre_pre_phi[11]*j*std::pow(pre_pre_phi[6],j-1)*pre_pre_phi_grad[6]
        + pre_pre_phi_grad[10]*pre_pre_phi[11]*std::pow(pre_pre_phi[6],j)
        + pre_pre_phi[10]*pre_pre_phi_grad[11]*std::pow(pre_pre_phi[6],j);
        pre_phi_dx[1+7*r+j] = dxBubble[0]*xvalue+xBubble*xtemp[0];
        pre_phi_dy[1+7*r+j] = dxBubble[1]*xvalue+xBubble*xtemp[1];
        pre_phi_dz[1+7*r+j] = dxBubble[2]*xvalue+xBubble*xtemp[2];
      }

      pre_phi[6+2*r+r-2] = xBubble*std::pow(pre_pre_phi[6],r-2);
      pre_phi[5+3*r+r-2] = xBubble*pre_pre_phi[10]*std::pow(pre_pre_phi[6],r-2);
      pre_phi[2+6*r+r-2] = xBubble*pre_pre_phi[11]*std::pow(pre_pre_phi[6],r-2);
      pre_phi[1+7*r+r-2] = xBubble*pre_pre_phi[10]*pre_pre_phi[11]*std::pow(pre_pre_phi[6],r-2);

      xvalue = std::pow(pre_pre_phi[6],r-2);
      xtemp = (r-2)*std::pow(pre_pre_phi[6],r-3)*pre_pre_phi_grad[6];
      pre_phi_dx[6+2*r+r-2] = dxBubble[0]*xvalue+xBubble*xtemp[0];
      pre_phi_dy[6+2*r+r-2] = dxBubble[1]*xvalue+xBubble*xtemp[1];
      pre_phi_dz[6+2*r+r-2] = dxBubble[2]*xvalue+xBubble*xtemp[2];            

      xvalue = pre_pre_phi[10]*std::pow(pre_pre_phi[6],r-2);
      xtemp = pre_pre_phi[10]*(r-2)*std::pow(pre_pre_phi[6],r-3)*pre_pre_phi_grad[6]
      + pre_pre_phi_grad[10]*std::pow(pre_pre_phi[6],r-2);
      pre_phi_dx[5+3*r+r-2] = dxBubble[0]*xvalue+xBubble*xtemp[0];
      pre_phi_dy[5+3*r+r-2] = dxBubble[1]*xvalue+xBubble*xtemp[1];
      pre_phi_dz[5+3*r+r-2] = dxBubble[2]*xvalue+xBubble*xtemp[2];

      xvalue = pre_pre_phi[11]*std::pow(pre_pre_phi[6],r-2);
      xtemp = pre_pre_phi[11]*(r-2)*std::pow(pre_pre_phi[6],r-3)*pre_pre_phi_grad[6]
      + pre_pre_phi_grad[11]*std::pow(pre_pre_phi[6],r-2);
      pre_phi_dx[2+6*r+r-2] = dxBubble[0]*xvalue+xBubble*xtemp[0];
      pre_phi_dy[2+6*r+r-2] = dxBubble[1]*xvalue+xBubble*xtemp[1];
      pre_phi_dz[2+6*r+r-2] = dxBubble[2]*xvalue+xBubble*xtemp[2];

      xvalue = pre_pre_phi[10]*pre_pre_phi[11]*std::pow(pre_pre_phi[6],r-2);
      xtemp = pre_pre_phi[10]*pre_pre_phi[11]*(r-2)*std::pow(pre_pre_phi[6],r-3)*pre_pre_phi_grad[6]
      + pre_pre_phi_grad[10]*pre_pre_phi[11]*std::pow(pre_pre_phi[6],r-2)
      + pre_pre_phi[10]*pre_pre_phi_grad[11]*std::pow(pre_pre_phi[6],r-2);
      pre_phi_dx[1+7*r+r-2] = dxBubble[0]*xvalue+xBubble*xtemp[0];
      pre_phi_dy[1+7*r+r-2] = dxBubble[1]*xvalue+xBubble*xtemp[1];
      pre_phi_dz[1+7*r+r-2] = dxBubble[2]*xvalue+xBubble*xtemp[2];

      if(r>3)
      {
        pre_phi_int[2] = (1. - pre_pre_phi[10])
        * B[0][2]*pre_pre_phi[0]
        * B[1][2]*pre_pre_phi[1]
        * B[4][2]*pre_pre_phi[4]
        * B[5][2]*pre_pre_phi[5];

        pre_phi_int[3] = (1. + pre_pre_phi[10])
        * B[0][3]*pre_pre_phi[0]
        * B[1][3]*pre_pre_phi[1]
        * B[4][3]*pre_pre_phi[4]
        * B[5][3]*pre_pre_phi[5];

        pre_phi_int_grad[2] = (-pre_pre_phi_grad[10])
        * B[0][2]*pre_pre_phi[0] * B[1][2]*pre_pre_phi[1]
        * B[4][2]*pre_pre_phi[4] * B[5][2]*pre_pre_phi[5]
        + (1. - pre_pre_phi[10])
        * (B[0][2]*pre_pre_phi_grad[0] * B[1][2]*pre_pre_phi[1]+B[0][2]*pre_pre_phi[0] * B[1][2]*pre_pre_phi_grad[1])
        * B[4][2]*pre_pre_phi[4] * B[5][2]*pre_pre_phi[5]
        + (1. - pre_pre_phi[10])
        * B[0][2]*pre_pre_phi[0] * B[1][2]*pre_pre_phi[1]
        * (B[4][2]*pre_pre_phi_grad[4] * B[5][2]*pre_pre_phi[5]+B[4][2]*pre_pre_phi[4] * B[5][2]*pre_pre_phi_grad[5]);

        pre_phi_int_grad[3] = (pre_pre_phi_grad[10])
        * B[0][3]*pre_pre_phi[0] * B[1][3]*pre_pre_phi[1]
        * B[4][3]*pre_pre_phi[4] * B[5][3]*pre_pre_phi[5]
        + (1. + pre_pre_phi[10])
        * (B[0][3]*pre_pre_phi_grad[0] * B[1][3]*pre_pre_phi[1]+B[0][3]*pre_pre_phi[0] * B[1][3]*pre_pre_phi_grad[1])
        * B[4][3]*pre_pre_phi[4] * B[5][3]*pre_pre_phi[5]
        + (1. + pre_pre_phi[10])
        * B[0][3]*pre_pre_phi[0] * B[1][3]*pre_pre_phi[1]
        * (B[4][3]*pre_pre_phi_grad[4] * B[5][3]*pre_pre_phi[5]+B[4][3]*pre_pre_phi[4] * B[5][3]*pre_pre_phi_grad[5]);
      }
      // z dir

      pre_pre_phi[8] = (quad_pt - center_pt)*Gamma_mid[2];
      pre_pre_phi[6] = B[0][2]*pre_pre_phi[0] - B[1][2]*pre_pre_phi[1];
      pre_pre_phi[7] = B[2][0]*pre_pre_phi[2] - B[3][0]*pre_pre_phi[3];

      pre_pre_phi_grad[8] = Gamma_mid[2];
      pre_pre_phi_grad[6] = B[0][2]*pre_pre_phi_grad[0] - B[1][2]*pre_pre_phi_grad[1];
      pre_pre_phi_grad[7] = B[2][0]*pre_pre_phi_grad[2] - B[3][0]*pre_pre_phi_grad[3];

      // R_x
      pre_pre_phi[9] = (B[0][2]*pre_pre_phi[0] - B[1][2]*pre_pre_phi[1])
      /(B[0][2]*pre_pre_phi[0] + B[1][2]*pre_pre_phi[1]);
      // R_y
      pre_pre_phi[10] = (B[2][0]*pre_pre_phi[2] - B[3][0]*pre_pre_phi[3])
      /(B[2][0]*pre_pre_phi[2] + B[3][0]*pre_pre_phi[3]); 
      AssertIsFinite(pre_pre_phi[9]);
      AssertIsFinite(pre_pre_phi[10]);      

      pre_pre_phi_grad[9] = 
      (pre_pre_phi_grad[6] -
        pre_pre_phi[9]*(B[0][2]*pre_pre_phi_grad[0] + B[1][2]*pre_pre_phi_grad[1])
        )/(B[0][2]*pre_pre_phi[0] + B[1][2]*pre_pre_phi[1]);

      pre_pre_phi_grad[10] = 
      (pre_pre_phi_grad[7] -
        pre_pre_phi[10]*(B[2][0]*pre_pre_phi_grad[2] + B[3][0]*pre_pre_phi_grad[3])
        )/(B[2][0]*pre_pre_phi[2] + B[3][0]*pre_pre_phi[3]);

      double zBubble = pre_pre_phi[4]*pre_pre_phi[5];
      Tensor<1,dim> dzBubble = pre_pre_phi_grad[4]*pre_pre_phi[5]
      + pre_pre_phi[4]*pre_pre_phi_grad[5];

      double zvalue;
      Tensor<1,dim> ztemp;

      pre_phi[8*r] = zBubble;
      pre_phi[9*r-1] = zBubble*pre_pre_phi[6];
      pre_phi[10*r-2] = zBubble*pre_pre_phi[7];
      pre_phi[11*r-3] = zBubble*(alpha_z*pre_pre_phi[7]*pre_pre_phi[9]
        + beta_z*pre_pre_phi[6]*pre_pre_phi[10]
        + eta_z*pre_pre_phi[9]*pre_pre_phi[10]);


      pre_phi_dx[8*r] = dzBubble[0];
      pre_phi_dy[8*r] = dzBubble[1];
      pre_phi_dz[8*r] = dzBubble[2];
      pre_phi_dx[9*r-1] = dzBubble[0]*pre_pre_phi[6]+zBubble*pre_pre_phi_grad[6][0];
      pre_phi_dy[9*r-1] = dzBubble[1]*pre_pre_phi[6]+zBubble*pre_pre_phi_grad[6][1];
      pre_phi_dz[9*r-1] = dzBubble[2]*pre_pre_phi[6]+zBubble*pre_pre_phi_grad[6][2];
      pre_phi_dx[10*r-2] = dzBubble[0]*pre_pre_phi[7]+zBubble*pre_pre_phi_grad[7][0];
      pre_phi_dy[10*r-2] = dzBubble[1]*pre_pre_phi[7]+zBubble*pre_pre_phi_grad[7][1];
      pre_phi_dz[10*r-2] = dzBubble[2]*pre_pre_phi[7]+zBubble*pre_pre_phi_grad[7][2];

      zvalue = alpha_z*pre_pre_phi[7]*pre_pre_phi[9]
        + beta_z*pre_pre_phi[6]*pre_pre_phi[10]
        + eta_z*pre_pre_phi[9]*pre_pre_phi[10];

      ztemp = alpha_z*pre_pre_phi_grad[7]*pre_pre_phi[9]
        + alpha_z*pre_pre_phi[7]*pre_pre_phi_grad[9]
        + beta_z*pre_pre_phi_grad[6]*pre_pre_phi[10]
        + beta_z*pre_pre_phi[6]*pre_pre_phi_grad[10]
        + eta_z*pre_pre_phi_grad[9]*pre_pre_phi[10]
        + eta_z*pre_pre_phi[9]*pre_pre_phi_grad[10];

      pre_phi_dx[11*r-3] = dzBubble[0]*zvalue+zBubble*ztemp[0];
      pre_phi_dy[11*r-3] = dzBubble[1]*zvalue+zBubble*ztemp[1];
      pre_phi_dz[11*r-3] = dzBubble[2]*zvalue+zBubble*ztemp[2];

      for(unsigned int j=1; j<r-2; ++j)
      {
        pre_phi[8*r+j] = zBubble*std::pow(pre_pre_phi[8],j);
        pre_phi[9*r-1+j] = zBubble*pre_pre_phi[6]*std::pow(pre_pre_phi[8],j);
        pre_phi[10*r-2+j] = zBubble*pre_pre_phi[7]*std::pow(pre_pre_phi[8],j);
        pre_phi[11*r-3+j] = zBubble*pre_pre_phi[10]*pre_pre_phi[9]*std::pow(pre_pre_phi[8],j);

        zvalue = std::pow(pre_pre_phi[8],j);
        ztemp = j*std::pow(pre_pre_phi[8],j-1)*pre_pre_phi_grad[8];
        pre_phi_dx[8*r+j] = dzBubble[0]*zvalue+zBubble*ztemp[0];
        pre_phi_dy[8*r+j] = dzBubble[1]*zvalue+zBubble*ztemp[1];
        pre_phi_dz[8*r+j] = dzBubble[2]*zvalue+zBubble*ztemp[2];

        zvalue = pre_pre_phi[6]*std::pow(pre_pre_phi[8],j);
        ztemp = pre_pre_phi[6]*j*std::pow(pre_pre_phi[8],j-1)*pre_pre_phi_grad[8]
        + pre_pre_phi_grad[6]*std::pow(pre_pre_phi[8],j);
        pre_phi_dx[9*r-1+j] = dzBubble[0]*zvalue+zBubble*ztemp[0];
        pre_phi_dy[9*r-1+j] = dzBubble[1]*zvalue+zBubble*ztemp[1];
        pre_phi_dz[9*r-1+j] = dzBubble[2]*zvalue+zBubble*ztemp[2];

        zvalue = pre_pre_phi[7]*std::pow(pre_pre_phi[8],j);
        ztemp =  pre_pre_phi[7]*j*std::pow(pre_pre_phi[8],j-1)*pre_pre_phi_grad[8]
        + pre_pre_phi_grad[7]*std::pow(pre_pre_phi[8],j);
        pre_phi_dx[10*r-2+j] = dzBubble[0]*zvalue+zBubble*ztemp[0];
        pre_phi_dy[10*r-2+j] = dzBubble[1]*zvalue+zBubble*ztemp[1];
        pre_phi_dz[10*r-2+j] = dzBubble[2]*zvalue+zBubble*ztemp[2];

        zvalue = pre_pre_phi[10]*pre_pre_phi[9]*std::pow(pre_pre_phi[8],j);
        ztemp = pre_pre_phi[10]*pre_pre_phi[9]*j*std::pow(pre_pre_phi[8],j-1)*pre_pre_phi_grad[8]
        + pre_pre_phi_grad[10]*pre_pre_phi[9]*std::pow(pre_pre_phi[8],j)
        + pre_pre_phi[10]*pre_pre_phi_grad[9]*std::pow(pre_pre_phi[8],j);
        pre_phi_dx[11*r-3+j] = dzBubble[0]*zvalue+zBubble*ztemp[0];
        pre_phi_dy[11*r-3+j] = dzBubble[1]*zvalue+zBubble*ztemp[1];
        pre_phi_dz[11*r-3+j] = dzBubble[2]*zvalue+zBubble*ztemp[2];
      }

      pre_phi[8*r+r-2] = zBubble*std::pow(pre_pre_phi[8],r-2);
      pre_phi[9*r-1+r-2] = zBubble*pre_pre_phi[9]*std::pow(pre_pre_phi[8],r-2);
      pre_phi[10*r-2+r-2] = zBubble*pre_pre_phi[10]*std::pow(pre_pre_phi[8],r-2);
      pre_phi[11*r-3+r-2] = zBubble*pre_pre_phi[9]*pre_pre_phi[10]*std::pow(pre_pre_phi[8],r-2); 

      zvalue = std::pow(pre_pre_phi[8],r-2);
      ztemp = (r-2)*std::pow(pre_pre_phi[8],r-3)*pre_pre_phi_grad[8];
      pre_phi_dx[8*r+r-2] = dzBubble[0]*zvalue+zBubble*ztemp[0];
      pre_phi_dy[8*r+r-2] = dzBubble[1]*zvalue+zBubble*ztemp[1];
      pre_phi_dz[8*r+r-2] = dzBubble[2]*zvalue+zBubble*ztemp[2];

      zvalue = pre_pre_phi[9]*std::pow(pre_pre_phi[8],r-2);
      ztemp = pre_pre_phi[9]*(r-2)*std::pow(pre_pre_phi[8],r-3)*pre_pre_phi_grad[8]
      + pre_pre_phi_grad[9]*std::pow(pre_pre_phi[8],r-2);
      pre_phi_dx[9*r-1+r-2] = dzBubble[0]*zvalue+zBubble*ztemp[0];
      pre_phi_dy[9*r-1+r-2] = dzBubble[1]*zvalue+zBubble*ztemp[1];
      pre_phi_dz[9*r-1+r-2] = dzBubble[2]*zvalue+zBubble*ztemp[2];

      zvalue = pre_pre_phi[10]*std::pow(pre_pre_phi[8],r-2);
      ztemp = pre_pre_phi[10]*(r-2)*std::pow(pre_pre_phi[8],r-3)*pre_pre_phi_grad[8]
      + pre_pre_phi_grad[10]*std::pow(pre_pre_phi[8],r-2);
      pre_phi_dx[10*r-2+r-2] = dzBubble[0]*zvalue+zBubble*ztemp[0];
      pre_phi_dy[10*r-2+r-2] = dzBubble[1]*zvalue+zBubble*ztemp[1];
      pre_phi_dz[10*r-2+r-2] = dzBubble[2]*zvalue+zBubble*ztemp[2];

      zvalue = pre_pre_phi[9]*pre_pre_phi[10]*std::pow(pre_pre_phi[8],r-2); 
      ztemp = pre_pre_phi[9]*pre_pre_phi[10]*(r-2)*std::pow(pre_pre_phi[8],r-3)*pre_pre_phi_grad[8]
      + pre_pre_phi_grad[9]*pre_pre_phi[10]*std::pow(pre_pre_phi[8],r-2)
      + pre_pre_phi[9]*pre_pre_phi_grad[10]*std::pow(pre_pre_phi[8],r-2);
      pre_phi_dx[11*r-3+r-2] = dzBubble[0]*zvalue+zBubble*ztemp[0];
      pre_phi_dy[11*r-3+r-2] = dzBubble[1]*zvalue+zBubble*ztemp[1];
      pre_phi_dz[11*r-3+r-2] = dzBubble[2]*zvalue+zBubble*ztemp[2];

    } //end dim==3

    Vector<double> soln;
    Vector<double> soln_dx;
    Vector<double> soln_dy;
    Vector<double> soln_dz;
    double pre_phi_int_2d=0.;
    double pre_phi_int_dx=0.;
    double pre_phi_int_dy=0.;
    double pre_phi_int_dz=0.;

    if(dim==2)
    {
      soln.reinit(4*fe_degree);
      soln_dx.reinit(4*fe_degree);
      soln_dy.reinit(4*fe_degree);
      A.Tvmult(soln, pre_phi);
      A.Tvmult(soln_dx, pre_phi_dx);
      A.Tvmult(soln_dy, pre_phi_dy);

      pre_phi_int_2d = pre_pre_phi[0]*pre_pre_phi[1]
      *pre_pre_phi[2]*pre_pre_phi[3];
      // lambda_0*lambda_1*lambda_2*lambda_3

      pre_phi_int_dx = 
      pre_pre_phi_grad[0][0]*pre_pre_phi[1]
      *pre_pre_phi[2]*pre_pre_phi[3]
      + pre_pre_phi[0]*pre_pre_phi_grad[1][0]
      *pre_pre_phi[2]*pre_pre_phi[3]
      + pre_pre_phi[0]*pre_pre_phi[1]
      *pre_pre_phi_grad[2][0]*pre_pre_phi[3]
      + pre_pre_phi[0]*pre_pre_phi[1]
      *pre_pre_phi[2]*pre_pre_phi_grad[3][0];

      pre_phi_int_dy = 
      pre_pre_phi_grad[0][1]*pre_pre_phi[1]
      *pre_pre_phi[2]*pre_pre_phi[3]
      + pre_pre_phi[0]*pre_pre_phi_grad[1][1]
      *pre_pre_phi[2]*pre_pre_phi[3]
      + pre_pre_phi[0]*pre_pre_phi[1]
      *pre_pre_phi_grad[2][1]*pre_pre_phi[3]
      + pre_pre_phi[0]*pre_pre_phi[1]
      *pre_pre_phi[2]*pre_pre_phi_grad[3][1];
    }

    if(dim==3)
    {
      soln.reinit(12*fe_degree-4);
      soln_dx.reinit(12*fe_degree-4);
      soln_dy.reinit(12*fe_degree-4);
      soln_dz.reinit(12*fe_degree-4);      
      A.Tvmult(soln, pre_phi);
      A.Tvmult(soln_dx, pre_phi_dx);
      A.Tvmult(soln_dy, pre_phi_dy);
      A.Tvmult(soln_dz, pre_phi_dz);
    }

    for(unsigned int i=0; i<this->dofs_per_cell; ++i)
    {
      if(dim==2)
      {
         if(i<4*fe_degree)
         {
          //vertex and edge dofs
          if (flags & update_values)
            data.shape_values(i,k) = soln[i];

          if (flags & update_gradients)
          {
            data.shape_gradients[i][k][0] = soln_dx[i];
            data.shape_gradients[i][k][1] = soln_dy[i];
          }
         }
         else
         {
          Assert(fe_degree>=4, ExcMessage("if r<4, should see no interior dof"));
          Assert(i-4*fe_degree>=0, ExcMessage("if r<4, should see no interior dof"));
          int p=0, q=0;

          switch(i-4*fe_degree)
          {
            case 0:
              p=0;q=0; break;
            case 1:
              p=1;q=0; break;
            case 2:
              p=0;q=1; break;
            default:
              Assert(false, ExcNotImplemented());
          }

          // interior dofs
          if(flags & update_values)
          {
            data.shape_values(i,k) = pre_phi_int_2d* std::pow(pre_pre_phi[4],p)
            * std::pow(pre_pre_phi[5],q);
          }
          if (flags & update_gradients)
          { 
            data.shape_gradients[i][k][0] = 
            pre_phi_int_dx * std::pow(pre_pre_phi[4],p)
            * std::pow(pre_pre_phi[5],q)
            + pre_phi_int_2d * ((p==0)?0.: p*std::pow(pre_pre_phi[4],p-1)*pre_pre_phi_grad[4][0])
            * std::pow(pre_pre_phi[5],q)
            + pre_phi_int_2d * std::pow(pre_pre_phi[4],p)
            * ((q==0)? 0.: q*std::pow(pre_pre_phi[5],q-1)*pre_pre_phi_grad[5][0]);

            data.shape_gradients[i][k][1] = 
            pre_phi_int_dy * std::pow(pre_pre_phi[4],p)
            * std::pow(pre_pre_phi[5],q)
            + pre_phi_int_2d * ((p==0)?0.: p*std::pow(pre_pre_phi[4],p-1)*pre_pre_phi_grad[4][1])
            * std::pow(pre_pre_phi[5],q)
            + pre_phi_int_2d * std::pow(pre_pre_phi[4],p)
            * ((q==0)? 0.: q*std::pow(pre_pre_phi[5],q-1)*pre_pre_phi_grad[5][1]);
          }
         }
      } //end if dim==2 
      else if(dim==3)
      {
        if(i<12*fe_degree-4)
         {
          //vertex and edge dofs
          if (flags & update_values)
            data.shape_values(i,k) = soln[i];

          if (flags & update_gradients)
          {
            data.shape_gradients[i][k][0] = soln_dx[i];
            data.shape_gradients[i][k][1] = soln_dy[i];
            data.shape_gradients[i][k][2] = soln_dz[i];
          }
         }
         else
         {
          Assert(r>3,ExcMessage("should not go here if degree<=3"));
          Assert(r<5,ExcMessage("higer order not available yet"));
          int ii = i-(12*fe_degree-4);

          if (flags & update_values)
            data.shape_values(i,k) = pre_phi_int[ii];

          if (flags & update_gradients)
            data.shape_gradients[i][k]= pre_phi_int_grad[ii];
         }
      }
    }
  } //end of step 4)
}

// explicit instantiations
#include "fe_s_np_base.inst"

DEAL_II_NAMESPACE_CLOSE