// ---------------------------------------------------------------------
//
// Copyright (C) 2009 - 2014 by the deal.II authors
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


#include <deal.II/fe/fe_face_np.h>
#include <deal.II/fe/fe_poly_face_np.templates.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/householder.h>
#include <sstream>

DEAL_II_NAMESPACE_OPEN


namespace
{
  std::vector<Point<1> >
  get_QGaussLobatto_points (const unsigned int degree)
  {
    if (degree > 0)
      return QGaussLobatto<1>(degree+1).get_points();
    else
      return std::vector<Point<1> >(1, Point<1>(0.5));
  }
}

// --------------------------------------- FE_FaceP --------------------------

template <int dim, int spacedim>
FE_FaceP_NP<dim,spacedim>::FE_FaceP_NP (const unsigned int degree)
  :
  FE_PolyFace_NP<PolynomialSpace<dim-1>, dim, spacedim>
  (PolynomialSpace<dim-1>(Polynomials::Legendre::generate_complete_basis(degree)),
  FiniteElementData<dim>(get_dpo_vector(degree), 1, degree, FiniteElementData<dim>::L2),
  std::vector<bool>(1,true))
{}



template <int dim, int spacedim>
FiniteElement<dim,spacedim> *
FE_FaceP_NP<dim,spacedim>::clone() const
{
  return new FE_FaceP_NP<dim,spacedim>(this->degree);
}



template <int dim, int spacedim>
std::string
FE_FaceP_NP<dim,spacedim>::get_name () const
{
  // note that the FETools::get_fe_from_name function depends on the
  // particular format of the string this function returns, so they have to be
  // kept in synch
  std::ostringstream namebuf;
  namebuf << "FE_FaceP_NP<"
          << Utilities::dim_string(dim,spacedim)
          << ">(" << this->degree << ")";

  return namebuf.str();
}



template <int dim, int spacedim>
bool
FE_FaceP_NP<dim,spacedim>::has_support_on_face (
  const unsigned int shape_index,
  const unsigned int face_index) const
{
  return (face_index == (shape_index/this->dofs_per_face));
}



template <int dim, int spacedim>
std::vector<unsigned int>
FE_FaceP_NP<dim,spacedim>::get_dpo_vector (const unsigned int deg)
{
  std::vector<unsigned int> dpo(dim+1, 0U);
  dpo[dim-1] = deg+1;
  for (unsigned int i=1; i<dim-1; ++i)
    {
      dpo[dim-1] *= deg+1+i;
      dpo[dim-1] /= i+1;
    }
  return dpo;
}




template <int dim, int spacedim>
bool
FE_FaceP_NP<dim,spacedim>::hp_constraints_are_implemented () const
{
  return true;
}



template <int dim, int spacedim>
FiniteElementDomination::Domination
FE_FaceP_NP<dim,spacedim>::
compare_for_face_domination (const FiniteElement<dim,spacedim> &fe_other) const
{
  if (const FE_FaceP<dim,spacedim> *fe_q_other
      = dynamic_cast<const FE_FaceP<dim,spacedim>*>(&fe_other))
    {
      if (this->degree < fe_q_other->degree)
        return FiniteElementDomination::this_element_dominates;
      else if (this->degree == fe_q_other->degree)
        return FiniteElementDomination::either_element_can_dominate;
      else
        return FiniteElementDomination::other_element_dominates;
    }
  else if (dynamic_cast<const FE_Nothing<dim>*>(&fe_other) != 0)
    {
      // the FE_Nothing has no degrees of freedom and it is typically used in
      // a context where we don't require any continuity along the interface
      return FiniteElementDomination::no_requirements;
    }

  Assert (false, ExcNotImplemented());
  return FiniteElementDomination::neither_element_dominates;
}




template <int dim, int spacedim>
void
FE_FaceP_NP<dim,spacedim>::
get_face_interpolation_matrix (const FiniteElement<dim,spacedim> &source_fe,
                               FullMatrix<double>       &interpolation_matrix) const
{
  get_subface_interpolation_matrix (source_fe, numbers::invalid_unsigned_int,
                                    interpolation_matrix);
}



template <int dim, int spacedim>
void
FE_FaceP_NP<dim,spacedim>::
get_subface_interpolation_matrix (const FiniteElement<dim,spacedim> &x_source_fe,
                                  const unsigned int        subface,
                                  FullMatrix<double>       &interpolation_matrix) const
{
  // this function is similar to the respective method in FE_Q

  Assert (interpolation_matrix.n() == this->dofs_per_face,
          ExcDimensionMismatch (interpolation_matrix.n(),
                                this->dofs_per_face));
  Assert (interpolation_matrix.m() == x_source_fe.dofs_per_face,
          ExcDimensionMismatch (interpolation_matrix.m(),
                                x_source_fe.dofs_per_face));

  // see if source is a FaceP element
  if (const FE_FaceP_NP<dim,spacedim> *source_fe
      = dynamic_cast<const FE_FaceP_NP<dim,spacedim> *>(&x_source_fe))
    {
      // Make sure that the element for which the DoFs should be constrained
      // is the one with the higher polynomial degree.  Actually the procedure
      // will work also if this assertion is not satisfied. But the matrices
      // produced in that case might lead to problems in the hp procedures,
      // which use this method.
      Assert (this->dofs_per_face <= source_fe->dofs_per_face,
              (typename FiniteElement<dim,spacedim>::
               ExcInterpolationNotImplemented ()));

      // do this as in FETools by solving a least squares problem where we
      // force the source FE polynomial to be equal the given FE on all
      // quadrature points
      const QGauss<dim-1> face_quadrature (source_fe->degree+1);

      // Rule of thumb for FP accuracy, that can be expected for a given
      // polynomial degree.  This value is used to cut off values close to
      // zero.
      const double eps = 2e-13*(this->degree+1)*(dim-1);

      FullMatrix<double> mass (face_quadrature.size(), source_fe->dofs_per_face);

      for (unsigned int k = 0; k < face_quadrature.size(); ++k)
        {
          const Point<dim-1> p =
            subface == numbers::invalid_unsigned_int ?
            face_quadrature.point(k) :
            GeometryInfo<dim-1>::child_to_cell_coordinates (face_quadrature.point(k),
                                                            subface);

          for (unsigned int j = 0; j < source_fe->dofs_per_face; ++j)
            mass (k , j) = source_fe->poly_space.compute_value(j, p);
        }

      Householder<double> H(mass);
      Vector<double> v_in(face_quadrature.size());
      Vector<double> v_out(source_fe->dofs_per_face);


      // compute the interpolation matrix by evaluating on the fine side and
      // then solving the least squares problem
      for (unsigned int i=0; i<this->dofs_per_face; ++i)
        {
          for (unsigned int k = 0; k < face_quadrature.size(); ++k)
            {
              const Point<dim-1> p = numbers::invalid_unsigned_int ?
                                     face_quadrature.point(k) :
                                     GeometryInfo<dim-1>::child_to_cell_coordinates (face_quadrature.point(k),
                                         subface);
              v_in(k) = this->poly_space.compute_value(i, p);
            }
          const double result = H.least_squares(v_out, v_in);
          (void)result;
          Assert(result < 1e-12, FETools::ExcLeastSquaresError (result));

          for (unsigned int j = 0; j < source_fe->dofs_per_face; ++j)
            {
              double matrix_entry = v_out(j);

              // Correct the interpolated value. I.e. if it is close to 1 or 0,
              // make it exactly 1 or 0. Unfortunately, this is required to avoid
              // problems with higher order elements.
              if (std::fabs (matrix_entry - 1.0) < eps)
                matrix_entry = 1.0;
              if (std::fabs (matrix_entry) < eps)
                matrix_entry = 0.0;

              interpolation_matrix(j,i) = matrix_entry;
            }
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



template <int dim, int spacedim>
std::pair<Table<2,bool>, std::vector<unsigned int> >
FE_FaceP_NP<dim,spacedim>::get_constant_modes () const
{
  Table<2,bool> constant_modes(1, this->dofs_per_cell);
  for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
    constant_modes(0, face*this->dofs_per_face) = true;
  return std::pair<Table<2,bool>, std::vector<unsigned int> >
         (constant_modes, std::vector<unsigned int>(1, 0));
}

// explicit instantiations
#include "fe_face_np.inst"


DEAL_II_NAMESPACE_CLOSE
