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

#ifndef __deal2__fe_face_np_h
#define __deal2__fe_face_np_h

#include <deal.II/base/config.h>
#include <deal.II/base/polynomial_space.h>
#include <deal.II/fe/fe_poly_face_np.h>

DEAL_II_NAMESPACE_OPEN
/**
 * A finite element, which is a Legendre element of complete polynomials on
 * each face (i.e., it is the face equivalent of what FE_DGP is on cells) and
 * undefined in the interior of the cells. The basis functions on the faces
 * are from Polynomials::Legendre.
 *
 * Although the name does not give it away, the element is discontinuous at
 * locations where faces of cells meet. The element serves in hybridized
 * methods, e.g. in combination with the FE_DGP element. An example of
 * hybridizes methods can be found in the step-51 tutorial program.
 *
 * @note Since this element is defined only on faces, only FEFaceValues and
 * FESubfaceValues will be able to extract reasonable values from any face
 * polynomial. In order to make the use of FESystem simpler, using a (cell)
 * FEValues object will not fail using this finite element space, but all
 * shape function values extracted will be equal to zero.
 *
 * @ingroup fe
 * @author Martin Kronbichler
 * @date 2013
 */
template <int dim, int spacedim=dim>
class FE_FaceP_NP : public FE_PolyFace_NP<PolynomialSpace<dim-1>, dim, spacedim>
{
public:
  /**
   * Constructor for complete basis of polynomials of degree <tt>p</tt>. The
   * shape functions created using this constructor correspond to Legendre
   * polynomials in each coordinate direction.
   */
  FE_FaceP_NP(unsigned int p);

  /**
   * Clone method.
   */
  virtual FiniteElement<dim,spacedim> *clone() const;

  /**
   * Return a string that uniquely identifies a finite element. This class
   * returns <tt>FE_FaceP<dim>(degree)</tt> , with <tt>dim</tt> and
   * <tt>degree</tt> replaced by appropriate values.
   */
  virtual std::string get_name () const;

  /**
   * Return the matrix interpolating from a face of of one element to the face
   * of the neighboring element.  The size of the matrix is then
   * <tt>source.dofs_per_face</tt> times <tt>this->dofs_per_face</tt>. This
   * element only provides interpolation matrices for elements of the same
   * type and FE_Nothing. For all other elements, an exception of type
   * FiniteElement<dim,spacedim>::ExcInterpolationNotImplemented is thrown.
   */
  virtual void
  get_face_interpolation_matrix (const FiniteElement<dim,spacedim> &source,
                                 FullMatrix<double>       &matrix) const;

  /**
   * Return the matrix interpolating from a face of of one element to the face
   * of the neighboring element.  The size of the matrix is then
   * <tt>source.dofs_per_face</tt> times <tt>this->dofs_per_face</tt>. This
   * element only provides interpolation matrices for elements of the same
   * type and FE_Nothing. For all other elements, an exception of type
   * FiniteElement<dim,spacedim>::ExcInterpolationNotImplemented is thrown.
   */
  virtual void
  get_subface_interpolation_matrix (const FiniteElement<dim,spacedim> &source,
                                    const unsigned int        subface,
                                    FullMatrix<double>       &matrix) const;

  /**
   * This function returns @p true, if the shape function @p shape_index has
   * non-zero function values somewhere on the face @p face_index.
   */
  virtual bool has_support_on_face (const unsigned int shape_index,
                                    const unsigned int face_index) const;

  /**
   * Return whether this element implements its hanging node constraints in
   * the new way, which has to be used to make elements "hp compatible".
   */
  virtual bool hp_constraints_are_implemented () const;

  /**
   * Return whether this element dominates the one given as argument when they
   * meet at a common face, whether it is the other way around, whether
   * neither dominates, or if either could dominate.
   *
   * For a definition of domination, see FiniteElementBase::Domination and in
   * particular the
   * @ref hp_paper "hp paper".
   */
  virtual
  FiniteElementDomination::Domination
  compare_for_face_domination (const FiniteElement<dim,spacedim> &fe_other) const;

  /**
   * Returns a list of constant modes of the element. For this element, the
   * first entry on each face is true, all other are false (as the constant
   * function is represented by the first base function of Legendre
   * polynomials).
   */
  virtual std::pair<Table<2,bool>, std::vector<unsigned int> >
  get_constant_modes () const;

private:
  /**
   * Return vector with dofs per vertex, line, quad, hex.
   */
  static std::vector<unsigned int> get_dpo_vector (const unsigned int deg);
};


DEAL_II_NAMESPACE_CLOSE

#endif
