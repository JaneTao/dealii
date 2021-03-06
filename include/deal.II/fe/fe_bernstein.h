// ---------------------------------------------------------------------
//
// Copyright (C) 2000 - 2015 by the deal.II authors
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

#ifndef __deal2__fe_bernstein_h
#define __deal2__fe_bernstein_h

#include <deal.II/base/config.h>
#include <deal.II/base/tensor_product_polynomials.h>
#include <deal.II/fe/fe_q_base.h>

DEAL_II_NAMESPACE_OPEN


/*!@addtogroup fe */
/*@{*/

/**
 * Implementation of a scalar Bernstein finite element @p that we call FE_Bernstein
 * in analogy with FE_Q that yields the
 * finite element space of continuous, piecewise Bernstein polynomials of degree @p p in
 * each coordinate direction. This class is realized using tensor product
 * polynomials of Bernstein basis polynomials.
 *
 *
 * The standard constructor of this class takes the degree @p p of this finite
 * element.
 *
 * For more information about the <tt>spacedim</tt> template parameter
 * check the documentation of FiniteElement or the one of
 * Triangulation.
 *
 * <h3>Implementation</h3>
 *
 * The constructor creates a TensorProductPolynomials object that includes the
 * tensor product of @p Bernstein polynomials of degree @p p. This
 * @p TensorProductPolynomials object provides all values and derivatives of
 * the shape functions.
 *
 * <h3>Numbering of the degrees of freedom (DoFs)</h3>
 *
 * The original ordering of the shape functions represented by the
 * TensorProductPolynomials is a tensor product
 * numbering. However, the shape functions on a cell are renumbered
 * beginning with the shape functions whose support points are at the
 * vertices, then on the line, on the quads, and finally (for 3d) on
 * the hexes. See the documentation of FE_Q for more details.
 *
 *
 * @author Marco Tezzele, Luca Heltai
 * @date 2013, 2015
 */

template <int dim, int spacedim=dim>
class FE_Bernstein : public FE_Q_Base<TensorProductPolynomials<dim>,dim,spacedim>
{
public:
  /**
   * Constructor for tensor product polynomials of degree @p p.
   */
  FE_Bernstein (const unsigned int p);

  /**
   * Return a string that uniquely identifies a finite element. This class
   * returns <tt>FE_Bernstein<dim>(degree)</tt>, with @p dim and @p degree replaced by
   * appropriate values.
   */
  virtual std::string get_name () const;

protected:

  /**
   * @p clone function instead of a copy constructor.
   *
   * This function is needed by the constructors of @p FESystem.
   */
  virtual FiniteElement<dim,spacedim> *clone() const;

  /**
   * Only for internal use. Its full name is @p get_dofs_per_object_vector
   * function and it creates the @p dofs_per_object vector that is needed
   * within the constructor to be passed to the constructor of @p
   * FiniteElementData.
   */
  static std::vector<unsigned int> get_dpo_vector(const unsigned int degree);

  /**
   * This function renumbers Bernstein basis functions from hierarchic to
   * lexicographic numbering.
   */
  TensorProductPolynomials<dim> renumber_bases(const unsigned int degree);
};



/*@}*/

DEAL_II_NAMESPACE_CLOSE

#endif
