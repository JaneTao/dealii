// ---------------------------------------------------------------------
//
// Copyright (C) 2010 - 2014 by the deal.II authors
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

#ifndef __deal2__fe_dg_vector_npc_h
#define __deal2__fe_dg_vector_npc_h

#include <deal.II/base/config.h>
#include <deal.II/base/table.h>
#include <deal.II/base/polynomials_bdm.h>
#include <deal.II/base/polynomials_acfull.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor_product_polynomials.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_poly_tensor_npc.h>

#include <vector>

DEAL_II_NAMESPACE_OPEN

template <int dim, int spacedim> class MappingQ;


/**
 * DG elements based on vector valued polynomials.
 *
 * These elements use vector valued polynomial spaces as they have been
 * introduced for H<sup>div</sup> and H<sup>curl</sup> conforming finite
 * elements, but do not use the usual continuity of these elements. Thus, they
 * are suitable for DG and hybrid formulations involving these function
 * spaces.
 *
 * The template argument <tt>POLY</tt> refers to a vector valued polynomial
 * space like PolynomialsRaviartThomas or PolynomialsNedelec. Note that the
 * dimension of the polynomial space and the argument <tt>dim</tt> must
 * coincide.
 *
 * @ingroup febase
 * @author Guido Kanschat
 * @date 2010
 */
template <class POLY, int dim, int spacedim=dim>
class FE_DGVector_NPC
  :
  public FE_PolyTensor_NPC<POLY, dim, spacedim>
{
public:
  /**
   * Constructor for the vector element of degree @p p.
   */
  FE_DGVector_NPC (const unsigned int p, MappingType m);
public:

  FiniteElement<dim, spacedim> *clone() const;

  /**
   * Return a string that uniquely identifies a finite element. This class
   * returns <tt>FE_RaviartThomas<dim>(degree)</tt>, with @p dim and @p degree
   * replaced by appropriate values.
   */
  virtual std::string get_name () const;


  /**
   * This function returns @p true, if the shape function @p shape_index has
   * non-zero function values somewhere on the face @p face_index.
   *
   * For this element, we always return @p true.
   */
  virtual bool has_support_on_face (const unsigned int shape_index,
                                    const unsigned int face_index) const;

  virtual void interpolate(std::vector<double>                &local_dofs,
                           const std::vector<double> &values) const;
  virtual void interpolate(std::vector<double>                &local_dofs,
                           const std::vector<Vector<double> > &values,
                           unsigned int offset = 0) const;
  virtual void interpolate(
    std::vector<double> &local_dofs,
    const VectorSlice<const std::vector<std::vector<double> > > &values) const;
  virtual std::size_t memory_consumption () const;

private:
  /**
   * Only for internal use. Its full name is @p get_dofs_per_object_vector
   * function and it creates the @p dofs_per_object vector that is needed
   * within the constructor to be passed to the constructor of @p
   * FiniteElementData.
   */
  static std::vector<unsigned int>
  get_dpo_vector (const unsigned int degree);

  /**
   * Initialize the @p generalized_support_points field of the FiniteElement
   * class and fill the tables with @p interior_weights. Called from the
   * constructor.
   *
   * See the
   * @ref GlossGeneralizedSupport "glossary entry on generalized support points"
   * for more information.
   */
  void initialize_support_points (const unsigned int degree);

  /**
   * Initialize the interpolation from functions on refined mesh cells onto
   * the father cell. According to the philosophy of the Raviart-Thomas
   * element, this restriction operator preserves the divergence of a function
   * weakly.
   */
  void initialize_restriction ();

  // internalData?
  // class InternalData : public FiniteElement<dim,spacedim>::InternalDataBase
  // {
  //   std::vector<std::vector<Tensor<1,dim> > > shape_values;
  //   std::vector< std::vector< DerivativeForm<1, dim, spacedim> > > shape_grads;
  // };


  Table<3, double> interior_weights;
};



template <int dim, int spacedim=dim>
class FE_ATRed : public FE_DGVector_NPC<PolynomialsBDM<dim>, dim, spacedim>
{
public:

  FE_ATRed (const unsigned int p);

  virtual std::string get_name () const;
};

template <int dim, int spacedim=dim>
class FE_ATFull : public FE_DGVector_NPC<PolynomialsACFull<dim>, dim, spacedim>
{
public:

  FE_ATFull (const unsigned int p);

  virtual std::string get_name () const;
};


DEAL_II_NAMESPACE_CLOSE

#endif
