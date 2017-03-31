// ---------------------------------------------------------------------
//
// Copyright (C) 2005 - 2014 by the deal.II authors
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

#ifndef __deal2__fe_poly_tensor_npc_h
#define __deal2__fe_poly_tensor_npc_h


#include <deal.II/lac/full_matrix.h>
#include <deal.II/fe/fe.h>
#include <deal.II/base/derivative_form.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

DEAL_II_NAMESPACE_OPEN

template <class POLY, int dim, int spacedim=dim>
class FE_PolyTensor_NPC : public FiniteElement<dim,spacedim>
{
public:

  FE_PolyTensor_NPC  (const unsigned int degree,
                      const FiniteElementData<dim> &fe_data,
                      const std::vector<bool> &restriction_is_additive_flags,
                      const std::vector<ComponentMask> &nonzero_components);

  virtual double shape_value (const unsigned int i,
                              const Point<dim> &p) const;

  virtual double shape_value_component (const unsigned int i,
                                        const Point<dim> &p,
                                        const unsigned int component) const;

  virtual Tensor<1,dim> shape_grad (const unsigned int  i,
                                    const Point<dim>   &p) const;

  virtual Tensor<1,dim> shape_grad_component (const unsigned int i,
                                              const Point<dim> &p,
                                              const unsigned int component) const;

  virtual Tensor<2,dim> shape_grad_grad (const unsigned int  i,
                                         const Point<dim> &p) const;

  virtual Tensor<2,dim> shape_grad_grad_component (const unsigned int i,
                                                   const Point<dim> &p,
                                                   const unsigned int component) const;

  virtual UpdateFlags update_once (const UpdateFlags flags) const;
 
  virtual UpdateFlags update_each (const UpdateFlags flags) const;
  
  MappingType mapping_type;

  virtual
  typename Mapping<dim,spacedim>::InternalDataBase *
  get_data (const UpdateFlags,
            const Mapping<dim,spacedim> &mapping,
            const Quadrature<dim> &quadrature) const ;

  virtual void
  fill_fe_values (const Mapping<dim,spacedim>                       &mapping,
                  const typename Triangulation<dim,spacedim>::cell_iterator &cell,
                  const Quadrature<dim>                             &quadrature,
                  typename Mapping<dim,spacedim>::InternalDataBase  &mapping_internal,
                  typename Mapping<dim,spacedim>::InternalDataBase  &fe_internal,
                  FEValuesData<dim,spacedim>                        &data,
                  CellSimilarity::Similarity                   &cell_similarity) const;

  virtual void
  fill_fe_face_values (const Mapping<dim,spacedim> &mapping,
                       const typename Triangulation<dim,spacedim>::cell_iterator &cell,
                       const unsigned int                                  face_no,
                       const Quadrature<dim-1>                            &quadrature,
                       typename Mapping<dim,spacedim>::InternalDataBase   &mapping_internal,
                       typename Mapping<dim,spacedim>::InternalDataBase   &fe_internal,
                       FEValuesData<dim,spacedim> &data) const ;

  virtual void
  fill_fe_subface_values (const Mapping<dim,spacedim> &mapping,
                          const typename Triangulation<dim,spacedim>::cell_iterator &cell,
                          const unsigned int                    face_no,
                          const unsigned int                    sub_no,
                          const Quadrature<dim-1>                &quadrature,
                          typename Mapping<dim,spacedim>::InternalDataBase      &mapping_internal,
                          typename Mapping<dim,spacedim>::InternalDataBase      &fe_internal,
                          FEValuesData<dim,spacedim> &data) const ;

  class InternalData : public FiniteElement<dim,spacedim>::InternalDataBase
  {
  public:
    InternalData(const unsigned int n_shape_functions);

    std::vector<Tensor<1,dim> > shape_derivatives_x;
    std::vector<Tensor<1,dim> > shape_derivatives_y;
    std::vector<Tensor<1,dim> > shape_derivatives_z;    
    std::vector<Tensor<1,dim> > corner_derivatives;

    std::vector<Point<spacedim> > mapping_support_points;

    unsigned int n_shape_functions;

    Tensor<1,dim> derivative (const unsigned int qpoint,
                              const unsigned int shape_nr,
                              const unsigned int proj_dir) const;

    Tensor<1,dim> &derivative (const unsigned int qpoint,
                               const unsigned int shape_nr,
                               const unsigned int proj_dir);

    Tensor<1,dim> corner_derivative (const unsigned int cn_nr,
      const unsigned int shape_nr) const;

    Tensor<1,dim> &corner_derivative (const unsigned int cn_nr,
      const unsigned int shape_nr);

    //===== for fe shape functions =====

    std::vector<std::vector<Tensor<1,dim> > > shape_values;
    std::vector< std::vector< DerivativeForm<1, dim, spacedim> > > shape_grads;
  };

  void compute_shapes (const std::vector<Point<dim> > &unit_points,
                       InternalData &data) const;

  virtual void compute_mapping_support_points(
    const typename Triangulation<dim,spacedim>::cell_iterator &cell,
    std::vector<Point<spacedim> > &a) const;


  virtual void compute_shapes_virtual (const std::vector<Point<dim> > &unit_points,
                                       InternalData &data) const;

  POLY poly_space;

  unsigned int piola_boundary;

  static const unsigned int n_shape_functions = GeometryInfo<dim>::vertices_per_cell;

  FullMatrix<double> inverse_node_matrix;

  mutable Point<dim> cached_point;

  mutable std::vector<Tensor<1,dim> > cached_values;

  mutable std::vector<Tensor<2,dim> > cached_grads;

  mutable std::vector<Tensor<3,dim> > cached_grad_grads;
};


template<class POLY, int dim, int spacedim>
inline
Tensor<1,dim>
FE_PolyTensor_NPC<POLY,dim,spacedim>::InternalData::derivative (const unsigned int qpoint,
                                                           const unsigned int shape_nr,
                                                           const unsigned int proj_dir) const
{
  Assert(qpoint*n_shape_functions + shape_nr < shape_derivatives_x.size(),
         ExcIndexRange(qpoint*n_shape_functions + shape_nr, 0,
                       shape_derivatives_x.size()));

  if(proj_dir == 1)
  {
    return shape_derivatives_x [qpoint*n_shape_functions + shape_nr];
  }else{
    if(proj_dir == 2)
      return shape_derivatives_y [qpoint*n_shape_functions + shape_nr];
    else
      return shape_derivatives_z [qpoint*n_shape_functions + shape_nr];
  }
}

template<class POLY, int dim, int spacedim>
inline
Tensor<1,dim> &
FE_PolyTensor_NPC<POLY,dim,spacedim>::InternalData::derivative (const unsigned int qpoint,
                                                           const unsigned int shape_nr,
                                                           const unsigned int proj_dir)
{
  Assert(qpoint*n_shape_functions + shape_nr < shape_derivatives_x.size(),
         ExcIndexRange(qpoint*n_shape_functions + shape_nr, 0,
                       shape_derivatives_x.size()));

  if(proj_dir == 1)
  {
    return shape_derivatives_x [qpoint*n_shape_functions + shape_nr];
  }else{
    if(proj_dir == 2)
      return shape_derivatives_y [qpoint*n_shape_functions + shape_nr];
    else
      return shape_derivatives_z [qpoint*n_shape_functions + shape_nr];   
  }
}


template<class POLY, int dim, int spacedim>
inline
Tensor<1,dim>
FE_PolyTensor_NPC<POLY,dim,spacedim>::InternalData::corner_derivative (const unsigned int cn_nr,
  const unsigned int shape_nr) const
{
  Assert(cn_nr*n_shape_functions + shape_nr < corner_derivatives.size(),
         ExcIndexRange(cn_nr*n_shape_functions + shape_nr, 0,
                       corner_derivatives.size()));
  return corner_derivatives [cn_nr*n_shape_functions + shape_nr];
}

template<class POLY, int dim, int spacedim>
inline
Tensor<1,dim> &
FE_PolyTensor_NPC<POLY,dim,spacedim>::InternalData::corner_derivative (const unsigned int cn_nr,
  const unsigned int shape_nr)
{
  Assert(cn_nr*n_shape_functions + shape_nr < corner_derivatives.size(),
         ExcIndexRange(cn_nr*n_shape_functions + shape_nr, 0,
                       corner_derivatives.size()));
  return corner_derivatives [cn_nr*n_shape_functions + shape_nr];
}

DEAL_II_NAMESPACE_CLOSE

#endif
