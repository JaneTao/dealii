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

#include <deal.II/base/derivative_form.h>
#include <deal.II/base/qprojector.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/polynomials_bdm.h>
#include <deal.II/base/polynomials_acfull.h>
#include <deal.II/fe/fe_poly_tensor_npc.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/grid/tria_accessor.h>

DEAL_II_NAMESPACE_OPEN

template<class POLY, int dim, int spacedim>
const unsigned int FE_PolyTensor_NPC<POLY,dim,spacedim>::n_shape_functions;

template<class POLY, int dim, int spacedim>
FE_PolyTensor_NPC<POLY,dim,spacedim>::InternalData::InternalData (const unsigned int n_shape_functions)
  :
  n_shape_functions (n_shape_functions)
{}


namespace
{
//---------------------------------------------------------------------------
// Utility method, which is used to determine the change of sign for
// the DoFs on the faces of the given cell.
//---------------------------------------------------------------------------

  /**
   * On noncartesian grids, the sign of the DoFs associated with the faces of
   * the elements has to be changed in some cases.  This procedure implements an
   * algorithm, which determines the DoFs, which need this sign change for a
   * given cell.
   */
  void
  get_face_sign_change_rt (const Triangulation<1>::cell_iterator &,
                           const unsigned int                     ,
                           std::vector<double>                   &face_sign)
  {
    // nothing to do in 1d
    std::fill (face_sign.begin (), face_sign.end (), 1.0);
  }



  void
  get_face_sign_change_rt (const Triangulation<2>::cell_iterator &cell,
                           const unsigned int                     dofs_per_face,
                           std::vector<double>                   &face_sign)
  {
    const unsigned int dim = 2;
    const unsigned int spacedim = 2;

    // Default is no sign
    // change. I.e. multiply by one.
    std::fill (face_sign.begin (), face_sign.end (), 1.0);

    for (unsigned int f = GeometryInfo<dim>::faces_per_cell / 2;
         f < GeometryInfo<dim>::faces_per_cell; ++f)
      {
        Triangulation<dim,spacedim>::face_iterator face = cell->face (f);
        if (!face->at_boundary ())
          {
            const unsigned int nn = cell->neighbor_face_no(f);

            if (nn < GeometryInfo<dim>::faces_per_cell / 2)
              for (unsigned int j = 0; j < dofs_per_face; ++j)
                {
                  Assert (f * dofs_per_face + j < face_sign.size(),
                          ExcInternalError());

//TODO: This is probably only going to work for those elements for which all dofs are face dofs
                  face_sign[f * dofs_per_face + j] = -1.0;
                }
          }
      }
  }



  void
  get_face_sign_change_rt (const Triangulation<3>::cell_iterator &/*cell*/,
                           const unsigned int                     /*dofs_per_face*/,
                           std::vector<double>                   &face_sign)
  {
    std::fill (face_sign.begin (), face_sign.end (), 1.0);
//TODO: think about what it would take here
  }

  void
  get_face_sign_change_nedelec (const Triangulation<1>::cell_iterator &/*cell*/,
                                const unsigned int                     /*dofs_per_face*/,
                                std::vector<double>                   &face_sign)
  {
    // nothing to do in 1d
    std::fill (face_sign.begin (), face_sign.end (), 1.0);
  }



  void
  get_face_sign_change_nedelec (const Triangulation<2>::cell_iterator &/*cell*/,
                                const unsigned int                     /*dofs_per_face*/,
                                std::vector<double>                   &face_sign)
  {
    std::fill (face_sign.begin (), face_sign.end (), 1.0);
//TODO: think about what it would take here
  }


  void
  get_face_sign_change_nedelec (const Triangulation<3>::cell_iterator &cell,
                                const unsigned int                     /*dofs_per_face*/,
                                std::vector<double>                   &face_sign)
  {
    const unsigned int dim = 3;
    std::fill (face_sign.begin (), face_sign.end (), 1.0);
//TODO: This is probably only going to work for those elements for which all dofs are face dofs
    for (unsigned int l = 0; l < GeometryInfo<dim>::lines_per_cell; ++l)
      if (!(cell->line_orientation (l)))
        face_sign[l] = -1.0;
  }
}



template <class POLY, int dim, int spacedim>
FE_PolyTensor_NPC<POLY,dim,spacedim>::FE_PolyTensor_NPC (const unsigned int degree,
                                                       const FiniteElementData<dim> &fe_data,
                                                       const std::vector<bool> &restriction_is_additive_flags,
                                                       const std::vector<ComponentMask> &nonzero_components)
  :
  FiniteElement<dim,spacedim> (fe_data,
                              restriction_is_additive_flags,
                              nonzero_components),
  poly_space(POLY(degree))
{
  cached_point(0) = -1;
  // Set up the table converting
  // components to base
  // components. Since we have only
  // one base element, everything
  // remains zero except the
  // component in the base, which is
  // the component itself
  for (unsigned int comp=0; comp<this->n_components() ; ++comp)
    this->component_to_base_table[comp].first.second = comp;

  if (dim == 1)
    piola_boundary = 0;
  else
    {
      if (dim == 2)
        {
          piola_boundary = 2 - ((degree==0) ? 1 : 0);
        }
      else
        {
          piola_boundary = 3*(degree+1) - ((degree==0) ? 1 : 0);
        }
    }
  // std::cout<<"degree: "<<degree<<" piola_boundary: "<<piola_boundary<<" poly_space: "
  // <<poly_space.n()<<std::endl;
  Assert(piola_boundary < poly_space.n(),
         ExcIndexRange (piola_boundary, 0, poly_space.n()));

}



template <class POLY, int dim, int spacedim>
double
FE_PolyTensor_NPC<POLY,dim,spacedim>::shape_value (const unsigned int, const Point<dim> &) const

{
  typedef    FiniteElement<dim,spacedim> FEE;
  Assert(false, typename FEE::ExcFENotPrimitive());
  return 0.;
}



template <class POLY, int dim, int spacedim>
double
FE_PolyTensor_NPC<POLY,dim,spacedim>::shape_value_component (const unsigned int i,
                                                            const Point<dim> &p,
                                                            const unsigned int component) const
{
  Assert (i<this->dofs_per_cell, ExcIndexRange(i,0,this->dofs_per_cell));
  Assert (component < dim, ExcIndexRange (component, 0, dim));

  if (cached_point != p || cached_values.size() == 0)
    {
      cached_point = p;
      cached_values.resize(poly_space.n());
      poly_space.compute(p, cached_values, cached_grads, cached_grad_grads);
    }

  double s = 0;
  if (inverse_node_matrix.n_cols() == 0)
    return cached_values[i][component];
  else
    for (unsigned int j=0; j<inverse_node_matrix.n_cols(); ++j)
      s += inverse_node_matrix(j,i) * cached_values[j][component];
  return s;
}



template <class POLY, int dim, int spacedim>
Tensor<1,dim>
FE_PolyTensor_NPC<POLY,dim,spacedim>::shape_grad (const unsigned int,
                                                 const Point<dim> &) const
{
  typedef    FiniteElement<dim,spacedim> FEE;
  Assert(false, typename FEE::ExcFENotPrimitive());
  return Tensor<1,dim>();
}



template <class POLY, int dim, int spacedim>
Tensor<1,dim>
FE_PolyTensor_NPC<POLY,dim,spacedim>::shape_grad_component (const unsigned int i,
                                                           const Point<dim> &p,
                                                           const unsigned int component) const
{
  Assert (i<this->dofs_per_cell, ExcIndexRange(i,0,this->dofs_per_cell));
  Assert (component < dim, ExcIndexRange (component, 0, dim));

  if (cached_point != p || cached_grads.size() == 0)
    {
      cached_point = p;
      cached_grads.resize(poly_space.n());
      poly_space.compute(p, cached_values, cached_grads, cached_grad_grads);
    }

  Tensor<1,dim> s;
  if (inverse_node_matrix.n_cols() == 0)
    return cached_grads[i][component];
  else
    for (unsigned int j=0; j<inverse_node_matrix.n_cols(); ++j)
      s += inverse_node_matrix(j,i) * cached_grads[j][component];

  return s;
}



template <class POLY, int dim, int spacedim>
Tensor<2,dim>
FE_PolyTensor_NPC<POLY,dim,spacedim>::shape_grad_grad (const unsigned int, const Point<dim> &) const
{
  typedef    FiniteElement<dim,spacedim> FEE;
  Assert(false, typename FEE::ExcFENotPrimitive());
  return Tensor<2,dim>();
}



template <class POLY, int dim, int spacedim>
Tensor<2,dim>
FE_PolyTensor_NPC<POLY,dim,spacedim>::shape_grad_grad_component (const unsigned int i,
    const Point<dim> &p,
    const unsigned int component) const
{
  Assert (i<this->dofs_per_cell, ExcIndexRange(i,0,this->dofs_per_cell));
  Assert (component < dim, ExcIndexRange (component, 0, dim));

  if (cached_point != p || cached_grad_grads.size() == 0)
    {
      cached_point = p;
      cached_grad_grads.resize(poly_space.n());
      poly_space.compute(p, cached_values, cached_grads, cached_grad_grads);
    }

  Tensor<2,dim> s;
  if (inverse_node_matrix.n_cols() == 0)
    return cached_grad_grads[i][component];
  else
    for (unsigned int j=0; j<inverse_node_matrix.n_cols(); ++j)
      s += inverse_node_matrix(i,j) * cached_grad_grads[j][component];

  return s;
}



//---------------------------------------------------------------------------
// Data field initialization
//---------------------------------------------------------------------------

template <class POLY, int dim, int spacedim>
typename Mapping<dim,spacedim>::InternalDataBase *
FE_PolyTensor_NPC<POLY,dim,spacedim>::get_data (
  const UpdateFlags      update_flags,
  const Mapping<dim,spacedim>    &mapping,
  const Quadrature<dim> &quadrature) const
{
  // generate a new data object and
  // initialize some fields
  InternalData *data = new InternalData(n_shape_functions);

  // check what needs to be
  // initialized only once and what
  // on every cell/face/subface we
  // visit
  data->update_once = update_once(update_flags);
  data->update_each = update_each(update_flags);
  data->update_flags = data->update_once | data->update_each;

  const UpdateFlags flags(data->update_flags);
  const unsigned int n_q_points = quadrature.size();

  // some scratch arrays
  std::vector<Tensor<1,dim> > values(0);
  std::vector<Tensor<2,dim> > grads(0);
  std::vector<Tensor<3,dim> > grad_grads(0);

  // initialize fields only if really
  // necessary. otherwise, don't
  // allocate memory
  if (flags & update_values)
    {
      values.resize (this->dofs_per_cell);
      data->shape_values.resize (this->dofs_per_cell);
      for (unsigned int i=0; i<this->dofs_per_cell; ++i)
        data->shape_values[i].resize (n_q_points);
    }

  if (flags & update_gradients)
    {
      grads.resize (this->dofs_per_cell);
      data->shape_grads.resize (this->dofs_per_cell);
      for (unsigned int i=0; i<this->dofs_per_cell; ++i)
        data->shape_grads[i].resize (n_q_points);
    }

  // if second derivatives through
  // finite differencing is required,
  // then initialize some objects for
  // that
  if (flags & update_hessians)
    {
//      grad_grads.resize (this->dofs_per_cell);
      data->initialize_2nd (this, mapping, quadrature);
    }

  // Compute shape function values
  // and derivatives on the reference
  // cell. Make sure, that for the
  // node values N_i holds
  // N_i(v_j)=\delta_ij for all basis
  // functions v_j
  if (flags & (update_values | update_gradients))
    for (unsigned int k=0; k<n_q_points; ++k)
      {
        poly_space.compute(quadrature.point(k),
                           values, grads, grad_grads);

        if (flags & update_values)
          {
            if (inverse_node_matrix.n_cols() == 0)
              for (unsigned int i=0; i<this->dofs_per_cell; ++i)
                data->shape_values[i][k] = values[i];
            else
              for (unsigned int i=0; i<this->dofs_per_cell; ++i)
                {
                  Tensor<1,dim> add_values;
                  for (unsigned int j=0; j<this->dofs_per_cell; ++j)
                    add_values += inverse_node_matrix(j,i) * values[j];
                  data->shape_values[i][k] = add_values;
                }
          }

        if (flags & update_gradients)
          {
            if (inverse_node_matrix.n_cols() == 0)
              for (unsigned int i=0; i<this->dofs_per_cell; ++i)
                data->shape_grads[i][k] = grads[i];
            else
              for (unsigned int i=0; i<this->dofs_per_cell; ++i)
                {
                  Tensor<2,dim> add_grads;
                  for (unsigned int j=0; j<this->dofs_per_cell; ++j)
                    add_grads += inverse_node_matrix(j,i) * grads[j];
                  data->shape_grads[i][k] = add_grads;
                }
          }

      }

  // ========== compute shapes =====
  data->shape_derivatives_x.resize(data->n_shape_functions * n_q_points);
  data->shape_derivatives_y.resize(data->n_shape_functions * n_q_points);
  data->shape_derivatives_z.resize(data->n_shape_functions * n_q_points);
  data->corner_derivatives.resize(data->n_shape_functions * GeometryInfo<dim>::vertices_per_cell);

  compute_shapes (quadrature.get_points(), *data);

  // Assert(false, ExcMessage("get_data_done"));
  return data;
}


template <class POLY, int dim, int spacedim>
void
FE_PolyTensor_NPC<POLY,dim,spacedim>::compute_shapes (const std::vector<Point<dim> > &unit_points,
                                         InternalData &data) const
{
    FE_PolyTensor_NPC<POLY,dim,spacedim>::compute_shapes_virtual(unit_points, data);
}

namespace internal
{
  namespace FE_PolyTensor_NPC
  {
    template <class POLY, int spacedim>
    void
    compute_shapes_virtual (const unsigned int            n_shape_functions,
                            const std::vector<Point<1> > &unit_points,
                            typename dealii::FE_PolyTensor_NPC<POLY,1,spacedim>::InternalData &data)
    {
      Assert(false, ExcNotImplemented());
    }

    template <class POLY, int spacedim>
    void
    compute_shapes_virtual (const unsigned int            n_shape_functions,
                            const std::vector<Point<2> > &unit_points,
                            typename dealii::FE_PolyTensor_NPC<POLY,2,spacedim>::InternalData &data)
    {
      Assert(false, ExcNotImplemented());
    }

    template <class POLY, int spacedim>
    void
    compute_shapes_virtual (const unsigned int            n_shape_functions,
                            const std::vector<Point<3> > &unit_points,
                            typename dealii::FE_PolyTensor_NPC<POLY,3,spacedim>::InternalData &data)
    {
      (void)n_shape_functions;
      const unsigned int n_points=unit_points.size();
      for (unsigned int k = 0 ; k < n_points ; ++k)
        {
          double x = unit_points[k](0);
          double y = unit_points[k](1);
          double z = unit_points[k](2);

          if (data.shape_derivatives_x.size()!=0)
            {
              Assert(data.shape_derivatives_x.size()==n_shape_functions*n_points,
                     ExcInternalError());
              // replace x with 1.0
              data.derivative(k,0,1)[0] = (y-1.)*(1.-z);
              data.derivative(k,1,1)[0] = (1.-y)*(1.-z);
              data.derivative(k,2,1)[0] = -y*(1.-z);
              data.derivative(k,3,1)[0] = y*(1.-z);
              data.derivative(k,4,1)[0] = (y-1.)*z;
              data.derivative(k,5,1)[0] = (1.-y)*z;
              data.derivative(k,6,1)[0] = -y*z;
              data.derivative(k,7,1)[0] = y*z;
              data.derivative(k,0,1)[1] = 0.0;
              data.derivative(k,1,1)[1] = -(1.-z);
              data.derivative(k,2,1)[1] = 0.0;
              data.derivative(k,3,1)[1] = (1.-z);
              data.derivative(k,4,1)[1] = 0.0;
              data.derivative(k,5,1)[1] = -z;
              data.derivative(k,6,1)[1] = 0.0;
              data.derivative(k,7,1)[1] = z;
              data.derivative(k,0,1)[2] = 0.0;
              data.derivative(k,1,1)[2] = (y-1.);
              data.derivative(k,2,1)[2] = 0.0;
              data.derivative(k,3,1)[2] = -y;
              data.derivative(k,4,1)[2] = 0.0;
              data.derivative(k,5,1)[2] = (1.-y);
              data.derivative(k,6,1)[2] = 0.0;
              data.derivative(k,7,1)[2] = y;
            }
          if (data.shape_derivatives_y.size()!=0)
            {
              Assert(data.shape_derivatives_y.size()==n_shape_functions*n_points,
                     ExcInternalError());
              // replace y with 1.0
              data.derivative(k,0,2)[0] = 0.0;
              data.derivative(k,1,2)[0] = 0.0;
              data.derivative(k,2,2)[0] = -(1.-z);
              data.derivative(k,3,2)[0] = (1.-z);
              data.derivative(k,4,2)[0] = 0.0;
              data.derivative(k,5,2)[0] = 0.0;
              data.derivative(k,6,2)[0] = -z;
              data.derivative(k,7,2)[0] = z;
              data.derivative(k,0,2)[1] = (x-1.)*(1.-z);
              data.derivative(k,1,2)[1] = -x*(1.-z);
              data.derivative(k,2,2)[1] = (1.-x)*(1.-z);
              data.derivative(k,3,2)[1] = x*(1.-z);
              data.derivative(k,4,2)[1] = (x-1.)*z;
              data.derivative(k,5,2)[1] = -x*z;
              data.derivative(k,6,2)[1] = (1.-x)*z;
              data.derivative(k,7,2)[1] = x*z;
              data.derivative(k,0,2)[2] = 0.0;
              data.derivative(k,1,2)[2] = 0.0;
              data.derivative(k,2,2)[2] = (x-1.);
              data.derivative(k,3,2)[2] = -x;
              data.derivative(k,4,2)[2] = 0.0;
              data.derivative(k,5,2)[2] = 0.0;
              data.derivative(k,6,2)[2] = (1.-x);
              data.derivative(k,7,2)[2] = x;
            }
          if (data.shape_derivatives_z.size()!=0)
            {
              Assert(data.shape_derivatives_z.size()==n_shape_functions*n_points,
                     ExcInternalError());
              // replace z with 1.0
              data.derivative(k,0,3)[0] = 0.0;
              data.derivative(k,1,3)[0] = 0.0;
              data.derivative(k,2,3)[0] = 0.0;
              data.derivative(k,3,3)[0] = 0.0;
              data.derivative(k,4,3)[0] = (y-1.);
              data.derivative(k,5,3)[0] = (1.-y);
              data.derivative(k,6,3)[0] = -y;
              data.derivative(k,7,3)[0] = y;
              data.derivative(k,0,3)[1] = 0.0;
              data.derivative(k,1,3)[1] = 0.0;
              data.derivative(k,2,3)[1] = 0.0;
              data.derivative(k,3,3)[1] = 0.0;
              data.derivative(k,4,3)[1] = (x-1.);
              data.derivative(k,5,3)[1] = -x;
              data.derivative(k,6,3)[1] = (1.-x);
              data.derivative(k,7,3)[1] = x;
              data.derivative(k,0,3)[2] = (x-1)*(1.-y);
              data.derivative(k,1,3)[2] = x*(y-1.);
              data.derivative(k,2,3)[2] = (x-1.)*y;
              data.derivative(k,3,3)[2] = -x*y;
              data.derivative(k,4,3)[2] = (1.-x)*(1.-y);
              data.derivative(k,5,3)[2] = x*(1.-y);
              data.derivative(k,6,3)[2] = (1.-x)*y;
              data.derivative(k,7,3)[2] = x*y; 
            }                        
        } // for each quadrature point

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
    } // compute_shapes_virtual_3D
  }
}


template <class POLY, int dim, int spacedim>
void
FE_PolyTensor_NPC<POLY,dim,spacedim>::
compute_shapes_virtual (const std::vector<Point<dim> > &unit_points,
                        InternalData &data) const
{
  internal::FE_PolyTensor_NPC::
  compute_shapes_virtual<POLY,spacedim> (n_shape_functions, unit_points, data);
}


template <class POLY, int dim, int spacedim>
void
FE_PolyTensor_NPC<POLY,dim,spacedim>::compute_mapping_support_points(
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
FE_PolyTensor_NPC<POLY,dim,spacedim>::fill_fe_values (
  const Mapping<dim,spacedim>                      &mapping,
  const typename Triangulation<dim,spacedim>::cell_iterator &cell,
  const Quadrature<dim>                            &quadrature,
  typename Mapping<dim,spacedim>::InternalDataBase &mapping_data,
  typename Mapping<dim,spacedim>::InternalDataBase &fedata,
  FEValuesData<dim,spacedim>                       &data,
  CellSimilarity::Similarity                  &cell_similarity) const
{



// for(unsigned int k=0; k<data.quadrature_points.size(); ++k)
// {
//        std::cout << data.quadrature_points[k] << "   JxW: "
//       << data.JxW_values[k]
//       <<std::endl;
// }

  // convert data object to internal
  // data for this class. fails with
  // an exception if that is not
  // possible
  Assert (dynamic_cast<InternalData *> (&fedata) != 0,
          ExcInternalError());
  InternalData &fe_data = static_cast<InternalData &> (fedata);

  const UpdateFlags flags(fe_data.current_update_flags());
  // const UpdateFlags flags(fe_data.update_each | fe_data.update_once);

  Assert (flags & update_quadrature_points, ExcInternalError());

  const unsigned int n_q_points = data.quadrature_points.size();

  Assert(!(flags & update_values) || fe_data.shape_values.size() == this->dofs_per_cell,
         ExcDimensionMismatch(fe_data.shape_values.size(), this->dofs_per_cell));
  Assert(!(flags & update_values) || fe_data.shape_values[0].size() == n_q_points,
         ExcDimensionMismatch(fe_data.shape_values[0].size(), n_q_points));

  // Create table with sign changes, due to the special structure of the RT elements.
  // TODO: Preliminary hack to demonstrate the overall prinicple!

  // Compute eventual sign changes depending on the neighborhood
  // between two faces.
  std::vector<double> sign_change (this->dofs_per_cell, 1.0);

  if (mapping_type == mapping_raviart_thomas)
    get_face_sign_change_rt (cell, this->dofs_per_face, sign_change);

  // for Piola mapping, the similarity
  // concept cannot be used because of
  // possible sign changes from one cell to
  // the next.
  if ( (mapping_type == mapping_piola) || (mapping_type == mapping_raviart_thomas) )
    if (cell_similarity == CellSimilarity::translation)
      cell_similarity = CellSimilarity::none;

  // ===================== non-mapping degree of freedoms =====================

  // some scratch arrays
  std::vector<Tensor<1,dim> > values(0);
  std::vector<Tensor<2,dim> > grads(0);
  std::vector<Tensor<3,dim> > grad_grads(0);

  if (flags & update_values)
    values.resize (this->dofs_per_cell);

  if (flags & update_gradients)
    grads.resize (this->dofs_per_cell);

  const Point<dim> center(cell->center());
  const double measure = cell->measure();
  const double h = std::pow(measure, 1.0/dim);

  if (flags & (update_values | update_gradients))
    for (unsigned int k=0; k<n_q_points; ++k)
      {

        poly_space.compute(Point<dim>(data.quadrature_points[k] - center)/h,
                           values, grads, grad_grads);

        for (unsigned int i=0; i<this->dofs_per_cell-piola_boundary; ++i)
          {
            const unsigned int first = data.shape_function_to_row_table[i * this->n_components() +
                                                                        this->get_nonzero_components(i).first_selected_component()];
            if (flags & update_values)
              {
                for (unsigned int d=0; d<dim; ++d)
                  data.shape_values(first+d,k) = sign_change[i] * values[i][d];
              }

            if (flags & update_gradients)
              {
                for (unsigned int d=0; d<dim; ++d)
                  data.shape_gradients[first+d][k] = sign_change[i] * grads[i][d]/h;
              }
          }
      }
  // ===================== piola-mapping degree of freedoms =====================

  unsigned int fe_degree = poly_space.degree()-1;

  // std::cout<<"degree = "<<fe_degree <<std::endl;

  // Assert(fe_degree == 0,
  //        ExcMessage("Only test on ATFull_0"));

  compute_mapping_support_points(cell, fe_data.mapping_support_points);
  const Tensor<1,spacedim> *supp_pts = &fe_data.mapping_support_points[0];

  // calculate coeffs for K2, K4 and K6
  Tensor<1, dim> g1;
  Tensor<1, dim> g2;
  Tensor<1, dim> g3;

  // K2 = A2 + B2 * y^hat + C2 * z^hat + D2 * y^hat * z^hat
  // A2 = K2(0,0)

  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g2[d] += supp_pts[k][d]*fe_data.corner_derivative(1,k)[1];
      g3[d] += supp_pts[k][d]*fe_data.corner_derivative(1,k)[2];
    }
  cross_product(g1, g2, g3);
  double A2 = g1.norm();

  // B2 = K2(1,0) - A2
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g2[d] += supp_pts[k][d] * fe_data.corner_derivative(3,k)[1];
      g3[d] += supp_pts[k][d] * fe_data.corner_derivative(3,k)[2];
    }
  cross_product(g1,g2,g3);
  double B2 = g1.norm() - A2;

  // C2 = K2(0,1) - A2
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g2[d] += supp_pts[k][d] * fe_data.corner_derivative(5,k)[1];
      g3[d] += supp_pts[k][d] * fe_data.corner_derivative(5,k)[2];
    }
  cross_product(g1,g2,g3);
  double C2 = g1.norm() - A2;

  // D2 = K2(1,1) - A2 - B2 - C2
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g2[d] += supp_pts[k][d] * fe_data.corner_derivative(7,k)[1];
      g3[d] += supp_pts[k][d] * fe_data.corner_derivative(7,k)[2];
    }
  cross_product(g1,g2,g3);
  double D2 = g1.norm() - A2 - B2 - C2; 
  double K2 = A2 + 0.5*B2 + 0.5*C2 + 0.25*D2;

  // K4 = A4 + B4 * x^hat + C4 * z^hat + D4 * x^hat * z^hat
  // A4 = K4(0,0)
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g1[d] += supp_pts[k][d] * fe_data.corner_derivative(2,k)[0];
      g3[d] += supp_pts[k][d] * fe_data.corner_derivative(2,k)[2];
    }
  cross_product(g2,g1,g3);
  double A4 = g2.norm();

  // B4 = K4(1,0) - A4
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g1[d] += supp_pts[k][d] * fe_data.corner_derivative(3,k)[0];
      g3[d] += supp_pts[k][d] * fe_data.corner_derivative(3,k)[2];
    }
  cross_product(g2,g1,g3);
  double B4 = g2.norm() - A4; 

  // C4 = K4(0,1) - A4  
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g1[d] += supp_pts[k][d] * fe_data.corner_derivative(6,k)[0];
      g3[d] += supp_pts[k][d] * fe_data.corner_derivative(6,k)[2];
    }
  cross_product(g2,g1,g3);
  double C4 = g2.norm() - A4; 

  // D4 = K4(1,1) - B - C - A
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g1[d] += supp_pts[k][d] * fe_data.corner_derivative(7,k)[0];
      g3[d] += supp_pts[k][d] * fe_data.corner_derivative(7,k)[2];
    }
  cross_product(g2,g1,g3);
  double D4 = g2.norm() - B4 - C4 - A4;
  double K4 = A4 + 0.5*(B4 + C4) + 0.25*D4;

  // K6 = A6 + B6 * x^hat + C6 * y^hat + D6 * x^hat * y^hat
  // A6 = K6(0,0)
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g1[d] += supp_pts[k][d] * fe_data.corner_derivative(4,k)[0];
      g2[d] += supp_pts[k][d] * fe_data.corner_derivative(4,k)[1];
    }
  cross_product(g3,g1,g2);
  double A6 = g3.norm();  

  // B6 ＝ K6（1,0) - A6
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g1[d] += supp_pts[k][d] * fe_data.corner_derivative(5,k)[0];
      g2[d] += supp_pts[k][d] * fe_data.corner_derivative(5,k)[1];
    }
  cross_product(g3,g1,g2);
  double B6 = g3.norm() - A6;    

  // C6 = K6(0,1) - A6
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g1[d] += supp_pts[k][d] * fe_data.corner_derivative(6,k)[0];
      g2[d] += supp_pts[k][d] * fe_data.corner_derivative(6,k)[1];
    }
  cross_product(g3,g1,g2);
  double C6 = g3.norm() - A6; 

  // D6 = K6(1,1) - A6 - B6 - C6
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g1[d] += supp_pts[k][d] * fe_data.corner_derivative(7,k)[0];
      g2[d] += supp_pts[k][d] * fe_data.corner_derivative(7,k)[1];
    }
  cross_product(g3,g1,g2);
  double D6 = g3.norm() - A6 - B6 - C6;   
  double K6 = A6 + 0.5*(B6+C6) + 0.25*D6; 

  // std::cout<<"cell "<<cell->index()<<"   K2="<<K2<<"   K4="<<K4<<"   K6="<<K6 <<std::endl;

  for (unsigned int i=this->dofs_per_cell-piola_boundary; i<this->dofs_per_cell; ++i)
    {
      const unsigned int first = data.shape_function_to_row_table[i * this->n_components() +
                                                                  this->get_nonzero_components(i).first_selected_component()];

      if (flags & update_values && cell_similarity != CellSimilarity::translation)
        {
          switch (mapping_type)
            {

            case mapping_raviart_thomas:
            case mapping_piola:
            {
              std::vector<Tensor<1,dim> > shape_values (n_q_points);
              if(fe_degree == 0) // ATFull_0
              {
                std::vector<Tensor<1,dim> > corrected_shape_values (n_q_points);

                for(unsigned int k=0; k<n_q_points; ++k)
                {
                  const double x = quadrature.get_points()[k](0);
                  const double y = quadrature.get_points()[k](1);
                  const double z = quadrature.get_points()[k](2);

                  Tensor<1,dim> u1;
                  Tensor<1,dim> u2;
                  const unsigned int dd1 = (i + piola_boundary - this->dofs_per_cell) % 3;
                  const unsigned int dd2 = (i + piola_boundary - this->dofs_per_cell + 1) % 3;

                  // u1[dd1]=fe_data.shape_values[i][k][dd1];
                  // u2[dd2]=fe_data.shape_values[i][k][dd2];

                  u1[dd1]=(dd1==0)?x:y;
                  u2[dd2]=(dd1==0)?-y:-z;

                  double a,b,c;

                  if(dd1 == 0)
                  {
                    a = 0.5*(B2+0.25*D2);
                    b = 0.5*(C2+0.25*D2);
                    c = 0.25*D2;
                    u1[dd1] *= A2 + B2*y + C2*z + D2*y*z;
                    u1[1] = y*(1.0-y)*(a + c*z);
                    u1[2] = z*(1.0-z)*(b + c*y);

                    a = -0.5*(B4+0.25*D4);
                    b = -0.5*(C4+0.25*D4);
                    c = -0.25*D4;
                    u2[dd2] *= A4 + B4*x + C4*z + D4*x*z;
                    u2[0] = x*(1.0-x)*(a + c*z);
                    u2[2] = z*(1.0-z)*(b + c*x);

                    corrected_shape_values[k] = u1/K2 + u2/K4;

                  }else if(dd1 == 1){
                    a = 0.5*(B4+0.25*D4);
                    b = 0.5*(C4+0.25*D4);
                    c = 0.25*D4;
                    u1[dd1] *= A4 + B4*x + C4*z + D4*x*z;
                    u1[0] = x*(1.0-x)*(a + c*z);
                    u1[2] = z*(1.0-z)*(b + c*x);

                    a = -0.5*(B6 + 0.25*D6);
                    b = -0.5*(C6 + 0.25*D6);
                    c = -0.25*D6;
                    u2[dd2] *= A6 + B6*x + C6*y + D6*x*y;
                    u2[0] = x*(1.0-x)*(a + c*y);
                    u2[1] = y*(1.0-y)*(b + c*x);

                    corrected_shape_values[k] = u1/K4 + u2/K6;
                  }else{
                    a = 0.5*(B6+0.25*D6);
                    b = 0.5*(C6+0.25*D6);
                    c = 0.25*D6;
                    u1[dd1] *= A6 + B6*x + C6*y + D6*x*y;
                    u1[0] = x*(1.0-x)*(a + c*y);
                    u1[1] = y*(1.0-y)*(b + c*x);

                    a = -0.5*(B2 + 0.25*D2);
                    b = -0.5*(C2 + 0.25*D2);
                    c = -0.25*D2;
                    u2[dd2] *= A2 + B2*y + C2*z + D2*y*z;
                    u2[1] = y*(1.0-y)*(a + c*z);
                    u2[2] = z*(1.0-z)*(b + c*y);

                    corrected_shape_values[k] = u1/K6 + u2/K2;
                  }
                }
                // instead of fe_data.shape_values[i], sent corrected 
                // shape values to piola mapping
                mapping.transform(corrected_shape_values, shape_values,
                                  mapping_data, mapping_piola);
              }
              else // if(degree > 1)
              {
                mapping.transform(fe_data.shape_values[i], shape_values,
                                  mapping_data, mapping_piola);
              }

              for (unsigned int k=0; k<n_q_points; ++k)
                  for (unsigned int d=0; d<dim; ++d)
                    data.shape_values(first+d,k)
                      = sign_change[i] * shape_values[k][d];
              break;
            }

            default:
              Assert(false, ExcNotImplemented());
            }
        }

      if (flags & update_gradients && cell_similarity != CellSimilarity::translation)
        {
          std::vector<Tensor<2, spacedim > > shape_grads1 (n_q_points);

          switch (mapping_type)
            {

            case mapping_raviart_thomas:
            case mapping_piola_gradient:
            {
              std::vector <Tensor<2,spacedim> > input_grads(fe_data.shape_grads[i].size());
              for (unsigned int k=0; k<input_grads.size(); ++k)
                input_grads[k] = fe_data.shape_grads[i][k];

              mapping.transform(input_grads, fe_data.shape_values[i], shape_grads1,
                                mapping_data, mapping_piola_gradient);

              for (unsigned int k=0; k<n_q_points; ++k)
                for (unsigned int d=0; d<dim; ++d)
                  data.shape_gradients[first+d][k] = sign_change[i] * shape_grads1[k][d];
              break;
            }

            default:
              Assert(false, ExcNotImplemented());
            }
        }
    }



  if (flags & update_hessians && cell_similarity != CellSimilarity::translation)
    this->compute_2nd (mapping, cell,
                       typename QProjector<dim>::DataSetDescriptor().cell(),
                       mapping_data, fe_data, data);
}



template <class POLY, int dim, int spacedim>
void
FE_PolyTensor_NPC<POLY,dim,spacedim>::fill_fe_face_values (
  const Mapping<dim,spacedim>                   &mapping,
  const typename Triangulation<dim,spacedim>::cell_iterator &cell,
  const unsigned int                    face,
  const Quadrature<dim-1>              &quadrature,
  typename Mapping<dim,spacedim>::InternalDataBase       &mapping_data,
  typename Mapping<dim,spacedim>::InternalDataBase       &fedata,
  FEValuesData<dim,spacedim>                    &data) const
{
  // convert data object to internal
  // data for this class. fails with
  // an exception if that is not
  // possible
  Assert (dynamic_cast<InternalData *> (&fedata) != 0,
          ExcInternalError());
  InternalData &fe_data = static_cast<InternalData &> (fedata);

  const UpdateFlags flags(fe_data.update_once | fe_data.update_each);

  Assert (flags & update_quadrature_points, ExcInternalError());
  const unsigned int n_q_points = data.quadrature_points.size();

  // offset determines which data set
  // to take (all data sets for all
  // faces are stored contiguously)

  const typename QProjector<dim>::DataSetDescriptor offset
    = QProjector<dim>::DataSetDescriptor::face (face,
                                                cell->face_orientation(face),
                                                cell->face_flip(face),
                                                cell->face_rotation(face),
                                                n_q_points);

  // std::cout<<"face = "<<face<<"   offset = "<<offset<<std::endl;
  // std::cout<<"fe_data.shape_values.size() = "<< fe_data.shape_values.size()<<std::endl;
  // std::cout<<"fe_data.shape_values[0].size() = "<< fe_data.shape_values[0].size()<<std::endl;
  // std::cout<<"n_q_points"<<n_q_points<<std::endl;
//TODO: Size assertions

// Create table with sign changes, due to the special structure of the RT elements.
// TODO: Preliminary hack to demonstrate the overall prinicple!

  // Compute eventual sign changes depending
  // on the neighborhood between two faces.
  std::vector<double> sign_change (this->dofs_per_cell, 1.0);

  if (mapping_type == mapping_raviart_thomas)
    get_face_sign_change_rt (cell, this->dofs_per_face, sign_change);

// ===================== non-mapping degree of freedoms =====================

  // some scratch arrays
  std::vector<Tensor<1,dim> > values(0);
  std::vector<Tensor<2,dim> > grads(0);
  std::vector<Tensor<3,dim> > grad_grads(0);

  if (flags & update_values)
    values.resize (this->dofs_per_cell);

  if (flags & update_gradients)
    grads.resize (this->dofs_per_cell);

  const Point<dim> center(cell->center());
  const double measure = cell->measure();
  const double h = std::pow(measure, 1.0/dim);

  if (flags & (update_values | update_gradients))
    for (unsigned int k=0; k<n_q_points; ++k)
      {
        poly_space.compute(Point<dim>(data.quadrature_points[k]-center)/h,
                           values, grads, grad_grads);

        for (unsigned int i=0; i<this->dofs_per_cell-piola_boundary; ++i)
          {
            const unsigned int first = data.shape_function_to_row_table[i * this->n_components() +
                                                                        this->get_nonzero_components(i).first_selected_component()];
            if (flags & update_values)
              {
                for (unsigned int d=0; d<dim; ++d)
                  data.shape_values(first+d,k) = sign_change[i] * values[i][d];
              }

            if (flags & update_gradients)
              {
                for (unsigned int d=0; d<dim; ++d)
                  data.shape_gradients[first+d][k] = sign_change[i] * grads[i][d]/h;
              }
          }
      }

// ===================== piola-mapping degree of freedoms =====================

  unsigned int fe_degree = poly_space.degree()-1;

  // std::cout<<"degree = "<<fe_degree <<std::endl;

  // Assert(fe_degree == 0,
  //        ExcMessage("Only test on ATFull_0"));

  compute_mapping_support_points(cell, fe_data.mapping_support_points);
  const Tensor<1,spacedim> *supp_pts = &fe_data.mapping_support_points[0];

  // calculate coeffs for K2, K4 and K6
  Tensor<1, dim> g1;
  Tensor<1, dim> g2;
  Tensor<1, dim> g3;

  // K2 = A2 + B2 * y^hat + C2 * z^hat + D2 * y^hat * z^hat
  // A2 = K2(0,0)

  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g2[d] += supp_pts[k][d]*fe_data.corner_derivative(1,k)[1];
      g3[d] += supp_pts[k][d]*fe_data.corner_derivative(1,k)[2];
    }
  cross_product(g1, g2, g3);
  double A2 = g1.norm();

  // B2 = K2(1,0) - A2
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g2[d] += supp_pts[k][d] * fe_data.corner_derivative(3,k)[1];
      g3[d] += supp_pts[k][d] * fe_data.corner_derivative(3,k)[2];
    }
  cross_product(g1,g2,g3);
  double B2 = g1.norm() - A2;

  // C2 = K2(0,1) - A2
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g2[d] += supp_pts[k][d] * fe_data.corner_derivative(5,k)[1];
      g3[d] += supp_pts[k][d] * fe_data.corner_derivative(5,k)[2];
    }
  cross_product(g1,g2,g3);
  double C2 = g1.norm() - A2;

  // D2 = K2(1,1) - A2 - B2 - C2
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g2[d] += supp_pts[k][d] * fe_data.corner_derivative(7,k)[1];
      g3[d] += supp_pts[k][d] * fe_data.corner_derivative(7,k)[2];
    }
  cross_product(g1,g2,g3);
  double D2 = g1.norm() - A2 - B2 - C2; 
  double K2 = A2 + 0.5*B2 + 0.5*C2 + 0.25*D2;

  // K4 = A4 + B4 * x^hat + C4 * z^hat + D4 * x^hat * z^hat
  // A4 = K4(0,0)
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g1[d] += supp_pts[k][d] * fe_data.corner_derivative(2,k)[0];
      g3[d] += supp_pts[k][d] * fe_data.corner_derivative(2,k)[2];
    }
  cross_product(g2,g1,g3);
  double A4 = g2.norm();

  // B4 = K4(1,0) - A4
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g1[d] += supp_pts[k][d] * fe_data.corner_derivative(3,k)[0];
      g3[d] += supp_pts[k][d] * fe_data.corner_derivative(3,k)[2];
    }
  cross_product(g2,g1,g3);
  double B4 = g2.norm() - A4; 

  // C4 = K4(0,1) - A4  
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g1[d] += supp_pts[k][d] * fe_data.corner_derivative(6,k)[0];
      g3[d] += supp_pts[k][d] * fe_data.corner_derivative(6,k)[2];
    }
  cross_product(g2,g1,g3);
  double C4 = g2.norm() - A4; 

  // D4 = K4(1,1) - B - C - A
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g1[d] += supp_pts[k][d] * fe_data.corner_derivative(7,k)[0];
      g3[d] += supp_pts[k][d] * fe_data.corner_derivative(7,k)[2];
    }
  cross_product(g2,g1,g3);
  double D4 = g2.norm() - B4 - C4 - A4;
  double K4 = A4 + 0.5*(B4 + C4) + 0.25*D4;

  // K6 = A6 + B6 * x^hat + C6 * y^hat + D6 * x^hat * y^hat
  // A6 = K6(0,0)
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g1[d] += supp_pts[k][d] * fe_data.corner_derivative(4,k)[0];
      g2[d] += supp_pts[k][d] * fe_data.corner_derivative(4,k)[1];
    }
  cross_product(g3,g1,g2);
  double A6 = g3.norm();  

  // B6 ＝ K6（1,0) - A6
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g1[d] += supp_pts[k][d] * fe_data.corner_derivative(5,k)[0];
      g2[d] += supp_pts[k][d] * fe_data.corner_derivative(5,k)[1];
    }
  cross_product(g3,g1,g2);
  double B6 = g3.norm() - A6;    

  // C6 = K6(0,1) - A6
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g1[d] += supp_pts[k][d] * fe_data.corner_derivative(6,k)[0];
      g2[d] += supp_pts[k][d] * fe_data.corner_derivative(6,k)[1];
    }
  cross_product(g3,g1,g2);
  double C6 = g3.norm() - A6; 

  // D6 = K6(1,1) - A6 - B6 - C6
  g1.clear();
  g2.clear();
  g3.clear();
  for(unsigned int k = 0; k<n_shape_functions; ++k)
    for(unsigned int d = 0; d<dim; ++d)
    {
      g1[d] += supp_pts[k][d] * fe_data.corner_derivative(7,k)[0];
      g2[d] += supp_pts[k][d] * fe_data.corner_derivative(7,k)[1];
    }
  cross_product(g3,g1,g2);
  double D6 = g3.norm() - A6 - B6 - C6;   
  double K6 = A6 + 0.5*(B6+C6) + 0.25*D6; 

  // std::cout<<"cell "<<cell->index()<<"  face "<<face<<"   K2="<<K2<<"   K4="<<K4<<"   K6="<<K6 <<std::endl;

  for (unsigned int i=this->dofs_per_cell-piola_boundary; i<this->dofs_per_cell; ++i)
    {
      const unsigned int first = data.shape_function_to_row_table[i * this->n_components() +
                                                                  this->get_nonzero_components(i).first_selected_component()];
      if (flags & update_values)
        {
          switch (mapping_type)
            {
            case mapping_raviart_thomas:
            case mapping_piola:
            {
              std::vector<Tensor<1,dim> > shape_values (n_q_points);

              if(fe_degree == 0) //ATFull_0
              {
                std::vector<Tensor<1,dim> > corrected_shape_values (n_q_points);

                Quadrature<dim> q_3d = QProjector<dim>::project_to_face(quadrature,face);

                for(unsigned int k=0; k<n_q_points; ++k)
                {

                  const double x = q_3d.get_points()[k](0);
                  const double y = q_3d.get_points()[k](1);
                  const double z = q_3d.get_points()[k](2);

                  Tensor<1,dim> u1;
                  Tensor<1,dim> u2;
                  const unsigned int dd1 = (i + piola_boundary - this->dofs_per_cell) % 3;
                  const unsigned int dd2 = (i + piola_boundary - this->dofs_per_cell + 1) % 3;

                  // u1[dd1]=fe_data.shape_values[i][k+offset][dd1];
                  // u2[dd2]=fe_data.shape_values[i][k+offset][dd2];

                  u1[dd1]=(dd1==0)?x:y;
                  u2[dd2]=(dd1==0)?-y:-z;

                  // std::cout<<"   cell "<<cell->index()
                  //          <<"   face "<<face
                  //          <<"   dd1  "<<dd1
                  //          <<"   q_pt("<<k
                  //          <<")=("<<x<<","<<y<<","<<z
                  //          <<")    "<<u1+u2
                  //          <<std::endl;

                  double a,b,c;

                  if(dd1 == 0)
                  {
                    a = 0.5*(B2+0.25*D2);
                    b = 0.5*(C2+0.25*D2);
                    c = 0.25*D2;
                    u1[dd1] *= A2 + B2*y + C2*z + D2*y*z;
                    u1[1] = y*(1.0-y)*(a + c*z);
                    u1[2] = z*(1.0-z)*(b + c*y);

                    a = -0.5*(B4+0.25*D4);
                    b = -0.5*(C4+0.25*D4);
                    c = -0.25*D4;
                    u2[dd2] *= A4 + B4*x + C4*z + D4*x*z;
                    u2[0] = x*(1.0-x)*(a + c*z);
                    u2[2] = z*(1.0-z)*(b + c*x);

                    corrected_shape_values[k] = u1/K2 + u2/K4;

                  }else if(dd1 == 1){
                    a = 0.5*(B4+0.25*D4);
                    b = 0.5*(C4+0.25*D4);
                    c = 0.25*D4;
                    u1[dd1] *= A4 + B4*x + C4*z + D4*x*z;
                    u1[0] = x*(1.0-x)*(a + c*z);
                    u1[2] = z*(1.0-z)*(b + c*x);

                    a = -0.5*(B6 + 0.25*D6);
                    b = -0.5*(C6 + 0.25*D6);
                    c = -0.25*D6;
                    u2[dd2] *= A6 + B6*x + C6*y + D6*x*y;
                    u2[0] = x*(1.0-x)*(a + c*y);
                    u2[1] = y*(1.0-y)*(b + c*x);

                    corrected_shape_values[k] = u1/K4 + u2/K6;
                  }else{
                    a = 0.5*(B6+0.25*D6);
                    b = 0.5*(C6+0.25*D6);
                    c = 0.25*D6;
                    u1[dd1] *= A6 + B6*x + C6*y + D6*x*y;
                    u1[0] = x*(1.0-x)*(a + c*y);
                    u1[1] = y*(1.0-y)*(b + c*x);

                    a = -0.5*(B2 + 0.25*D2);
                    b = -0.5*(C2 + 0.25*D2);
                    c = -0.25*D2;
                    u2[dd2] *= A2 + B2*y + C2*z + D2*y*z;
                    u2[1] = y*(1.0-y)*(a + c*z);
                    u2[2] = z*(1.0-z)*(b + c*y);

                    corrected_shape_values[k] = u1/K6 + u2/K2;
                  }
                }

                mapping.transform(corrected_shape_values,
                                  shape_values, mapping_data, mapping_piola);
              }
              else // if(degree > 1)
              {
                mapping.transform(make_slice(fe_data.shape_values[i], offset, n_q_points),
                                  shape_values, mapping_data, mapping_piola);
              }

              for (unsigned int k=0; k<n_q_points; ++k)
                for (unsigned int d=0; d<dim; ++d)
                  data.shape_values(first+d,k)
                    = sign_change[i] * shape_values[k][d];
              break;
            }
            default:
              Assert(false, ExcNotImplemented());
            }
        }

      if (flags & update_gradients)
        {
          std::vector<Tensor<2,dim> > shape_grads1 (n_q_points);
          //  std::vector<DerivativeForm<1,dim,spacedim> > shape_grads2 (n_q_points);
          switch (mapping_type)
            {
            case mapping_raviart_thomas:
            case mapping_piola_gradient:
            {
              std::vector <Tensor<2,spacedim> > input(fe_data.shape_grads[i].size());
              for (unsigned int k=0; k<input.size(); ++k)
                input[k] = fe_data.shape_grads[i][k];

              mapping.transform(make_slice(input, offset, n_q_points), shape_grads1,
                                mapping_data, mapping_piola_gradient);

              for (unsigned int k=0; k<n_q_points; ++k)
                for (unsigned int d=0; d<dim; ++d)
                  data.shape_gradients[first+d][k] = sign_change[i] * shape_grads1[k][d];

              break;
            }
            default:
              Assert(false, ExcNotImplemented());
            }
        }
    }

  if (flags & update_hessians)
    this->compute_2nd (mapping, cell, offset, mapping_data, fe_data, data);
}



template <class POLY, int dim, int spacedim>
void
FE_PolyTensor_NPC<POLY,dim,spacedim>::fill_fe_subface_values (
  const Mapping<dim,spacedim>                   &mapping,
  const typename Triangulation<dim,spacedim>::cell_iterator &cell,
  const unsigned int                    face,
  const unsigned int                    subface,
  const Quadrature<dim-1>              &quadrature,
  typename Mapping<dim,spacedim>::InternalDataBase       &mapping_data,
  typename Mapping<dim,spacedim>::InternalDataBase       &fedata,
  FEValuesData<dim,spacedim>                    &data) const
{
  // convert data object to internal
  // data for this class. fails with
  // an exception if that is not
  // possible
  Assert (dynamic_cast<InternalData *> (&fedata) != 0,
          ExcInternalError());
  InternalData &fe_data = static_cast<InternalData &> (fedata);

  const UpdateFlags flags(fe_data.update_once | fe_data.update_each);
  Assert (flags & update_quadrature_points, ExcInternalError());
  const unsigned int n_q_points = data.quadrature_points.size();

  // offset determines which data set
  // to take (all data sets for all
  // sub-faces are stored contiguously)
  const typename QProjector<dim>::DataSetDescriptor offset
    = QProjector<dim>::DataSetDescriptor::subface (face, subface,
                                                   cell->face_orientation(face),
                                                   cell->face_flip(face),
                                                   cell->face_rotation(face),
                                                   n_q_points,
                                                   cell->subface_case(face));



//   Assert(mapping_type == independent
//       || ( mapping_type == independent_on_cartesian
//            && dynamic_cast<const MappingCartesian<dim>*>(&mapping) != 0),
//       ExcNotImplemented());
//TODO: Size assertions

//TODO: Sign change for the face DoFs!

  // Compute eventual sign changes depending
  // on the neighborhood between two faces.
  std::vector<double> sign_change (this->dofs_per_cell, 1.0);

  if (mapping_type == mapping_raviart_thomas)
    get_face_sign_change_rt (cell, this->dofs_per_face, sign_change);

  // ===================== non-mapping degree of freedoms =====================

  // some scratch arrays
  std::vector<Tensor<1,dim> > values(0);
  std::vector<Tensor<2,dim> > grads(0);
  std::vector<Tensor<3,dim> > grad_grads(0);

  if (flags & update_values)
    values.resize (this->dofs_per_cell);

  if (flags & update_gradients)
    grads.resize (this->dofs_per_cell);

  const Point<dim> center(cell->center());
  const double measure = cell->measure();
  const double h = std::pow(measure, 1.0/dim);

  if (flags & (update_values | update_gradients))
    for (unsigned int k=0; k<n_q_points; ++k)
      {
        poly_space.compute(Point<dim>(data.quadrature_points[k]-center)/h,
                           values, grads, grad_grads);

        for (unsigned int i=0; i<this->dofs_per_cell-piola_boundary; ++i)
          {
            const unsigned int first = data.shape_function_to_row_table[i * this->n_components() +
                                                                        this->get_nonzero_components(i).first_selected_component()];
            if (flags & update_values)
              {
                for (unsigned int d=0; d<dim; ++d)
                  data.shape_values(first+d,k) = sign_change[i] * values[i][d];
              }

            if (flags & update_gradients)
              {
                for (unsigned int d=0; d<dim; ++d)
                  data.shape_gradients[first+d][k] = sign_change[i] * grads[i][d]/h;
              }
          }
      }




// ===================== piola-mapping degree of freedoms =====================

  for (unsigned int i=this->dofs_per_cell-piola_boundary; i<this->dofs_per_cell; ++i)
    {
      const unsigned int first = data.shape_function_to_row_table[i * this->n_components() +
                                                                  this->get_nonzero_components(i).first_selected_component()];

      if (flags & update_values)
        {
          switch (mapping_type)
            {
            case mapping_raviart_thomas:
            case mapping_piola:
            {
              std::vector<Tensor<1,dim> > shape_values (n_q_points);
              mapping.transform(make_slice(fe_data.shape_values[i], offset, n_q_points),
                                shape_values, mapping_data, mapping_piola);
              for (unsigned int k=0; k<n_q_points; ++k)
                for (unsigned int d=0; d<dim; ++d)
                  data.shape_values(first+d,k)
                    = sign_change[i] * shape_values[k][d];
              break;
            }
            default:
              Assert(false, ExcNotImplemented());
            }
        }

      if (flags & update_gradients)
        {
          std::vector<Tensor<2,dim> > shape_grads1 (n_q_points);

          switch (mapping_type)
            {

            case mapping_raviart_thomas:
            case mapping_piola_gradient:
            {

              std::vector <Tensor<2,spacedim> > input(fe_data.shape_grads[i].size());
              for (unsigned int k=0; k<input.size(); ++k)
                input[k] = fe_data.shape_grads[i][k];

              mapping.transform(make_slice(input, offset, n_q_points), shape_grads1,
                                mapping_data, mapping_piola_gradient);

              for (unsigned int k=0; k<n_q_points; ++k)
                for (unsigned int d=0; d<dim; ++d)
                  data.shape_gradients[first+d][k] = sign_change[i] * shape_grads1[k][d];

              break;
            }

            default:
              Assert(false, ExcNotImplemented());
            }
        }
    }

  if (flags & update_hessians)
    this->compute_2nd (mapping, cell, offset, mapping_data, fe_data, data);
}



template <class POLY, int dim, int spacedim>
UpdateFlags
FE_PolyTensor_NPC<POLY,dim,spacedim>::update_once (const UpdateFlags flags) const
{
  const bool values_once = (mapping_type == mapping_none);

  UpdateFlags out = update_default;

  if (values_once && (flags & update_values))
    out |= update_values;

  return out;
}


template <class POLY, int dim, int spacedim>
UpdateFlags
FE_PolyTensor_NPC<POLY,dim,spacedim>::update_each (const UpdateFlags flags) const
{
  UpdateFlags out = update_default;

  switch (mapping_type)
    {
    case mapping_raviart_thomas:
    case mapping_piola:
    {
      if (flags & update_values)
        out |= update_values | update_piola;

      if (flags & update_gradients)
        // out |= update_gradients | update_piola | update_covariant_transformation;
        out |= update_gradients | update_piola |
               update_covariant_transformation | update_jacobian_grads;

      if (flags & update_hessians)
        out |= update_hessians | update_piola | update_covariant_transformation;

      break;
    }


    case mapping_piola_gradient:
    {
      if (flags & update_values)
        out |= update_values | update_piola;

      if (flags & update_gradients)
        // out |= update_gradients | update_piola | update_covariant_transformation;
        out |= update_gradients | update_piola |
               update_covariant_transformation | update_jacobian_grads;


      if (flags & update_hessians)
        out |= update_hessians | update_piola | update_covariant_transformation;

      break;
    }

    default:
    {
      Assert (false, ExcNotImplemented());
    }
    }

  out |= update_quadrature_points;

  return out;
}



// explicit instantiations
#include "fe_poly_tensor_npc.inst"


DEAL_II_NAMESPACE_CLOSE
