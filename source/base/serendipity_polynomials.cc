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

#include <deal.II/base/serendipity_polynomials.h>
#include <deal.II/base/polynomials_piecewise.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/table.h>

DEAL_II_NAMESPACE_OPEN



/* ------------------- SerendipityPolynomials -------------- */


namespace internal
{
  namespace
  {
    template <int dim>
    inline
    void compute_tensor_index(const unsigned int,
                              const FiniteElementData<dim> &fe_data,
                              unsigned int      ( &)[dim])
    {
      Assert(false, ExcNotImplemented());
    }

    inline
    void compute_tensor_index(const unsigned int n,
                              const FiniteElementData<1> &fe_data,
                              unsigned int       (&indices)[1])
    {
      indices[0] = n;
    }

    inline
    void compute_tensor_index(const unsigned int n,
                              const FiniteElementData<2> &fe_data,
                              unsigned int       (&indices)[2])
    {
      if( n < fe_data.first_line_index )
      {
        //vertex dofs
        indices[0] = n % 2;
        indices[1] = n / 2;
        return;
      }
      else if (n < fe_data.first_quad_index)
      {
        //line dofs
        if(n < fe_data.first_line_index + fe_data.dofs_per_line)
        {
          //left line
          indices[0] = 0;
          indices[1] = n - fe_data.first_line_index + 2;
          return;
        }
        if(n >= fe_data.first_line_index + fe_data.dofs_per_line
          && n < fe_data.first_line_index + 2*fe_data.dofs_per_line)
        {
          //right line
          indices[0] = 1;
          indices[1] = n - fe_data.first_line_index - fe_data.dofs_per_line + 2;
          return;
        }
        if(n >= fe_data.first_line_index + 2*fe_data.dofs_per_line
          && n < fe_data.first_line_index + 3*fe_data.dofs_per_line)
        {
          //bottom line
          indices[1] = 0;
          indices[0] = n - fe_data.first_line_index - 2*fe_data.dofs_per_line + 2;
          return;
        }
        if(n >= fe_data.first_line_index + 3*fe_data.dofs_per_line
          && n < fe_data.first_line_index + 4*fe_data.dofs_per_line)
        {
          //top line
          indices[1] = 1;
          indices[0] = n - fe_data.first_line_index - 3*fe_data.dofs_per_line + 2;
          return;
        }
      }
      else
      {
        //quad dofs
        unsigned int n_quad = n - fe_data.first_quad_index;
        unsigned int n_1d = fe_data.degree-2*2 + 1;

        unsigned int k=0;
        for (unsigned int iy=0; iy<n_1d; ++iy)
          if (n_quad < k+n_1d-iy)
            {
              indices[0] = n_quad-k+2;
              indices[1] = iy+2;
              return;
            }
          else
            k+=n_1d-iy;
      }
    }

    inline
    void compute_tensor_index(const unsigned int n,
                              const FiniteElementData<3> &fe_data,
                              unsigned int       (&indices)[3])
    {
      if( n < fe_data.first_line_index )
      {
        //vertex dofs
        indices[0] = n % 2;
        indices[1] = (unsigned int)((n - indices[0])/2) % 2;
        indices[2] = n / 4;
        return;
      }
      else if (n < fe_data.first_quad_index)
      {
        //line dofs
        const unsigned int ln = (n - fe_data.first_line_index)/ fe_data.dofs_per_line;
        const unsigned int li = (n - fe_data.first_line_index)% fe_data.dofs_per_line;

        if(ln == 0 || ln == 1 || ln == 2 || ln == 3)
        {
          indices[2] = 0;
        }
        else
        {
          if(ln == 4 || ln == 5 || ln == 6 || ln == 7)
            indices[2] = 1;
          else
            indices[2] = li + 2;
        }

        if(ln == 0 || ln == 4 || ln == 8 || ln == 10)
        {
          indices[0] = 0;
        }
        else
        {
          if(ln == 1 || ln == 5 || ln == 9 || ln == 11)
            indices[0] = 1;
          else
            indices[0] = li + 2;
        }

        if(ln == 2 || ln == 6 || ln == 8 || ln == 9)
        {
          indices[1] = 0;
        }
        else
        {
          if(ln == 3 || ln == 7 || ln == 10 || ln == 11)
            indices[1] = 1;
          else
            indices[1] = li + 2;
        }
        return;
      }
      else if(n < fe_data.first_hex_index)
      {
        //quad dofs
        const unsigned int fn = (n - fe_data.first_quad_index) / fe_data.dofs_per_quad;
        const unsigned int fi = (n - fe_data.first_quad_index) % fe_data.dofs_per_quad;
        const unsigned int n_1d = fe_data.degree-2*2 + 1;
        Assert(n_1d > 0 , ExcInternalError()); 
             
        if(fn == 0 || fn == 1)
        {
          indices[0] = fn % 2;

          unsigned int k=0;
           for (unsigned int iy=0; iy<n_1d; ++iy)
             if (fi < k+n_1d-iy)
              {
                indices[1] = fi-k+2;
                indices[2] = iy+2;
                return;
              }
             else
              k+=n_1d-iy;

        }

        if(fn == 2 || fn == 3)
        {
          indices[1] = fn % 2;

          unsigned int k=0;
           for (unsigned int iy=0; iy<n_1d; ++iy)
             if (fi < k+n_1d-iy)
              {
                indices[2] = fi-k+2;
                indices[0] = iy+2;
                return;
              }
             else
              k+=n_1d-iy;
        }

        if(fn == 4 || fn == 5)
        {
          indices[2] = fn % 2;

          unsigned int k=0;
           for (unsigned int iy=0; iy<n_1d; ++iy)
             if (fi < k+n_1d-iy)
              {
                indices[0] = fi-k+2;
                indices[1] = iy+2;
                return;
              }
             else
              k+=n_1d-iy;
        }
        else
        {
          Assert(false, ExcNotImplemented());
        }
      }
      else if(n < fe_data.dofs_per_cell)
      {
        //hex dofs
        const unsigned int hi = n - fe_data.first_hex_index;
        const unsigned int n_1d = fe_data.degree-2*2 + 1;
        Assert(n_1d > 0 , ExcInternalError()); 

        unsigned int k=0;
        for (unsigned int iz=0; iz<n_1d; ++iz)
          for (unsigned int iy=0; iy<n_1d-iz; ++iy)
            if (hi < k+n_1d-iy-iz)
              {
                indices[0] = hi-k+2;
                indices[1] = iy+2;
                indices[2] = iz+2;
                return;
              }
            else
              k += n_1d-iy-iz;      
      }
      else
      {
        Assert(false, ExcNotImplemented());
      }

    }

  }
}


template <int dim, typename POLY>
std::vector<unsigned int>
SerendipityPolynomials<dim,POLY>::get_dpo_vector(const unsigned int deg)
{
  AssertThrow(deg>0,ExcMessage("serendipity polynomials needs to be of degree > 0."));
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


template <int dim, typename POLY>
inline
void
SerendipityPolynomials<dim,POLY>::
compute_index (const unsigned int i,
               unsigned int       (&indices)[(dim > 0 ? dim : 1)]) const
{
  Assert (i < n_seren_pols, ExcInternalError());
  internal::compute_tensor_index(index_map[i], fe_data, indices);
}



template <int dim, typename POLY>
void
SerendipityPolynomials<dim,POLY>::output_indices(std::ostream &out) const
{
  unsigned int ix[dim];
  for (unsigned int i=0; i<n_seren_pols; ++i)
    {
      compute_index(i,ix);
      out << i << "\t";
      for (unsigned int d=0; d<dim; ++d)
        out << ix[d] << " ";
      out << std::endl;
    }
}



template <int dim, typename POLY>
void
SerendipityPolynomials<dim,POLY>::set_numbering(
  const std::vector<unsigned int> &renumber)
{
  Assert(renumber.size()==index_map.size(),
         ExcDimensionMismatch(renumber.size(), index_map.size()));

  index_map=renumber;
  for (unsigned int i=0; i<index_map.size(); ++i)
    index_map_inverse[index_map[i]]=i;
}



template <>
double
SerendipityPolynomials<0,Polynomials::Polynomial<double> >
::compute_value(const unsigned int,
                const Point<0> &) const
{
  Assert (false, ExcNotImplemented());
  return 0;
}



template <int dim, typename POLY>
double
SerendipityPolynomials<dim,POLY>::compute_value (const unsigned int i,
                                                   const Point<dim> &p) const
{
  Assert(dim>0, ExcNotImplemented());

  unsigned int indices[dim];
  compute_index (i, indices);

  double value=1.;
  for (unsigned int d=0; d<dim; ++d)
    value *= polynomials[indices[d]].value(p(d));

  return value;
}



template <int dim, typename POLY>
Tensor<1,dim>
SerendipityPolynomials<dim,POLY>::compute_grad (const unsigned int i,
                                                  const Point<dim> &p) const
{
  unsigned int indices[dim];
  compute_index (i, indices);

  // compute values and
  // uni-directional derivatives at
  // the given point in each
  // co-ordinate direction
  double v [dim][2];
  {
    std::vector<double> tmp (2);
    for (unsigned int d=0; d<dim; ++d)
      {
        polynomials[indices[d]].value (p(d), tmp);
        v[d][0] = tmp[0];
        v[d][1] = tmp[1];
      }
  }

  Tensor<1,dim> grad;
  for (unsigned int d=0; d<dim; ++d)
    {
      grad[d] = 1.;
      for (unsigned int x=0; x<dim; ++x)
        grad[d] *= v[x][d==x];
    }

  return grad;
}



template <int dim, typename POLY>
Tensor<2,dim>
SerendipityPolynomials<dim,POLY>::compute_grad_grad (const unsigned int i,
                                                       const Point<dim> &p) const
{
  unsigned int indices[dim];
  compute_index (i, indices);

  double v [dim][3];
  {
    std::vector<double> tmp (3);
    for (unsigned int d=0; d<dim; ++d)
      {
        polynomials[indices[d]].value (p(d), tmp);
        v[d][0] = tmp[0];
        v[d][1] = tmp[1];
        v[d][2] = tmp[2];
      }
  }

  Tensor<2,dim> grad_grad;
  for (unsigned int d1=0; d1<dim; ++d1)
    for (unsigned int d2=0; d2<dim; ++d2)
      {
        grad_grad[d1][d2] = 1.;
        for (unsigned int x=0; x<dim; ++x)
          {
            unsigned int derivative=0;
            if (d1==x || d2==x)
              {
                if (d1==d2)
                  derivative=2;
                else
                  derivative=1;
              }
            grad_grad[d1][d2] *= v[x][derivative];
          }
      }

  return grad_grad;
}




template <int dim, typename POLY>
void
SerendipityPolynomials<dim,POLY>::
compute (const Point<dim>            &p,
         std::vector<double>         &values,
         std::vector<Tensor<1,dim> > &grads,
         std::vector<Tensor<2,dim> > &grad_grads) const
{
  Assert (values.size()==n_seren_pols    || values.size()==0,
          ExcDimensionMismatch2(values.size(), n_seren_pols, 0));
  Assert (grads.size()==n_seren_pols     || grads.size()==0,
          ExcDimensionMismatch2(grads.size(), n_seren_pols, 0));
  Assert (grad_grads.size()==n_seren_pols|| grad_grads.size()==0,
          ExcDimensionMismatch2(grad_grads.size(), n_seren_pols, 0));

  const bool update_values     = (values.size() == n_seren_pols),
             update_grads      = (grads.size()==n_seren_pols),
             update_grad_grads = (grad_grads.size()==n_seren_pols);

  // check how many
  // values/derivatives we have to
  // compute
  unsigned int n_values_and_derivatives = 0;
  if (update_values)
    n_values_and_derivatives = 1;
  if (update_grads)
    n_values_and_derivatives = 2;
  if (update_grad_grads)
    n_values_and_derivatives = 3;


  // compute the values (and derivatives, if
  // necessary) of all polynomials at this
  // evaluation point. to avoid many
  // reallocation, use one std::vector for
  // polynomial evaluation and store the
  // result as Tensor<1,3> (that has enough
  // fields for any evaluation of values and
  // derivatives)
  Table<2,Tensor<1,3> > v(dim, polynomials.size());
  {
    std::vector<double> tmp (n_values_and_derivatives);
    for (unsigned int d=0; d<dim; ++d)
      for (unsigned int i=0; i<polynomials.size(); ++i)
        {
          polynomials[i].value(p(d), tmp);
          for (unsigned int e=0; e<n_values_and_derivatives; ++e)
            v(d,i)[e] = tmp[e];
        };
  }

  for (unsigned int i=0; i<n_seren_pols; ++i)
    {
      // first get the
      // one-dimensional indices of
      // this particular tensor
      // product polynomial
      unsigned int indices[dim];
      compute_index (i, indices);

      if (update_values)
        {
          values[i] = 1;
          for (unsigned int x=0; x<dim; ++x)
            values[i] *= v(x,indices[x])[0];
        }

      if (update_grads)
        for (unsigned int d=0; d<dim; ++d)
          {
            grads[i][d] = 1.;
            for (unsigned int x=0; x<dim; ++x)
              grads[i][d] *= v(x,indices[x])[d==x];
          }

      if (update_grad_grads)
        for (unsigned int d1=0; d1<dim; ++d1)
          for (unsigned int d2=0; d2<dim; ++d2)
            {
              grad_grads[i][d1][d2] = 1.;
              for (unsigned int x=0; x<dim; ++x)
                {
                  unsigned int derivative=0;
                  if (d1==x || d2==x)
                    {
                      if (d1==d2)
                        derivative=2;
                      else
                        derivative=1;
                    }
                  grad_grads[i][d1][d2]
                  *= v(x,indices[x])[derivative];
                }
            }
    }
}

/* ------------------- explicit instantiations -------------- */
template class SerendipityPolynomials<1,Polynomials::Polynomial<double> >;
template class SerendipityPolynomials<2,Polynomials::Polynomial<double> >;
template class SerendipityPolynomials<3,Polynomials::Polynomial<double> >;

template class SerendipityPolynomials<1,Polynomials::PiecewisePolynomial<double> >;
template class SerendipityPolynomials<2,Polynomials::PiecewisePolynomial<double> >;
template class SerendipityPolynomials<3,Polynomials::PiecewisePolynomial<double> >;

DEAL_II_NAMESPACE_CLOSE
