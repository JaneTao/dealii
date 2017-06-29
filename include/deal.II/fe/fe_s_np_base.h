#ifndef __deal2__fe_s_np_base_h
#define __deal2__fe_s_np_base_h

#include <deal.II/base/config.h>
#include <deal.II/fe/fe_poly.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/derivative_form.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

DEAL_II_NAMESPACE_OPEN

template <class POLY, int dim=POLY::dimension, int spacedim=dim>
class FE_S_NP_Base : public FE_Poly<POLY,dim,spacedim>
{
public:

  FE_S_NP_Base (const POLY &poly_space,
             const FiniteElementData<dim> &fe_data,
             const std::vector<bool> &restriction_is_additive_flags);

  virtual void
  get_interpolation_matrix (const FiniteElement<dim,spacedim> &source,
                            FullMatrix<double>       &matrix) const;

  virtual void
  get_face_interpolation_matrix (const FiniteElement<dim,spacedim> &source,
                                 FullMatrix<double>       &matrix) const;

  virtual void
  get_subface_interpolation_matrix (const FiniteElement<dim,spacedim> &source,
                                    const unsigned int        subface,
                                    FullMatrix<double>       &matrix) const;

  virtual bool has_support_on_face (const unsigned int shape_index,
                                    const unsigned int face_index) const;

  virtual const FullMatrix<double> &
  get_restriction_matrix (const unsigned int child,
                          const RefinementCase<dim> &refinement_case=RefinementCase<dim>::isotropic_refinement) const;

  virtual const FullMatrix<double> &
  get_prolongation_matrix (const unsigned int child,
                           const RefinementCase<dim> &refinement_case=RefinementCase<dim>::isotropic_refinement) const;

  virtual
  unsigned int face_to_cell_index (const unsigned int face_dof_index,
                                   const unsigned int face,
                                   const bool face_orientation = true,
                                   const bool face_flip        = false,
                                   const bool face_rotation    = false) const;

  virtual std::pair<Table<2,bool>, std::vector<unsigned int> >
  get_constant_modes () const;

  virtual bool hp_constraints_are_implemented () const;

  virtual
  std::vector<std::pair<unsigned int, unsigned int> >
  hp_vertex_dof_identities (const FiniteElement<dim,spacedim> &fe_other) const;

  virtual
  std::vector<std::pair<unsigned int, unsigned int> >
  hp_line_dof_identities (const FiniteElement<dim,spacedim> &fe_other) const;

  virtual
  std::vector<std::pair<unsigned int, unsigned int> >
  hp_quad_dof_identities (const FiniteElement<dim,spacedim> &fe_other) const;

  virtual
  FiniteElementDomination::Domination
  compare_for_face_domination (const FiniteElement<dim,spacedim> &fe_other) const;
  //@}

protected:
  /**
   * Only for internal use. Its full name is @p get_dofs_per_object_vector
   * function and it creates the @p dofs_per_object vector that is needed
   * within the constructor to be passed to the constructor of @p
   * FiniteElementData.
   */
  static std::vector<unsigned int> get_dpo_vector(const unsigned int degree);

  void initialize (const std::vector<Point<1> > &support_points_1d);

  void initialize_constraints (const std::vector<Point<1> > &points);

  void initialize_unit_support_points (const std::vector<Point<1> > &points);

  void initialize_unit_face_support_points (const std::vector<Point<1> > &points);

  void initialize_quad_dof_index_permutation ();

  struct Implementation;

  friend struct FE_S_NP_Base<POLY,dim,spacedim>::Implementation;

  // ------------------------------------------------------------------
  // by Zhen Tao

public:
  
  virtual UpdateFlags update_once (const UpdateFlags flags) const;
 
  virtual UpdateFlags update_each (const UpdateFlags flags) const;

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

  // virtual void
  // fill_fe_subface_values (const Mapping<dim,spacedim> &mapping,
  //                         const typename Triangulation<dim,spacedim>::cell_iterator &cell,
  //                         const unsigned int                    face_no,
  //                         const unsigned int                    sub_no,
  //                         const Quadrature<dim-1>                &quadrature,
  //                         typename Mapping<dim,spacedim>::InternalDataBase      &mapping_internal,
  //                         typename Mapping<dim,spacedim>::InternalDataBase      &fe_internal,
  //                         FEValuesData<dim,spacedim> &data) const ;

  class InternalData : public FiniteElement<dim,spacedim>::InternalDataBase
  {
    public:
      InternalData(const unsigned int n_shape_functions);

      std::vector<Tensor<1,dim> > corner_derivatives;
      std::vector<double> corner_values;

      std::vector<Tensor<1,dim> > center_derivatives;
      std::vector<double> center_values;

      std::vector<Point<spacedim> > mapping_support_points;

      unsigned int n_shape_functions;

      Tensor<1,dim> corner_derivative (const unsigned int cn_nr,
        const unsigned int shape_nr) const;

      Tensor<1,dim> &corner_derivative (const unsigned int cn_nr,
        const unsigned int shape_nr);

      double corner_value (const unsigned int cn_nr,
        const unsigned int shape_nr) const;

      double &corner_value (const unsigned int cn_nr,
        const unsigned int shape_nr);

      Tensor<1,dim> center_derivative (const unsigned int cn_nr,
        const unsigned int shape_nr) const;

      Tensor<1,dim> &center_derivative (const unsigned int cn_nr,
        const unsigned int shape_nr);

      double center_value (const unsigned int cn_nr,
        const unsigned int shape_nr) const;

      double &center_value (const unsigned int cn_nr,
        const unsigned int shape_nr);

  };

  void compute_shapes (const std::vector<Point<dim> > &unit_points,
                       InternalData &data) const;

  virtual void compute_mapping_support_points(
    const typename Triangulation<dim,spacedim>::cell_iterator &cell,
    std::vector<Point<spacedim> > &a) const;


  virtual void compute_shapes_virtual (const std::vector<Point<dim> > &unit_points,
                                       InternalData &data) const;

  static const unsigned int n_shape_functions = GeometryInfo<dim>::vertices_per_cell;

private:
  mutable Threads::Mutex mutex;
};



template<class POLY, int dim, int spacedim>
inline
Tensor<1,dim>
FE_S_NP_Base<POLY,dim,spacedim>::InternalData::corner_derivative (const unsigned int cn_nr,
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
FE_S_NP_Base<POLY,dim,spacedim>::InternalData::corner_derivative (const unsigned int cn_nr,
  const unsigned int shape_nr)
{
  Assert(cn_nr*n_shape_functions + shape_nr < corner_derivatives.size(),
         ExcIndexRange(cn_nr*n_shape_functions + shape_nr, 0,
                       corner_derivatives.size()));
  return corner_derivatives [cn_nr*n_shape_functions + shape_nr];
}

template<class POLY, int dim, int spacedim>
inline
double
FE_S_NP_Base<POLY,dim,spacedim>::InternalData::corner_value (const unsigned int cn_nr,
  const unsigned int shape_nr) const
{
  Assert(cn_nr*n_shape_functions + shape_nr < corner_values.size(),
         ExcIndexRange(cn_nr*n_shape_functions + shape_nr, 0,
                       corner_values.size()));
  return corner_values [cn_nr*n_shape_functions + shape_nr];
}

template<class POLY, int dim, int spacedim>
inline
double &
FE_S_NP_Base<POLY,dim,spacedim>::InternalData::corner_value (const unsigned int cn_nr,
  const unsigned int shape_nr)
{
  Assert(cn_nr*n_shape_functions + shape_nr < corner_values.size(),
         ExcIndexRange(cn_nr*n_shape_functions + shape_nr, 0,
                       corner_values.size()));
  return corner_values [cn_nr*n_shape_functions + shape_nr];
}

template<class POLY, int dim, int spacedim>
inline
Tensor<1,dim>
FE_S_NP_Base<POLY,dim,spacedim>::InternalData::center_derivative (const unsigned int cn_nr,
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
FE_S_NP_Base<POLY,dim,spacedim>::InternalData::center_derivative (const unsigned int cn_nr,
  const unsigned int shape_nr)
{
  Assert(cn_nr*n_shape_functions + shape_nr < corner_derivatives.size(),
         ExcIndexRange(cn_nr*n_shape_functions + shape_nr, 0,
                       corner_derivatives.size()));
  return corner_derivatives [cn_nr*n_shape_functions + shape_nr];
}

template<class POLY, int dim, int spacedim>
inline
double
FE_S_NP_Base<POLY,dim,spacedim>::InternalData::center_value (const unsigned int cn_nr,
  const unsigned int shape_nr) const
{
  Assert(cn_nr*n_shape_functions + shape_nr < corner_values.size(),
         ExcIndexRange(cn_nr*n_shape_functions + shape_nr, 0,
                       corner_values.size()));
  return corner_values [cn_nr*n_shape_functions + shape_nr];
}

template<class POLY, int dim, int spacedim>
inline
double &
FE_S_NP_Base<POLY,dim,spacedim>::InternalData::center_value (const unsigned int cn_nr,
  const unsigned int shape_nr)
{
  Assert(cn_nr*n_shape_functions + shape_nr < corner_values.size(),
         ExcIndexRange(cn_nr*n_shape_functions + shape_nr, 0,
                       corner_values.size()));
  return corner_values [cn_nr*n_shape_functions + shape_nr];
}

/*@}*/

DEAL_II_NAMESPACE_CLOSE

#endif
