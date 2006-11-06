/* $Id$ */
/*    Copyright (C) 2006 by the deal.II authors */
/*    Author: Ivan Christov, Wolfgang Bangerth, Texas A&M University, 2006 */
/*                                                                */
/*    This file is subject to QPL and may not be  distributed     */
/*    without copyright and license information. Please refer     */
/*    to the file deal.II/doc/license.html for the text and       */
/*    further information on this license.                        */


				 // @sect3{Include files and global variables}

				 // For an explanation of the include
				 // files, the reader should refer to
				 // the example programs step-1
				 // through step-4. They are in the
				 // standard order, which is
				 // <code>base</code> --
				 // <code>lac</code> --
				 // <code>grid</code> --
				 // <code>dofs</code> --
				 // <code>fe</code> --
				 // <code>numerics</code> (since each
				 // of these categories roughly builds
				 // upon previous ones), then a few
				 // C++ headers for file, input/output
				 // and string streams.
#include <base/quadrature_lib.h>
#include <base/function.h>
#include <base/logstream.h>
#include <base/utilities.h>
#include <lac/vector.h>
#include <lac/full_matrix.h>
#include <lac/sparse_matrix.h>
#include <lac/solver_cg.h>
#include <lac/precondition.h>
#include <grid/tria.h>
#include <grid/grid_generator.h>
#include <grid/tria_accessor.h>
#include <grid/tria_iterator.h>
#include <dofs/dof_handler.h>
#include <dofs/dof_accessor.h>
#include <dofs/dof_tools.h>
#include <dofs/dof_constraints.h>
#include <fe/fe_q.h>
#include <fe/fe_values.h>
#include <numerics/vectors.h>
#include <numerics/matrices.h>
#include <numerics/data_out.h>

#include <fstream>
#include <iostream>


				 // The last step is as in all
				 // previous programs:
using namespace dealii;


				 // @sect3{The <code>SineGordonProblem</code> class template}

				 // The entire algorithm for solving the
				 // problem is encapsulated in this class. As
				 // in previous example programs, the class is
				 // declared with a template parameter, which
				 // is the spatial dimension, so that we can
				 // solve the sine-Gordon equation in one, two
				 // or three spatial dimensions. For more on
				 // the dimension-independent
				 // class-encapsulation of the problem, the
				 // reader should consult step-3 and step-4.
//TODO
template <int dim>
class SineGordonProblem 
{
  public:
    SineGordonProblem ();
    void run ();
    
  private:
    void make_grid_and_dofs ();
    void assemble_system ();
    void compute_nl_term (const Vector<double> &old_data, 
			  const Vector<double> &new_data,
			  Vector<double>       &nl_term) const;
    void compute_nl_matrix (const Vector<double> &old_data, 
			    const Vector<double> &new_data,
			    SparseMatrix<double> &nl_matrix) const;
    unsigned int solve ();
    void output_results (const unsigned int timestep_number);

    Triangulation<dim>   triangulation;
    FE_Q<dim>            fe;
    DoFHandler<dim>      dof_handler;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    
    double time, final_time, time_step;
    double theta;

    Vector<double>       solution, d_solution, old_solution;
    Vector<double>       massmatxvel;
    Vector<double>       system_rhs;

    static const unsigned int output_timestep_skip = 1;
    static const int n_global_refinements = 6;
};


				 // @sect3{Initial conditions}

				 // In the following two classes, we first
				 // implement the exact solution for 1d, 2d,
				 // and 3d mentioned in the introduction to
				 // this program. This space-time solution may
				 // be of independent interest if one wanted
				 // to test the accuracy of the program by
				 // comparing the numerical against the
				 // analytic solution (note however that the
				 // program uses a finite domain, whereas
				 // these are analytic solutions for an
				 // unbounded domain). This may, for example,
				 // be done using the
				 // VectorTools::integrate_difference
				 // function. Note again (as was already
				 // discussed in step-23) how we describe
				 // space-time functions as spatial functions
				 // that depend on a time variable that can be
				 // set and queried using the
				 // FunctionTime::set_time() and
				 // FunctionTime::get_time() member functions
				 // of the FunctionTime base class of the
				 // Function class.
template <int dim>
class ExactSolution : public Function<dim>
{
  public:
    ExactSolution (const unsigned int n_components = 1,
		   const double time = 0.) : Function<dim>(n_components, time) {};
    virtual double value (const Point<dim> &p,
			  const unsigned int component = 0) const;
};

template <int dim>
double ExactSolution<dim>::value (const Point<dim> &p,
				  const unsigned int /*component*/) const
{
  double t = this->get_time ();

  switch (dim)
    {
      case 1:
      {
	const double m = 0.5;
	const double c1 = 0.;
	const double c2 = 0.;
	return -4.*std::atan (m /
			      std::sqrt(1.-m*m) *
			      std::sin(std::sqrt(1.-m*m)*t+c2) /
			      std::cosh(m*p[0]+c1));
      }

      case 2:
      {
	const double theta  = deal_II_numbers::PI/4.;
	const double lambda  = 1.;
	const double a0  = 1.;
	const double s   = 1.;
	const double arg = p[0] * std::cos(theta) +
			   std::sin(theta) *
			   (p[1] * std::cosh(lambda) +
			    t * std::sinh(lambda));
	return 4.*std::atan(a0*std::exp(s*arg));
      }

      case 3:
      {
	double theta  = deal_II_numbers::PI;
	double phi = deal_II_numbers::PI;
	double tau = 1.;
	double c0  = 1.;
	double s   = 1.;
	double arg = p[0]*std::cos(theta) +
		     p[1]*std::sin(theta) * std::cos(phi) +
		     std::sin(theta) * std::sin(phi) *
		     (p[2]*std::cosh(tau)+t*std::sinh(tau));
	return 4.*std::atan(c0*std::exp(s*arg));
      }

      default:
	    Assert (false, ExcNotImplemented());
	    return -1e8;
    }
}

				 // The second part of this section is that we
				 // provide initial conditions. We are lazy
				 // (and cautious) and don't want to implement
				 // the same functions as above a second
				 // time. Rather, if we are queried for
				 // initial conditions, we create an object
				 // <code>ExactSolution</code>, set it to the
				 // correct time, and let it compute whatever
				 // values the exact solution has at that
				 // time:
template <int dim>
class InitialValues : public Function<dim>
{
  public:
    InitialValues (const unsigned int n_components = 1, 
		   const double time = 0.)
		    :
		    Function<dim>(n_components, time)
      {}
  
    virtual double value (const Point<dim> &p,
			  const unsigned int component = 0) const;
};

template <int dim>
double InitialValues<dim>::value (const Point<dim> &p,
				  const unsigned int component) const 
{   
  return ExactSolution<dim>(1, this->get_time()).value (p, component);
}



				 // @sect3{Implementation of the <code>SineGordonProblem</code> class}

				 // Let's move on to the implementation of the
				 // main class, as it implements the algorithm
				 // outlined in the introduction.

				 // @sect4{SineGordonProblem::SineGordonProblem}

				 // This is the constructor of the
				 // <code>SineGordonProblem</code> class. It
				 // specifies the desired polynomial degree of
				 // the finite elements, associates a
				 // <code>DoFHandler</code> to the
				 // <code>triangulation</code> object (just as
				 // in the example programs step-3 and
				 // step-4), initializes the current or
				 // initial time, the final time, the time
				 // step size, and the value of $\theta$ for
				 // the time stepping scheme. Since the
				 // solutions we compute here are
				 // time-periodic, the actual value of the
				 // start-time doesn't matter, and we choose
				 // it so that we start at an interesting
				 // time.
				 //
				 // Note that if we were to chose the explicit
				 // Euler time stepping scheme ($\theta = 0$),
				 // then we must pick a time step $k \le h$,
				 // otherwise the scheme is not stable and
				 // oscillations might arise in the
				 // solution. The Crank-Nicolson scheme
				 // ($\theta = \frac{1}{2}$) and the implicit
				 // Euler scheme ($\theta=1$) do not suffer
				 // from this deficiency, since they are
				 // unconditionally stable. However, even then
				 // the time step should be chosen to be on
				 // the order of $h$ in order to obtain a good
				 // solution. Since we know that our mesh
				 // results from the uniform subdivision of a
				 // rectangle, we can compute that time step
				 // easily; if we had a different domain, the
				 // technique in step-24 using
				 // GridTools::minimal_cell_diameter would
				 // work as well.
template <int dim>
SineGordonProblem<dim>::SineGordonProblem ()
		:
                fe (1),
		dof_handler (triangulation),
		time (-5.4414),
		final_time (2.7207),
		time_step (10*1./std::pow(2.,n_global_refinements)),
		theta (0.5)
{}

				 // @sect4{SineGordonProblem::make_grid_and_dofs}

				 // This function creates a rectangular grid
				 // in <code>dim</code> dimensions and refines
				 // it several times. Also, all matrix and
				 // vector members of the
				 // <code>SineGordonProblem</code> class are
				 // initialized to their appropriate sizes
				 // once the degrees of freedom have been
				 // assembled. Like step-24, we use the
				 // <code>MatrixCreator</code> class to
				 // generate a mass matrix $M$ and a Laplace
				 // matrix $A$ and store them in the
				 // appropriate variables for the remainder of
				 // the program's life.
template <int dim>
void SineGordonProblem<dim>::make_grid_and_dofs ()
{
  GridGenerator::hyper_cube (triangulation, -10, 10);
  triangulation.refine_global (n_global_refinements);
  
  std::cout << "   Number of active cells: "
	    << triangulation.n_active_cells()
	    << std::endl
	    << "   Total number of cells: "
	    << triangulation.n_cells()
	    << std::endl;

  dof_handler.distribute_dofs (fe);

  std::cout << "   Number of degrees of freedom: "
	    << dof_handler.n_dofs()
	    << std::endl;

  sparsity_pattern.reinit (dof_handler.n_dofs(), 
			   dof_handler.n_dofs(),
			   dof_handler.max_couplings_between_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
  sparsity_pattern.compress ();

  system_matrix.reinit  (sparsity_pattern);
  mass_matrix.reinit    (sparsity_pattern);
  laplace_matrix.reinit (sparsity_pattern);

  MatrixCreator::create_mass_matrix (dof_handler,
				     QGauss<dim>(3), 
				     mass_matrix);
  MatrixCreator::create_laplace_matrix (dof_handler,
					QGauss<dim>(3), 
					laplace_matrix);

  solution.reinit       (dof_handler.n_dofs());
  d_solution.reinit     (dof_handler.n_dofs());
  old_solution.reinit   (dof_handler.n_dofs());
  massmatxvel.reinit    (dof_handler.n_dofs());
  system_rhs.reinit     (dof_handler.n_dofs());
}

				 // @sect4{SineGordonProblem::assemble_system}

				 // This functions assembles the system matrix
				 // and right-hand side vector for each
				 // iteration of Newton's method. The reader
				 // should refer to the Introduction for the
				 // explicit formulas for the system matrix
				 // and right-hand side.
				 //
				 // Note that in each time step, we have to
				 // add up the various contributions to the
				 // matrix and right hand sides. In contrast
				 // to step-23 and step-24, this requires
				 // assembling a few more terms, since they
				 // depend on the solution of the previous
				 // time step or previous nonlinear step. We
				 // use the functions
				 // <code>compute_nl_matrix</code> and
				 // <code>compute_nl_term</code> to do this,
				 // while the present function provides the
				 // top-level logic.
template <int dim>
void SineGordonProblem<dim>::assemble_system () 
{  
				   // First we assemble the Jacobian
				   // matrix $F'_h(U^n_l)$, where
				   // $U^n_l$ is stored in the vector
				   // <code>solution</code> for
				   // convenience.
  system_matrix = 0;
  system_matrix.copy_from (mass_matrix);
  system_matrix.add (std::pow(time_step*theta,2), laplace_matrix);

  SparseMatrix<double> tmp_matrix (sparsity_pattern);
  compute_nl_matrix (old_solution, solution, tmp_matrix);
  system_matrix.add (-std::pow(time_step*theta,2), tmp_matrix);

				   // Then, we compute the right-hand
				   // side vector $-F_h(U^n_l)$.
  system_rhs = 0;

  tmp_matrix = 0;
  tmp_matrix.copy_from (mass_matrix);
  tmp_matrix.add (std::pow(time_step*theta,2), laplace_matrix);

  Vector<double> tmp_vector (solution.size());
  tmp_matrix.vmult (tmp_vector, solution);
  system_rhs += tmp_vector;

  tmp_matrix = 0;
  tmp_matrix.copy_from (mass_matrix);
  tmp_matrix.add (-std::pow(time_step,2)*theta*(1-theta), laplace_matrix);

  tmp_vector = 0;
  tmp_matrix.vmult (tmp_vector, old_solution);
  system_rhs -= tmp_vector;

  system_rhs.add (-time_step, massmatxvel);

  tmp_vector = 0;
  compute_nl_term (old_solution, solution, tmp_vector);
  system_rhs.add (std::pow(time_step,2)*theta, tmp_vector);

  system_rhs *= -1;
}

				 // @sect4{SineGordonProblem::compute_nl_term}

				 // This function computes the vector
				 // $S(\cdot,\cdot)$, which appears in the
				 // nonlinear term in the both equations of
				 // the split formulation. This function not
				 // only simplifies the repeated computation
				 // of this term, but it is also a fundamental
				 // part of the nonlinear iterative solver
				 // that we use when the time stepping is
				 // implicit (i.e. $\theta\ne 0$). Moreover,
				 // we must allow the function to receive as
				 // input an "old" and a "new" solution. These
				 // may not be the actual solutions of the
				 // problem stored in
				 // <code>old_solution</code> and
				 // <code>solution</code>, but are simply the
				 // two functions we linearize around. For the
				 // purposes of this function, let us call the
				 // first two arguments $w_{\mathrm{old}}$ and
				 // $w_{\mathrm{new}}$ in the documentation of
				 // this class below, respectively.
				 //
				 // As a side-note, it is perhaps worth
				 // investigating what order quadrature
				 // formula is best suited for this type of
				 // integration. Since $\sin(\cdot)$ is not a
				 // polynomial, there are probably no
				 // quadrature formulas that can integrate
				 // these terms exactly. It is usually
				 // sufficient to just make sure that the
				 // right hand side is integrated up to the
				 // same order of accuracy as the
				 // discretization scheme is, but it may be
				 // possible to improve on the constant in the
				 // asympotitic statement of convergence by
				 // choosing a more accurate quadrature
				 // formula.
template <int dim>
void SineGordonProblem<dim>::compute_nl_term (const Vector<double> &old_data,
					      const Vector<double> &new_data,
					      Vector<double>       &nl_term) const
{
  const QGauss<dim> quadrature_formula (3);
  FEValues<dim>     fe_values (fe, quadrature_formula, 
			       update_values |
			       update_JxW_values |
			       update_q_points);
  
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.n_quadrature_points;
  
  Vector<double> local_nl_term (dofs_per_cell);      
  std::vector<unsigned int> local_dof_indices (dofs_per_cell); 
  std::vector<double> old_data_values (n_q_points);
  std::vector<double> new_data_values (n_q_points);
  
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

  for (; cell!=endc; ++cell)
    { 
				       // Once we re-initialize our
				       // <code>FEValues</code> instantiation
				       // to the current cell, we make use of
				       // the <code>get_function_values</code>
				       // routine to get the values of the
				       // "old" data (presumably at
				       // $t=t_{n-1}$) and the "new" data
				       // (presumably at $t=t_n$) at the nodes
				       // of the chosen quadrature formula.
      fe_values.reinit (cell);
      fe_values.get_function_values (old_data, old_data_values);
      fe_values.get_function_values (new_data, new_data_values);
      
				       // Now, we can evaluate $\int_K
				       // \sin\left[\theta w_{\mathrm{new}} +
				       // (1-\theta) w_{\mathrm{old}}\right]
				       // \,\varphi_j\,\mathrm{d}x$ using the
				       // desired quadrature formula.
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
	for (unsigned int i=0; i<dofs_per_cell; ++i) 	   
	  local_nl_term(i) += (std::sin(theta * new_data_values[q_point] +
					(1-theta) * old_data_values[q_point]) *
			       fe_values.shape_value (i, q_point) *
			       fe_values.JxW (q_point));	    
      
				       // We conclude by adding up the
				       // contributions of the
				       // integrals over the cells to
				       // the global integral.
      cell->get_dof_indices (local_dof_indices);
      
      for (unsigned int i=0; i<dofs_per_cell; ++i)
	nl_term(local_dof_indices[i]) += local_nl_term(i);
      
      local_nl_term = 0;
    }
}

				 // @sect4{SineGordonProblem::compute_nl_matrix}

				 // This second function dealing with the
				 // nonlinear scheme computes the matrix
				 // $N(\cdot,\cdot)$ appearing in the
				 // nonlinear term in the Jacobian of
				 // $F(\cdot)$. Just as
				 // <code>compute_nl_term</code>, we must
				 // allow this function to receive as input an
				 // "old" and a "new" solution, which we again
				 // call $w_{\mathrm{old}}$ and
				 // $w_{\mathrm{new}}$ below, respectively.
template <int dim>
void SineGordonProblem<dim>::compute_nl_matrix (const Vector<double> &old_data, 
						const Vector<double> &new_data,
						SparseMatrix<double> &nl_matrix) const
{
  QGauss<dim>   quadrature_formula (3);
  FEValues<dim> fe_values (fe, quadrature_formula, 
			   update_values | update_JxW_values | update_q_points);
  
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.n_quadrature_points;
  
  FullMatrix<double> local_nl_matrix (dofs_per_cell, dofs_per_cell);
  std::vector<unsigned int> local_dof_indices (dofs_per_cell); 
  std::vector<double> old_data_values (n_q_points);
  std::vector<double> new_data_values (n_q_points);
  
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

  for (; cell!=endc; ++cell)
    { 
				       // Again, first we
				       // re-initialize our
				       // <code>FEValues</code>
				       // instantiation to the current
				       // cell.
      fe_values.reinit (cell);
      fe_values.get_function_values (old_data, old_data_values);
      fe_values.get_function_values (new_data, new_data_values);
      
				       // Then, we evaluate $\int_K
				       // \cos\left[\theta
				       // w_{\mathrm{new}} +
				       // (1-\theta)
				       // w_{\mathrm{old}}\right]\,
				       // \varphi_i\,
				       // \varphi_j\,\mathrm{d}x$
				       // using the desired quadrature
				       // formula.
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
	for (unsigned int i=0; i<dofs_per_cell; ++i) 
	  for (unsigned int j=0; j<dofs_per_cell; ++j) 
	    local_nl_matrix(i,j) += (std::cos(theta * new_data_values[q_point] +
					      (1-theta) * old_data_values[q_point]) *
				     fe_values.shape_value (i, q_point) *
				     fe_values.shape_value (j, q_point) *
				     fe_values.JxW (q_point));
      
				       // Finally, we add up the
				       // contributions of the
				       // integrals over the cells to
				       // the global integral.
      cell->get_dof_indices (local_dof_indices);

      for (unsigned int i=0; i<dofs_per_cell; ++i) 
	for (unsigned int j=0; j<dofs_per_cell; ++j)
	  nl_matrix.add(local_dof_indices[i], local_dof_indices[j], 
			local_nl_matrix(i,j));

      local_nl_matrix = 0;
    }
}



				 // @sect4{SineGordonProblem::solve}

				 // As discussed in the Introduction, this
				 // function uses the CG iterative solver on
				 // the linear system of equations resulting
				 // from the finite element spatial
				 // discretization of each iteration of
				 // Newton's method for the (nonlinear) first
				 // equation of the split formulation. The
				 // solution to the system is, in fact,
				 // $\delta U^n_l$ so it is stored in
				 // <code>d_solution</code> and used to update
				 // <code>solution</code> in the
				 // <code>run</code> function.
				 //
				 // Note that we re-set the solution update to
				 // zero before solving for it. This is not
				 // necessary: iterative solvers can start
				 // from any point and converge to the correct
				 // solution. If one has a good estimate about
				 // the solution of a linear system, it may be
				 // worthwhile to start from that vector, but
				 // as a general observation it is a fact that
				 // the starting point doesn't matter very
				 // much: it has to be a very very good guess
				 // to reduce the number of iterations by more
				 // than a few. It turns out that here, it
				 // using the previous nonlinear update as a
				 // starting point actually hurts and
				 // increases the number of iterations needed,
				 // so we simply set it to zero.
				 //
				 // The function returns the number of
				 // iterations it took to converge to a
				 // solution. This number will later be used
				 // to generate output on the screen showing
				 // how many iterations were needed in each
				 // nonlinear iteration.
template <int dim>
unsigned int
SineGordonProblem<dim>::solve () 
{
  SolverControl solver_control (1000, 1e-12*system_rhs.l2_norm());
  SolverCG<> cg (solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);
  
  d_solution = 0;
  cg.solve (system_matrix, d_solution,
	    system_rhs,
	    preconditioner);

  return solver_control.last_step();
}

				 // @sect4{SineGordonProblem::output_results}

				 // This function outputs the results to a
				 // file. It is pretty much identical to the
				 // respective functions in step-23 and
				 // step-24:
template <int dim>
void SineGordonProblem<dim>::output_results (const unsigned int timestep_number)
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "u");
  data_out.build_patches ();

  const std::string filename =  "solution-" +
				Utilities::int_to_string (timestep_number, 3) +
				".vtk";

  std::ofstream output (filename.c_str());
  data_out.write_vtk (output);
}

				 // @sect4{SineGordonProblem::run}

				 // This function has the top-level
				 // control over everything: it runs
				 // the (outer) time-stepping loop,
				 // the (inner) nonlinear-solver loop,
				 // and outputs the solution after each
				 // time step.
template <int dim>
void SineGordonProblem<dim>::run () 
{
  make_grid_and_dofs ();

				   // To aknowledge the initial condition, we
				   // must use the function $u_0(x)$. To this
				   // end, below we will create an object of
				   // type <code>InitialValues</code>; ote
				   // that when we create this object (which
				   // is derived from the
				   // <code>Function</code> class), we set its
				   // internal time variable to $t_0$, to
				   // indicate that the initial condition is a
				   // function of space and time evaluated at
				   // $t=t_0$.
				   //
				   // Then we produce $U^0$ by projecting
				   // $u_0(x)$ onto the grid using
				   // <code>VectorTools::project</code>. We
				   // have to use the same construct using
				   // hanging node constraints as in step-21:
				   // the VectorTools::project function
				   // requires a hanging node constraints
				   // object, but to be used we first need to
				   // close it:
    {
      ConstraintMatrix constraints;
      constraints.close();
      VectorTools::project (dof_handler,
			    constraints,
			    QGauss<dim>(3),
			    InitialValues<dim> (1, time),
			    solution);
    }

				   // For completeness, we output the
				   // zeroth time step to a file just
				   // like any other other time step.
  output_results (0);

				   // Now we perform the time
				   // stepping: at every time step we
				   // solve the matrix equation(s)
				   // corresponding to the finite
				   // element discretization of the
				   // problem, and then advance our
				   // solution according to the time
				   // stepping formulas we discussed
				   // in the Introduction.
  unsigned int timestep_number = 1;
  for (time+=time_step; time<=final_time; time+=time_step, ++timestep_number)
    {
      old_solution = solution;

      std::cout << std::endl
		<< "Time step #" << timestep_number << "; "
		<< "advancing to t = " << time << "." 
		<< std::endl;

				       // The first step in each time step is
				       // that we must solve the nonlinear
				       // equation in the split formulation
				       // via Newton's method --- i.e. solve
				       // for $\delta U^n_l$ then compute
				       // $U^n_{l+1}$ and so on. As stopping
				       // criterion for this nonlinear
				       // iteration we choose that
				       // $\|F_h(U^n_l)\|_2 \le 10^{-6}
				       // \|F_h(U^n_0)\|_2$. To this end, we
				       // need to record the norm of the
				       // residual in the first
				       // iteration.
				       //
				       // At the end of each iteration, we
				       // output to the console how many
				       // linear solver iterations it took
				       // us. When the loop below is done, we
				       // have (an approximation of) $U^n$.
      double initial_rhs_norm = 0.;
      bool first_iteration = true;
      do 
	{
	  assemble_system ();

	  if (first_iteration == true)
	    initial_rhs_norm = system_rhs.l2_norm();

	  const unsigned int n_iterations
	    = solve ();

	  solution += d_solution;

	  if (first_iteration == true)
	    std::cout << "    " << n_iterations;
	  else
	    std::cout << '+' << n_iterations;
	  first_iteration = false;
	} 
      while (system_rhs.l2_norm() > 1e-6 * initial_rhs_norm);

      std::cout << " CG iterations per nonlinear step."
		<< std::endl;
      
				       // Upon obtaining the solution to the
				       // first equation of the problem at
				       // $t=t_n$, we must update the
				       // auxilliary velocity variable
				       // $V^n$. However, we do not compute
				       // and store $V^n$ since it is not a
				       // quantity we use directly in the
				       // problem. Hence, for simplicity, we
				       // update $MV^n$ directly:
      Vector<double> tmp_vector (solution.size());
      laplace_matrix.vmult (tmp_vector, solution);
      massmatxvel.add (-time_step*theta, tmp_vector);

      tmp_vector = 0;
      laplace_matrix.vmult (tmp_vector, old_solution);
      massmatxvel.add (-time_step*(1-theta), tmp_vector);
      
      tmp_vector = 0;
      compute_nl_term (old_solution, solution, tmp_vector);
      massmatxvel.add (-time_step, tmp_vector);

				       // Oftentimes, in particular for fine
				       // meshes, we must pick the time step
				       // to be quite small in order for the
				       // scheme to be stable. Therefore,
				       // there are a lot of time steps during
				       // which "nothing interesting happens"
				       // in the solution. To improve overall
				       // efficiency --- in particular, speed
				       // up the program and save disk space
				       // --- we only output the solution
				       // every
				       // <code>output_timestep_skip</code>:
      if (timestep_number % output_timestep_skip == 0)
	output_results (timestep_number);      
    }
}

				 // @sect3{The <code>main</code> function}

				 // This is the main function of the
				 // program. It creates an object of
				 // top-level class and calls its
				 // principal function. Also, we
				 // supress some of the library output
				 // by setting
				 // <code>deallog.depth_console</code>
				 // to zero. Furthermore, if
				 // exceptions are thrown during the
				 // execution of the run method of the
				 // <code>SineGordonProblem</code>
				 // class, we catch and report them
				 // here. For more information about
				 // exceptions the reader should
				 // consult step-6.
int main () 
{
  try
    {
      deallog.depth_console (0);

      SineGordonProblem<1> sg_problem;
      sg_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      
      return 1;
    }
  catch (...) 
    {
      std::cerr << std::endl << std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      std::cerr << "Unknown exception!" << std::endl
		<< "Aborting!" << std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      return 1;
    }
  
  return 0;
}
