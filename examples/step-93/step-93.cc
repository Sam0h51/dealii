/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2024 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Sam Scheuerman, 2024
 */



#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/refinement.h>
#include <deal.II/fe/fe_series.h>
#include <deal.II/numerics/smoothness_estimator.h>

#include <deal.II/fe/fe_enriched.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/base/timer.h>


namespace Step93
{
  using namespace dealii;

  // We start by defining a function class for u-bar; it represents the heat
  // profile we want to match. There is code available for either a circular
  // step function or a Gaussian.
  //
  // This class has two member variables:
  //
  //   - center: a Point object representing the center of the function
  //   - radius: a double representing the radius of the circular step, or the
  //   standard deviation of the Gaussian
  template <int dim>
  class TargetFunction : public Function<dim>
  {
  public:
    // Below is the default constructor for TargetFunction
    TargetFunction(const unsigned int n_components = 1)
      : Function<dim>(n_components)
      , center(Point<dim>(0))
      , radius(.1){};
    // Next, we define an overloaded constructor for TargetFunction, so we can
    // choose to set the center and radius
    //
    // Parameters:
    //
    //   - center: a constant Point pointer used to set the member variable
    //   center
    //   - radius: a constant double used to set the member variable radius
    TargetFunction(const unsigned int n_components,
                   const Point<dim>  &center,
                   const double       radius = .3)
      : Function<dim>(n_components)
      , center(center)
      , radius(radius){};

    // The value() function returns the value of the target function at point p.
    // Note that this value depends on which component the function is being
    // evaluated on.
    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override;

  private:
    // See above
    const Point<dim> center;
    const double     radius;
  };

  // The value() function returns the value of the function at point p and
  // component index 'component'. In this case, if the component corresponds to
  // the solution u, then the function returns a value based on either a step
  // function or a Gaussian. If the component corresponds to l or c, the
  // function always returns 0.
  template <int dim>
  double TargetFunction<dim>::value(const Point<dim>  &p,
                                    const unsigned int component) const
  {
    if (component == 0)
      {
        // First we have code for a Gaussian target function
        /* return
         * std::exp(-((p-center).norm()*(p-center).norm())/(radius*radius)); */

        // Then we have code for a step target function
        if ((p - center).norm() <= radius)
          return 1;
        else
          return 0;
      }
    else
      return 0;
  }

  // The next class we define is one for a circular indicator function. Unlike
  // the target function, this does not need a component argument because we
  // have to manually address where it gets used in the code. Objects of this
  // function type correspond to the nonlocal dofs.
  //
  //  Parameters:
  //
  //   - center: a constant Point object giving the center of the indicator
  //   region
  //   - radius: the radius of the region
  template <int dim>
  class CircularIndicatorFunction : public Function<dim>
  {
  public:
    CircularIndicatorFunction()
    {}

    // Once again, we have an overloaded constructor that allows the center and
    // radius to be set at initialization.
    CircularIndicatorFunction(const Point<dim> &center, const double radius);

    virtual double
    value(const Point<dim>                   &p,
          [[maybe_unused]] const unsigned int component = 0) const override;

  private:
    const Point<dim> center;
    const double     radius;
  };

  template <int dim>
  CircularIndicatorFunction<dim>::CircularIndicatorFunction(
    const Point<dim> &center,
    const double      radius)
    : center(center)
    , radius(radius)
  {}

  template <int dim>
  double CircularIndicatorFunction<dim>::value(
    const Point<dim>                   &p,
    [[maybe_unused]] const unsigned int component) const
  {
    if ((center - p).norm() <= radius)
      return 1;
    else
      return 0;
  }

  // The main class is very similar to step-4. However, there are four new
  // member variables:
  //
  //   nonlocal_dofs:    this is a std::vector of dof indices that stores the
  //   dof index for the nonlocal dofs.
  //
  //   heat_centers:     this is a std::vector of Point objects, used to set the
  //   center of the CircularIndicatorFunction objects used as heat sources
  //   in this program.
  //
  //   heat_functions:   a std::vector of CircularIndicatorFunction objects;
  //   these are the heat sources.
  //
  //   target_function:  this is the function we want to match. We store it as a
  //   class variable because it is used both in assemble_system() and
  //   output_results().
  template <int dim>
  class Step93
  {
  public:
    Step93();

    // For the main class, we also define an overloaded constructor that allows
    // us to set the center of the target function at instantiation
    Step93(const Point<dim> &target_center);
    ~Step93();

    void run();


  private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void solve();
    void output_results() const;

    Triangulation<dim> triangulation;
    DoFHandler<dim>    dof_handler;

    hp::FECollection<dim> fe_collection;
    hp::QCollection<dim>  quadrature_collection;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> system_rhs;

    // The vector below stores the indices of the FE_DGQ elements which serve as
    // the nonlocal dofs
    std::vector<types::global_dof_index> nonlocal_dofs;

    // A second vector of Points stores the centers of the non-local
    // dofs. A circular step function centered at each point will be
    // interpolated later in the program. Since each step function is wider than
    // a single cell, we must have non-local dofs to capture this behavior.
    std::vector<Point<dim>> heat_centers;

    // Then we have a third vector of CircularIndicatorFunction<dim> objects,
    // which are used for assembling the system and again when outputting the
    // results.
    std::vector<CircularIndicatorFunction<dim>> heat_functions;

    // Finally, target_function is a constant variable which stores the
    // TargetFunction object used to construct the system rhs in
    // assemble_system(), and for output in output_results().
    const TargetFunction<dim> target_function;
  };

  // The default constructor below has several functions: it initializes
  // dof_handler and target_function, it constructs the hp finite element
  // collection, it generates a vector of center points for the heat functions,
  // and it generates a vector of CircularIndicatorFunctions objects which are
  // the heat functions.
  template <int dim>
  Step93<dim>::Step93()
    : dof_handler(triangulation)
    , target_function(3, Point<dim>())
  {
    // Here, we generate the finite element collection, which is basically a
    // list of all the possible finite elements we could use on each cell. This
    // collection has two elements: one FE_System that has two degree 2 FE_Q
    // elements and one FE_Nothing element, and one FE_System that has two
    // degree 2 FE_Q elements and one degree 0 FE_DGQ element.
    fe_collection.push_back(
      FESystem<dim>(FE_Q<dim>(2), 2, FE_Nothing<dim>(), 1));
    fe_collection.push_back(FESystem<dim>(FE_Q<dim>(2), 2, FE_DGQ<dim>(0), 1));

    // The quadrature collection is just one degree 3 QGauss element.
    quadrature_collection.push_back(QGauss<dim>(3));

    // Here, we create the vector of center points by enumerating the
    // vertices of a lattice, in this case the corners of a hypercube
    // centered at 0, with side length 1.
    double                           coordinate_points[2] = {.5, -.5};
    std::vector<std::vector<double>> center_points;

    for (int i = 0; i < std::pow(2, dim); ++i)
      {
        // Here we initialize a 1 x dim tensor to store the coordinates of the
        // point
        Tensor<1, dim, double> temp_point;
        for (int j = 0; j < dim; ++j)
          {
            // We create an 8-bit number to encode the current enumerated point
            u_int8_t binary_index = i;

            // We shift the index right by 7-j places
            binary_index = binary_index << (7 - j);
            // Then, we shift the index left by 7 places
            binary_index = binary_index >> 7;

            // Now the jth binary digit is in the 0th position, so binary_index
            // is either 0 or 1 depending on this digit. This determines if the
            // jth entry of temporary_point is 0.5 or -0.5
            temp_point[j] = coordinate_points[binary_index];
          }
        // Here, we use the Tensor point to initialize a Point object with the
        // same coordinates...
        const Point<dim> setting_point(temp_point);
        //...we add the point to the vector of center points...
        heat_centers.emplace_back(setting_point);
        //...and finally we create a CircularIndicatorFunction object with the
        // generated center, and add this to the vector of indicator functions.
        heat_functions.emplace_back(
          CircularIndicatorFunction<dim>(setting_point, 0.2));
      }
  }

  // The overloaded constructor below is the same as above, but with a parameter
  // for the center of the target function.
  template <int dim>
  Step93<dim>::Step93(const Point<dim> &target_center)
    : dof_handler(triangulation)
    , target_function(3, target_center)
  {
    // Here, we generate the finite element collection, which is basically a
    // list of all the possible finite elements we could use on each cell. This
    // collection has two elements: one FE_System that has two degree 2 FE_Q
    // elements and one FE_Nothing element, and one FE_System that has two
    // degree 2 FE_Q elements and one degree 0 FE_DGQ element.
    fe_collection.push_back(
      FESystem<dim>(FE_Q<dim>(2), 2, FE_Nothing<dim>(), 1));
    fe_collection.push_back(FESystem<dim>(FE_Q<dim>(2), 2, FE_DGQ<dim>(0), 1));

    // The quadrature collection is just on degree 3 QGauss element.
    quadrature_collection.push_back(QGauss<dim>(3));

    // Here, we create the vector of center points by enumerating the
    // vertices of a lattice, in this case the corners of a hypercube
    // centered at 0, with side length 1.
    double                           coordinate_points[2] = {.5, -.5};
    std::vector<std::vector<double>> center_points;

    for (int i = 0; i < std::pow(2, dim); ++i)
      {
        // Initialize a 1 x dim tensor to store the coordinates of the point
        Tensor<1, dim, double> temp_point;
        for (int j = 0; j < dim; ++j)
          {
            u_int8_t binary_index = i;

            // Shift right by 7-j places
            binary_index = binary_index << (7 - j);
            // Shift left by 7 places
            binary_index = binary_index >> 7;

            // Now the jth binary digit is in the 0th position, so binary_index
            // is either 0 or 1 depending on this digit. This determines if the
            // jth entry of temporary_point is 0.5 or -0.5
            temp_point[j] = coordinate_points[binary_index];
          }
        // Here, we use the Tensor point to initialize a Point object with the
        // same coordinates...
        const Point<dim> setting_point(temp_point);
        //...we add the point to the vector of center points...
        heat_centers.emplace_back(setting_point);
        //...and finally we create a CircularIndicatorFunction object with the
        // generated center, and add this to the vector of indicator functions.
        heat_functions.emplace_back(
          CircularIndicatorFunction<dim>(setting_point, 0.2));
      }
  }

  // The destructor below is the standard destructor
  template <int dim>
  Step93<dim>::~Step93()
  {
    dof_handler.clear();
  }

  // The make_grid() function makes a hypercube grid, see step-4
  template <int dim>
  void Step93<dim>::make_grid()
  {
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(7);

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl;
  }

  // The setup_system() function is similar to step-4, except we have to add a
  // few steps to prepare for the nonlocal dofs
  template <int dim>
  void Step93<dim>::setup_system()
  {
    // To start, we create an unsigned int variable to count how many non-local
    // dofs have been assigned
    unsigned int number_of_c_active_cells = 0;

    // Then, we loop over the cells and set the FE_System index
    // to 1, which corresponds to the system with 2 FE_Q elements and one FE_DGQ
    // element. We do this until we have enough dofs for each heat function.
    // Then, we call distribute_dofs.
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell->set_active_fe_index(1);
        ++number_of_c_active_cells;
        if (number_of_c_active_cells >= heat_functions.size())
          break;
      }
    dof_handler.distribute_dofs(fe_collection);

    // Once we've assigned dofs, the code block below counts the number of dofs
    // in the system, and outputs to the console
    const std::vector<types::global_dof_index> dofs_per_component =
      DoFTools::count_dofs_per_fe_component(dof_handler);
    const unsigned int dofs_per_u = dofs_per_component[0],
                       dofs_per_l = dofs_per_component[1],
                       dofs_per_c = dofs_per_component[2];
    std::cout << "Number of degrees of freedom: " << dofs_per_u << "+"
              << dofs_per_l << "+" << dofs_per_l << " = "
              << dofs_per_u + dofs_per_l + dofs_per_c << std::endl;

    // Here we make the hanging node constraints
    // TODO: Do I have these?
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(3),
                                             constraints);
    constraints.close();

    // Here we make the base sparsity pattern before we add entries
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

    // Next, we create a dummy hp_fe_values that only updates the quadrature
    // points and values. We use this for the set_nonnegative_c function, and to
    // determine where to increase the sparse matrix.
    hp::FEValues<dim> hp_fe_values(fe_collection,
                                   quadrature_collection,
                                   update_quadrature_points | update_values);

    // Now, we need to extract the indices of the finite elements which
    // correspond to the non-local dofs.

    // First, we make a component mask, which is false except for the third
    // component. This will extract only the dofs from the third component of
    // the FE system.
    const ComponentMask component_mask({false, false, true});

    // Next, we actually extract the dofs, and store them in an index set.
    const IndexSet non_zero_c =
      DoFTools::extract_dofs(dof_handler, component_mask);

    // Finally, we add each extracted index to the member array nonlocal_dofs
    for (auto non_local_index : non_zero_c)
      {
        nonlocal_dofs.push_back(non_local_index);
      }
    std::cout << "Number of nonlocal dofs: " << nonlocal_dofs.size()
              << std::endl;


    std::vector<types::global_dof_index> local_dof_indices;

    // The non-local dofs will need to interact with the second component of
    // the fe system, so we extract this scalar field to use below.
    const FEValuesExtractors::Scalar l(1);

    // Then, we loop over the cells, then over the quadrature points, and
    // finally over the indices, as if we were constructing a mass matrix.
    // However, what we instead do here is check 2 things. First, we check if
    // the quadrature point is within the radius of a circular indicator
    // function that represents my non-local dof. Then, we check if the l
    // component of my fe system is non-zero at this quadrature point. If both
    // of these are true, then we add an entry to the sparse matrix at the
    // (nonlocal dof index, l dof index) entry and the (l dof index, nonlocal
    // dof index) entry (because the block system we solve has both the
    // l-nonlocal interacting block and its transpose).
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        hp_fe_values.reinit(cell);

        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

        local_dof_indices.resize(fe_values.dofs_per_cell);

        cell->get_dof_indices(local_dof_indices);

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
            for (const unsigned int i : fe_values.dof_indices())
              {
                const double phi_i_l = fe_values[l].value(i, q_index);

                const Point<dim> q_point = fe_values.quadrature_point(q_index);

                for (unsigned int j = 0; j < heat_functions.size(); ++j)
                  {
                    // Within the loop, we need the code below to check if
                    // q_point is within the desired radius of the heat_center,
                    // and if phi_i_l is not zero at this quadrature point. If
                    // so, we add the requisite entries to the sparsity pattern.
                    if (heat_functions[j].value(q_point) > 1e-2 && phi_i_l != 0)
                      {
                        dsp.add(local_dof_indices[i], nonlocal_dofs[j]);
                        dsp.add(nonlocal_dofs[j], local_dof_indices[i]);
                      }
                  }
              }
          }
      }

    // The rest (below) is standard setup code, see step-4
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }

  // The assemble_system() function works very similar to how is does in other
  // tutorial programs (cf. step-4, step-6, step-8).
  // However, there is an additional component to constructing the
  // system matrix, because we need to handle the nonlocal dofs
  // manually.
  template <int dim>
  void Step93<dim>::assemble_system()
  {
    hp::FEValues<dim> hp_fe_values(fe_collection,
                                   quadrature_collection,
                                   update_values | update_gradients |
                                     update_quadrature_points |
                                     update_JxW_values);

    // First, we create a vector that stores the coefficients of u-bar in
    // the finite element basis. We use this vector to construct the rhs
    // component associated to the u derivatives of the Lagrangian.
    // Notice that we instantiate the vector and interpolate the
    // target function here, but this is NOT what goes in the RHS
    // vector. We first have to multiply this vector by the mass
    // matrix, which we do in the same loop that constructs the
    // system matrix. Note also that the mass matrix is a block
    // component of the system matrix.
    Vector<double> rhs_coefficients(dof_handler.n_dofs());
    VectorTools::interpolate(dof_handler, target_function, rhs_coefficients);

    FullMatrix<double> cell_matrix;
    Vector<double>     cell_rhs;

    std::vector<types::global_dof_index> local_dof_indices;

    const FEValuesExtractors::Scalar u(0);
    const FEValuesExtractors::Scalar l(1);
    const FEValuesExtractors::Scalar c(2);

    // Next, we do a standard loop setup for constructing the system matrix.
    // Note that we do this manually rather than using an existing
    // function, because we must handle the nonlocal dofs manually.
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();

        cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_rhs.reinit(dofs_per_cell);
        hp_fe_values.reinit(cell);

        cell_matrix = 0;
        cell_rhs    = 0;

        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

        local_dof_indices.resize(fe_values.dofs_per_cell);

        cell->get_dof_indices(local_dof_indices);



        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
            const double JxW = fe_values.JxW(q_index);
            for (const unsigned int i : fe_values.dof_indices())
              {
                const double phi_i_u = fe_values[u].value(i, q_index),
                             phi_i_l = fe_values[l].value(i, q_index);

                const Tensor<1, dim> grad_i_u =
                                       fe_values[u].gradient(i, q_index),
                                     grad_i_l =
                                       fe_values[l].gradient(i, q_index);

                for (const unsigned int j : fe_values.dof_indices())
                  {
                    const double phi_j_u = fe_values[u].value(j, q_index);

                    const Tensor<1, dim> grad_j_u =
                                           fe_values[u].gradient(j, q_index),
                                         grad_j_l =
                                           fe_values[l].gradient(j, q_index);

                    cell_matrix(i, j) += phi_i_u * phi_j_u * JxW;
                    cell_matrix(i, j) += -grad_i_u * grad_j_l * JxW;
                    cell_matrix(i, j) += -grad_i_l * grad_j_u * JxW;

                    cell_rhs(i) +=
                      (rhs_coefficients[local_dof_indices[j]] * // u bar
                       phi_j_u * phi_i_u * JxW);
                  }


                // Here, we deal with the nonlocal dofs. We start be getting an
                // actual quadrature point, rather than just an index.
                const Point<dim> q_point = fe_values.quadrature_point(q_index);

                // Next, we loop over the heat functions, adding the numeric
                // integral of each heat equation with each l component finite
                // element, at the appropriate indices (which we found in
                // setup_system()). Note that if we try to add 0 to an
                // uninitialized entry, there will not be a problem, but if we
                // try to add a nonzero value to an uninitialized entry we will
                // get an error. So, this part of the code checks that we
                // adjusted the dsp correctly.
                for (unsigned int j = 0; j < heat_functions.size(); ++j)
                  {
                    system_matrix.add(local_dof_indices[i],
                                      nonlocal_dofs[j],
                                      heat_functions[j].value(q_point) *
                                        phi_i_l * JxW);
                    system_matrix.add(nonlocal_dofs[j],
                                      local_dof_indices[i],
                                      heat_functions[j].value(q_point) *
                                        phi_i_l * JxW);
                  }
              }
          }

        // Here we apply constraints
        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }

    // Here we interpolate and apply boundary values.
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(3),
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);
  }

  // The solve() function works similar to how it does in step-6
  // and step-8, except we need to use a different solver because
  // the point we hope to find is a saddle point. So, the best
  // choice of solver is a SolverMinRes. This solver could be
  // improved with the use of preconditioners, but we don't do
  // that here for simplicity (see Possibilities for extension).
  // Included also is a commented direct solver, which can be
  // faster when the number of dofs is small.
  template <int dim>
  void Step93<dim>::solve()
  {
    // Notice that we also time how long this process takes (below).
    std::cout << "Beginning solve" << std::endl;
    Timer timer;

    SolverControl solver_control(5'000'000, 1e-6 * system_rhs.l2_norm());
    SolverMinRes<Vector<double>> solver(solver_control);

    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

    // Included also is code to use a direct solver rather than a CG solver
    // (below). Comment out the above block and uncomment this block to use it.
    /* SparseDirectUMFPACK direct_solver;
    direct_solver.initialize(system_matrix);
    direct_solver.vmult(solution, system_rhs); */

    // Finally, we stop the timer and output to the console.
    timer.stop();
    std::cout << "Wall time: " << timer.wall_time() << "s" << std::endl;
    std::cout << "Solved in " << solver_control.last_step()
              << " MINRES iterations." << std::endl;
  }

  // The output_results() function is a bit more robust for this program than
  // is typical. This is because, in order to visualize the heat functions,
  // we need to do extra work and interpolate them onto a mesh. We do this
  // by instantiating a new DoFHandler object and then using the helper
  // function VectorTools::interpolate().
  template <int dim>
  void Step93<dim>::output_results() const
  {
    // The beginning part is standard for vector valued problems (cf. step-8)
    std::vector<std::string> solution_names(1, "u");
    solution_names.emplace_back("l");
    solution_names.emplace_back("c");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(1, DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);


    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler,
                             solution,
                             solution_names,
                             interpretation);

    // Now, we create a new dummy dof handler to output the target function and
    // heat plate values
    DoFHandler<dim> toy_dof_handler(triangulation);

    // We use a finite element degree that matches what we used to solve for u
    // and l, although in reality this is an arbitrary choice
    const FE_Q<dim> toy_fe(2);
    toy_dof_handler.distribute_dofs(toy_fe);

    // To get started with the visualization, we need a vector which stores the
    // interpolated target function. We create the vector, interpolate it onto
    // the mesh, then add the data to our data_out object.
    Vector<double> target(toy_dof_handler.n_dofs());
    VectorTools::interpolate(toy_dof_handler, target_function, target);
    data_out.add_data_vector(toy_dof_handler, target, "TargetFunction");

    // Next, we create a vector which will store the sum of all the heat
    // functions.
    Vector<double> full_heat_profile(toy_dof_handler.n_dofs());

    // Then, we loop through the heat functions, create a vector to store
    // the interpolated data, call the interpolate() function to fill the
    // vector, multiply the interpolated data by the nonlocal dof value
    //(so that the heat plate is set to the correct temperature), and
    // then add this data to the data_out object. We also add this data
    // vector to the full_heat_profile vector, which will display all
    // the heat functions together at the end.
    for (unsigned int i = 0; i < heat_functions.size(); ++i)
      {
        Vector<double> hot_plate_i(toy_dof_handler.n_dofs());

        VectorTools::interpolate(toy_dof_handler,
                                 heat_functions[i],
                                 hot_plate_i);

        hot_plate_i *= solution[nonlocal_dofs[i]];

        // Here we iteratively name the heat function data
        std::string data_name = "Heat_Source_" + Utilities::int_to_string(i);

        data_out.add_data_vector(toy_dof_handler, hot_plate_i, data_name);

        // And then add each individual heat function to the total profile
        full_heat_profile += hot_plate_i;
      }

    // Once all the heat functions have been combined, we add them to the
    // data_out object
    data_out.add_data_vector(toy_dof_handler,
                             full_heat_profile,
                             "Full_Heat_Profile");

    data_out.build_patches();

    std::ofstream output("solution.vtu");
    data_out.write_vtu(output);

    // Finally, we output the temperature settings to the console
    std::cout << "The c coefficients were " << std::endl;
    for (long unsigned int i = 0; i < nonlocal_dofs.size(); ++i)
      {
        std::cout << "\tc" << i + 1 << ": " << solution[nonlocal_dofs[i]]
                  << std::endl;
      }
  }

  // The run() function runs through each step of the program, nothing new here
  template <int dim>
  void Step93<dim>::run()
  {
    make_grid();
    setup_system();
    assemble_system();
    solve();
    output_results();
  }
} // namespace Step93


int main()
{
  try
    {
      using namespace dealii;

      // In main(), we create integer which stores the dimension for the
      // problem. We use this in the loop below and in instantiating the Step93
      // object, so it gets stored in a variable
      const unsigned int dim = 2;

      // We also use the following small piece of code to construct a center for
      // the target function dependent on the dimension of the problem. The
      // center point for the target function will be (0.5), (0.5, 0.5), (0.5,
      // 0.5, 0.5), etc.
      Tensor<1, dim, double> center_setter;
      for (unsigned int i = 0; i < dim; ++i)
        center_setter[i] = 0.5;

      const Point<dim> center(center_setter);

      // Finally, we pass the center to the Step93 object and run the program.
      Step93::Step93<dim> heat_optimization_problem(center);
      heat_optimization_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
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
      std::cerr << std::endl
                << std::endl
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
