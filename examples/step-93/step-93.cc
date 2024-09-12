/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2023 by the deal.II authors
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
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
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
#include <deal.II/lac/solver_cg.h>
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

using namespace dealii;

template<int dim>
class Target_Function : public Function<dim>
{
  public:

    Target_Function(const unsigned int n_components = 1) : Function<dim>(n_components) {};

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
};

template <>
double Target_Function<2>::value(const Point<2> &p, const unsigned int component) const
{
  if(component == 0)
  {
    if((p-Point<2>(-.5, .5)).norm() <= 0.1) return 12;
    else return 0;
  }
  else return 0;
}

template<int dim>
class Heat_Plate_0 : public Function<dim>
{
  public:
    Heat_Plate_0(){}
    Heat_Plate_0(const Point<dim> &p);

    virtual double value(const Point<dim> &p, [[maybe_unused]] const unsigned int component = 0) const override;

  private:
    const Point<dim> heat_center;
};

template<int dim>
Heat_Plate_0<dim>::Heat_Plate_0(const Point<dim> &p) : heat_center(p)
{}

template<>
double Heat_Plate_0<2>::value(const Point<2> &p, [[maybe_unused]] const unsigned int component) const
{
  if((heat_center - p).norm() <= .2) return 1;
  else return 0;
}

template<int dim>
class Heat_Plate_1 : public Function<dim>
{
  public:
    Heat_Plate_1(){}
    Heat_Plate_1(const Point<dim> &p, const double radius);

    virtual double value(const Point<dim> &p, [[maybe_unused]] const unsigned int component = 0) const override;

  private:
    const Point<dim> heat_center;
    const double radius;
};

template<int dim>
Heat_Plate_1<dim>::Heat_Plate_1(const Point<dim> &p, const double radius) : heat_center(p), radius(radius)
{}

template<>
double Heat_Plate_1<2>::value(const Point<2> &p, [[maybe_unused]] const unsigned int component) const
{
  if((heat_center - p).norm() <= radius) return 1;
  else return 0;
}

template<int dim>
class Heat_Plate_2 : public Function<dim>
{
  public:
    Heat_Plate_2(){}
    Heat_Plate_2(const Point<dim> &p);

    virtual double value(const Point<dim> &p, [[maybe_unused]] const unsigned int component = 0) const override;

  private:
    const Point<dim> heat_center;
};

template<int dim>
Heat_Plate_2<dim>::Heat_Plate_2(const Point<dim> &p) : heat_center(p)
{}

template<>
double Heat_Plate_2<2>::value(const Point<2> &p, [[maybe_unused]] const unsigned int component) const
{
  if((heat_center - p).norm() <= .2) return 1;
  else return 0;
}

template<int dim>
class Heat_Plate_3 : public Function<dim>
{
  public:
    Heat_Plate_3(){}
    Heat_Plate_3(const Point<dim> &p);

    virtual double value(const Point<dim> &p, [[maybe_unused]] const unsigned int component = 0) const override;

  private:
    const Point<dim> heat_center;
};

template<int dim>
Heat_Plate_3<dim>::Heat_Plate_3(const Point<dim> &p) : heat_center(p)
{}

template<>
double Heat_Plate_3<2>::value(const Point<2> &p, [[maybe_unused]] const unsigned int component) const
{
  if((heat_center - p).norm() <= .2) return 1;
  else return 0;
}

template<int dim>
class Region_Indicator : public Function<dim>
{
  public:
    Region_Indicator() : center(0), radius(.1) {}
    Region_Indicator(const Point<dim> &set_center, const double set_radius);

    virtual double value(const Point<dim> &p, [[maybe_unused]] const unsigned int component = 0) const override;
  private:
    const Point<dim> center;
    const double radius;
};

template<int dim>
Region_Indicator<dim>::Region_Indicator(const Point<dim> &set_center, const double set_radius) :
  center(set_center),
  radius(set_radius)
{}

template<>
double Region_Indicator<2>::value(const Point<2> &p, [[maybe_unused]] const unsigned int component) const
{
  if((center - p).norm() <= radius) return 1;
  else return 0;
}


class Step3
{
public:
  Step3();
  ~Step3();

  void run();


private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  void set_nonnegative_c(hp::FEValues<2> &hp_fe_values);

  Triangulation<2> triangulation;
  DoFHandler<2> dof_handler;

  hp::FECollection<2> fe_collection;
  hp::QCollection<2> quadrature_collection;
  hp::QCollection<1> face_quadrature_collection;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;

  //Stores the indices of the DGQ elements which serve as the nonlocal dofs
  std::vector<types::global_dof_index> active_c_indices;

  //Four points which record the center of the non-local dofs. A circular step
  //function centered at each point will be interpolated later in the program.
  //Since each step function is wider than a single cell, we must have non-local
  //dofs to capture this behavior.
  const Point<2> heat_center_0,
                 heat_center_1,
                 heat_center_2,
                 heat_center_3;
};


Step3::Step3()
  : dof_handler(triangulation),
    heat_center_0(-.5, .5),
    heat_center_1(.5, .5),
    heat_center_2(-.5, -.5),
    heat_center_3(.5, -.5)
{
  fe_collection.push_back(FESystem<2>
                            (FE_Q<2>(2), 2, FE_Nothing<2>(), 1));
  fe_collection.push_back(FESystem<2>
                            (FE_Q<2>(2), 2, FE_DGQ<2>(0), 1));

  quadrature_collection.push_back(QGauss<2>(3));
  face_quadrature_collection.push_back(QGauss<1>(3));
}

Step3::~Step3()
{
  dof_handler.clear();
}



void Step3::make_grid()
{
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(7);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;
}




void Step3::setup_system()
{
  unsigned int number_of_c_active_cells = 0;   //Counts how many non-local dofs have been assigned

  //Here, I loop over the cells looking for ones which are not on the boundary. I do this
  //because I thought that boundary cells would be set to 0 with Dirichlet boundary conditions.
  //Since I am using FE_DGQ elements, I don't actually need to worry about this, so this could
  //just be a loop over the first n active cells.
  //
  //Once a cell is found, I set the FE_System index to 1, which corresponds to the system with
  //2 FE_Q elements and one FE_DGQ element. Then, I call distribute_dofs.
  for(const auto &cell : dof_handler.active_cell_iterators()){
    // if(!cell->at_boundary()){
    cell->set_active_fe_index(1);
    ++number_of_c_active_cells;
    if(number_of_c_active_cells >=4) break;
    // }
  }
  dof_handler.distribute_dofs(fe_collection);

  const std::vector<types::global_dof_index> dofs_per_component = 
      DoFTools::count_dofs_per_fe_component(dof_handler);
  const unsigned int  dofs_per_u = dofs_per_component[0],
                      dofs_per_l = dofs_per_component[1],
                      dofs_per_c = dofs_per_component[2];
  std::cout << "Number of degrees of freedom: " << dofs_per_u << "+"
            << dofs_per_l << "+"
            << dofs_per_l << " = "
            << dofs_per_u + dofs_per_l + dofs_per_c
            << std::endl;

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<2>(3),
                                           constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

  //This is a dummy hp_fe_values that only updates the quadrature points and
  //values. I use this for the set_nonnegative_c function, and to determine
  //where to increase the sparse matrix.
  hp::FEValues<2> hp_fe_values(fe_collection,
                               quadrature_collection,
                               update_quadrature_points | update_values);


  //Here I call a function to record the indices of the non-local dofs
  //set_nonnegative_c(hp_fe_values);

  //Now, I need to extract the indices of the finite elements which correspond
  //to the non-local dofs.

  //First, I make a component mask, which is false except for the third component.
  //This will extract only the dofs from the third component of the FE system.
  const ComponentMask component_mask({false, false, true});

  //Next, I actually extract the dofs, and store them in an index set.
  const IndexSet non_zero_c =  DoFTools::extract_dofs(dof_handler, component_mask);

  //Finally, I add each extracted index to the member array active_c_indices
  for (auto non_local_index : non_zero_c)
  {
    active_c_indices.push_back(non_local_index);
    std::cout << "An index: " << non_local_index << std::endl;
  }


  std::vector<types::global_dof_index> local_dof_indices;

  //The non-local dofs will need to interact with the second component of
  //the fe system, so I extract this scalar field to use below.
  const FEValuesExtractors::Scalar l(1);

  //This loops over the cells, then over the quadrature points, and finally
  //over the indices, as if I were constructing a mass matrix. However, what
  //I instead do here is check 2 things. First, I check if the quadrature
  //point is within the radius of a circular indicator function that represents
  //my non-local dof. Then, I check if the l component of my fe system is non-zero
  //at this quadrature point. If both of these are true, then I add an entry to the
  //sparse matrix at the (nonlocal dof index, l dof index) entry and the
  //(l dof index, nonlocal dof index) entry (because the block system I solve has both
  //the l-nonlocal interacting block and its transpose).
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      hp_fe_values.reinit(cell);

      const FEValues<2> &fe_values = hp_fe_values.get_present_fe_values();

      local_dof_indices.resize(fe_values.dofs_per_cell);

      cell->get_dof_indices(local_dof_indices);

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          for (const unsigned int i : fe_values.dof_indices()){
            const double /* phi_i_u = fe_values[u].value(i, q_index), */
                         phi_i_l = fe_values[l].value(i, q_index)/*,
                         phi_i_c = fe_values[c].value(i, q_index)*/;

            const Point<2> q_point = fe_values.quadrature_point(q_index);

            //Checks if q_point is within the desired radius of the heat_center,
            //and if phi_i_l is not zero at this quadrature point.
            if((heat_center_0 - q_point).norm() <= 0.2 && phi_i_l != 0)
            {
              dsp.add(local_dof_indices[i], active_c_indices[0]);
              dsp.add(active_c_indices[0], local_dof_indices[i]);
            }

            if((heat_center_1 - q_point).norm() <= 0.2 && phi_i_l != 0)
            {
              dsp.add(local_dof_indices[i], active_c_indices[1]);
              dsp.add(active_c_indices[1], local_dof_indices[i]);
            }

            if((heat_center_2 - q_point).norm() <= 0.2 && phi_i_l != 0)
            {
              dsp.add(local_dof_indices[i], active_c_indices[2]);
              dsp.add(active_c_indices[2], local_dof_indices[i]);
            }

            if((heat_center_3 - q_point).norm() <= 0.2 && phi_i_l != 0)
            {
              dsp.add(local_dof_indices[i], active_c_indices[3]);
              dsp.add(active_c_indices[3], local_dof_indices[i]);
            }
          } 
        }
    }




  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}

//This function loops through all the cells until it finds the cells with fe index 1.
//It then loops over all dof indices and then all quadrature points of that dof. The
//goal is to determine the global index of the n dofs that we want to correspond to
//the non-local degrees of freedom. I pass in an hp::FEValues object so that I can
//evaluate to shape functions at the quadrature points, and look for where they are
//non-zero.
void Step3::set_nonnegative_c(hp::FEValues<2> &hp_fe_values)
{
  for(const auto &cell : dof_handler.active_cell_iterators())
  {
    if(cell->active_fe_index() == 1)
    {
      //cell->set_active_fe_index(1);

      hp_fe_values.reinit(cell);

      const FEValues<2> &fe_values = hp_fe_values.get_present_fe_values();

      //I only care about the index of the DGQ elements, so we extract this scalar field
      const FEValuesExtractors::Scalar c(2);

      //I use this value to add up all values at each quadrature point. If this is ever
      //non-zero, then I have found an index where the DGQ element is set to 1.
      double total_value;

      std::cout << "DoFs on non-boundary cell is " << fe_values.dofs_per_cell << std::endl;

      std::vector<types::global_dof_index> local_dof_indices(cell->get_fe().dofs_per_cell);

      cell->get_dof_indices(local_dof_indices);

      for(const unsigned int dof_index : fe_values.dof_indices())
      {
        total_value = 0;
        // std::cout << "dof_index is " << dof_index << std::endl;

        //Loop over the quadratur points, add the absolute value of the DGQ element
        //at each quadrature point to total_value
        for(const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          /* std::cout << "\t Value at the " 
                    << q_index 
                    << "th quadrature point is " 
                    << fe_values[c].value(dof_index, q_index)
                    << std::endl; */
          total_value += std::abs(fe_values[c].value(dof_index, q_index));
        }
        //Check if total_value is not 0, and store the index in the member std::vector
        //if so.
        if(total_value >= 1e-6) active_c_indices.push_back(local_dof_indices[dof_index]);
      }
      
      //For this program, I only need 4 non-local dofs, so we stop the loop once the indices
      //for these have been found.
      if(active_c_indices.size() >= 4) break;
    }
  }

  if(active_c_indices.size() != 4) std::cout << "An error occured while determining c indices" 
                                         << std::endl;
  std::cout << "c indices were "
                                         << active_c_indices[0]
                                         << " "
                                         << active_c_indices[1]
                                         << " "
                                         << active_c_indices[2]
                                         << " "
                                         << active_c_indices[3]
                                         << std::endl;
}



void Step3::assemble_system()
{
  hp::FEValues<2> hp_fe_values(fe_collection,
                               quadrature_collection,
                               update_values | update_gradients |
                               update_quadrature_points |
                               update_JxW_values);

  //The function I want to match for the optimization problem
  Target_Function<2> target_function(3);
  //const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  Vector<double> rhs_coefficients(dof_handler.n_dofs());

  std::cout << "Pre Interpolate" << std::endl;
  VectorTools::interpolate(dof_handler, target_function, rhs_coefficients);
  std::cout << "Post Interpolate" << std::endl;

  FullMatrix<double> cell_matrix;
  Vector<double>     cell_rhs;

  std::vector<types::global_dof_index> local_dof_indices;

  const FEValuesExtractors::Scalar u(0);
  const FEValuesExtractors::Scalar l(1);
  const FEValuesExtractors::Scalar c(2);

  //NOTE: THIS IS NOT CURRENTLY USED, IGNORE THESE TWO LINES
  //If desired, create a region of interest for error constraint
  const Point<2> center_of_interest(.5, .5);
  const Region_Indicator<2> region_indicator(center_of_interest, .2);

  //set_nonnegative_c(hp_fe_values);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();

      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);
      hp_fe_values.reinit(cell);

      cell_matrix = 0;
      cell_rhs    = 0;

      const FEValues<2> &fe_values = hp_fe_values.get_present_fe_values();

      local_dof_indices.resize(fe_values.dofs_per_cell);

      cell->get_dof_indices(local_dof_indices);



      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          const double JxW = fe_values.JxW(q_index);
          for (const unsigned int i : fe_values.dof_indices()){
            const double phi_i_u = fe_values[u].value(i, q_index),
                         phi_i_l = fe_values[l].value(i, q_index)/*,
                         phi_i_c = fe_values[c].value(i, q_index)*/;

            const Tensor<1, 2> grad_i_u = fe_values[u].gradient(i, q_index),
                               grad_i_l = fe_values[l].gradient(i, q_index);

            for (const unsigned int j : fe_values.dof_indices()){
              const double phi_j_u = fe_values[u].value(j, q_index)/* ,
                           phi_j_l = fe_values[l].value(j, q_index) *//*,
                           phi_j_c = fe_values[c].value(j, q_index)*/;

              const Tensor<1, 2> grad_j_u = fe_values[u].gradient(j, q_index),
                                 grad_j_l = fe_values[l].gradient(j, q_index);
              
              cell_matrix(i, j) += 2*phi_i_u*phi_j_u*JxW;
              cell_matrix(i, j) += -grad_i_u*grad_j_l*JxW;
              cell_matrix(i, j) += -grad_i_l*grad_j_u*JxW;
              //cell_matrix(i, j) += phi_i_l*phi_j_c*JxW;
              //cell_matrix(i, j) += phi_i_c*phi_j_l*JxW;

              cell_rhs(i) += 2*(rhs_coefficients[local_dof_indices[j]]*         //u bar
                              phi_j_u*
                              phi_i_u*
                              JxW);
            }

            

            const Point<2> q_point = fe_values.quadrature_point(q_index);

            const Heat_Plate_0<2> heat_plate_0(heat_center_0);
            const Heat_Plate_1<2> heat_plate_1(heat_center_1, 0.2);
            const Heat_Plate_2<2> heat_plate_2(heat_center_2);
            const Heat_Plate_3<2> heat_plate_3(heat_center_3);

            
            system_matrix.add(local_dof_indices[i], active_c_indices[0],
                              heat_plate_0.value(q_point)*phi_i_l*JxW);
            system_matrix.add(active_c_indices[0], local_dof_indices[i],
                              heat_plate_0.value(q_point)*phi_i_l*JxW);
            

            system_matrix.add(local_dof_indices[i], active_c_indices[1],
                              heat_plate_1.value(q_point)*phi_i_l*JxW);
            system_matrix.add(active_c_indices[1], local_dof_indices[i],
                              heat_plate_1.value(q_point)*phi_i_l*JxW);

            system_matrix.add(local_dof_indices[i], active_c_indices[2],
                              heat_plate_2.value(q_point)*phi_i_l*JxW);
            system_matrix.add(active_c_indices[2], local_dof_indices[i],
                              heat_plate_2.value(q_point)*phi_i_l*JxW);

            system_matrix.add(local_dof_indices[i], active_c_indices[3],
                              heat_plate_3.value(q_point)*phi_i_l*JxW);
            system_matrix.add(active_c_indices[3], local_dof_indices[i],
                              heat_plate_3.value(q_point)*phi_i_l*JxW);
          } 
        }

      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);

      /* for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          system_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix(i, j));

      for (const unsigned int i : fe_values.dof_indices())
        system_rhs(local_dof_indices[i]) += cell_rhs(i); */
    }


  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<2>(3),
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
}



void Step3::solve()
{
  SolverControl            solver_control(500000, 1e-4 * system_rhs.l2_norm());
  SolverCG<Vector<double>> solver(solver_control);

  /* PreconditionRelaxation<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.2); */

  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
}



void Step3::output_results() const
{
  std::vector<std::string> solution_names(1, "u");
  solution_names.emplace_back("l");
  solution_names.emplace_back("c");

  std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation(1, DataComponentInterpretation::component_is_scalar);
  interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  interpretation.push_back(DataComponentInterpretation::component_is_scalar);


  DataOut<2> data_out;
  data_out.add_data_vector(dof_handler,
                           solution,
                           solution_names,
                           interpretation);
  // data_out.build_patches();

  /* std::ofstream output("solution.vtu");
  data_out.write_vtu(output); */

  //Now, we create a new dummy dof handler to output the target function and heat plate values
  DoFHandler<2> toy_dof_handler(triangulation);
  const FE_Q<2> toy_fe(2);
  toy_dof_handler.distribute_dofs(toy_fe);

  Vector<double> target(toy_dof_handler.n_dofs()),
                 hot_plate_0(toy_dof_handler.n_dofs()),
                 hot_plate_1(toy_dof_handler.n_dofs()),
                 hot_plate_2(toy_dof_handler.n_dofs()),
                 hot_plate_3(toy_dof_handler.n_dofs());

  const Target_Function<2> target_function;
  const Heat_Plate_0<2> heat_plate_0(heat_center_0);
  const Heat_Plate_1<2> heat_plate_1(heat_center_1, 0.2);
  const Heat_Plate_2<2> heat_plate_2(heat_center_2);
  const Heat_Plate_3<2> heat_plate_3(heat_center_3);

  VectorTools::interpolate(toy_dof_handler,
                           target_function,
                           target);
  VectorTools::interpolate(toy_dof_handler,
                           heat_plate_0,
                           hot_plate_0);
  VectorTools::interpolate(toy_dof_handler,
                           heat_plate_1,
                           hot_plate_1);
  VectorTools::interpolate(toy_dof_handler,
                           heat_plate_2,
                           hot_plate_2);                      
  VectorTools::interpolate(toy_dof_handler,
                           heat_plate_3,
                           hot_plate_3);

  hot_plate_0 *= solution[active_c_indices[0]];
  hot_plate_1 *= solution[active_c_indices[1]];
  hot_plate_2 *= solution[active_c_indices[2]];
  hot_plate_3 *= solution[active_c_indices[3]];

  data_out.add_data_vector(toy_dof_handler,
                           target,
                           "Target_Function");
  data_out.add_data_vector(toy_dof_handler,
                           hot_plate_0,
                           "Heat_Source_0");
  data_out.add_data_vector(toy_dof_handler,
                           hot_plate_1,
                           "Heat_Source_1");                       
  data_out.add_data_vector(toy_dof_handler,
                           hot_plate_2,
                           "Heat_Source_2");
  data_out.add_data_vector(toy_dof_handler,
                           hot_plate_3,
                           "Heat_Source_3");

  Vector<double> full_heat_profile(toy_dof_handler.n_dofs());
  full_heat_profile += hot_plate_0;
  full_heat_profile += hot_plate_1;
  full_heat_profile += hot_plate_2;
  full_heat_profile += hot_plate_3;

  data_out.add_data_vector(toy_dof_handler,
                           full_heat_profile,
                           "Full_Heat_Profile");

  data_out.build_patches();

  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);

  std::cout << "The c coefficients were "
            << std::endl
            << "\tc1: "
            << solution[active_c_indices[0]] << std::endl
            << "\tc2: "
            << solution[active_c_indices[1]] << std::endl
            << "\tc3: "
            << solution[active_c_indices[2]] << std::endl
            << "\tc4: "
            << solution[active_c_indices[3]] << std::endl;
}



void Step3::run()
{
  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}



int main()
{
  deallog.depth_console(2);

  Step3 laplace_problem;
  laplace_problem.run();

  return 0;
}
