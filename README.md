# F-bar_element_formulation-dealii
Functions to implement volumetric locking-free elements using the F-bar element formulation in deal.II

will be documented soon together with a "how to use" code and maybe even some benchmarks, meanwhile you can check out selective reduced integration that also alleviates volumetric locking
https://github.com/jfriedlein/selective_reduced_integration_SRI-dealii

## Background
see book "COMPUTATIONAL METHODS FOR PLASTICITY" by Neto

## Argument list
fill this

* FEValues<dim> fe_values_ref; // The FEValues element that corresponds to the displacement dofs
* Tensor<2,dim> DeformationGradient_c; // deformation gradient at the centre of an element, unsymmetric second order tensor
* ...
	

## Implementation/Usage: Modifying and extending your existing code


1. Declarations
In your main class (in deal.II it is named for instance "step3" [deal.II step3 tutorial](https://www.dealii.org/current/doxygen/deal.II/step_3.html) where you declare your typical QGauss quadrature rule `qf_cell` for the integration over the cell, add another QGauss rule named `qf_cell_RI`. The latter will describe the reduced integration (RI) that we use to get the quadrature at the element centre. Additionally, declare an FEValues object named `fe_values_ref_RI*  that uses the `qf_cell_RI` rule.
```
	const QGauss<dim> qf_cell;
	FEValues<dim> fe_values_ref;
	
	const QGauss<dim> qf_cell_RI;
	FEValues<dim> fe_values_ref_RI;
```

2. Constructor
In the constructor for the above main class, we now also have to  initialise the new variables, which we do as follows
```
	...
  qf_cell( degree +1 ),
	qf_cell_RI( degree +1 -1 ),
	fe_values_ref_RI (	fe,//The used FiniteElement
					            qf_cell_RI,//The quadrature rule for the cell
					            update_values | //UpdateFlag for shape function values
					            update_gradients | //shape function gradients
					            update_JxW_values ), //transformed quadrature weights multiplied with Jacobian of transformation
	...
```
The standard `qf_cell` is initialised as usual, where `degree` denotes the polynomial order that is used for the element (1: linear element, 2: quadratic element, ...). The reduced integration uses one order less, so we init the `qf_cell_RI` with the order of `qf_cell` minus 1.

3. Assembly routine
Inside the loop over cell, we now reinit `fe_values_ref` and `fe_values_ref_RI`
```
  ...
  for(;cell!=endc;++cell) {
    ...
	  fe_values_ref.reinit(cell);
	  fe_values_ref_RI.reinit(cell);
    ...
```
Next, we determine the displacement gradients grouped for all quadrature points as "usual" ("usual" means here that is is unaffected by the F-bar formulation)
```
   fe_values_ref[u_fe].get_function_gradients(current_solution,solution_grads_u);
```
We determine the deformation gradient in the centre of the current cell `DeformationGradient_c` by calling
```
    Fbar::prepare_DefoGradC( fe_values_ref_RI, current_solution, DeformationGradient_c );
```
Note that this is only done once per cell, because every cell has only one centre. We use this centre deformation gradient now identically for each quadrature point.
Within the loop over the quadrature points `for (k)`, we first determine the compatible deformation gradient for the k-th quadrature point as "usual".
```
    ...
		for( unsigned int k=0; k < n_quadrature_points; ++k ) {
      const Tensor<2,dim> DefoGrad_compatible = (Tensor<2, dim>(unit_symmetric_tensor<dim>()) + solution_grads_u[k]);
      ...
```  
For the F-bar element formulation, we modify the volumetric part of the compatible deformation gradient. This is done by the function
```
      DeformationGradient = Fbar::get_vol_part_ratio( DefoGrad_compatible, DeformationGradient_c ) * DefoGrad_compatible;
```
which automatically replaced the volumetric part of the input `DefoGrad_compatible` by the volumetric part of the centre quadrature point contained in `DeformationGradient_c`.

Additionally, because our routines utilise the second Piola-Kirchhoff stress (PK2), we also need to scale this stress tensor to account for the changing deformation gradient (details see book by Neto and his papers). This would not be necessary if we used the Cauchy stress.
```
      stress_scaler = Fbar::get_stress_scaler_PK2(DefoGrad_compatible,DeformationGradient_c);
```
Moreover, we compute the some derivative for the later linearisation of the residual. We compute these here, because these values are constant for each dof, so save substantial computation time if we do this only once for each QP.
```
      Tensor<4,dim> dFbar_dF  = Fbar::dFbar_dF <dim>( DefoGrad_compatible, DeformationGradient_c );
      Tensor<4,dim> dFbar_dFc = Fbar::dFbar_dFc<dim>( DefoGrad_compatible, DeformationGradient_c );
```

### ToDo
update the code and the docu, also including the linearisation, what defoGrad to use when and 2D,3D,axisym
