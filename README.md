# F-bar_element_formulation-dealii
Functions to implement volumetric locking-free elements using the F-bar element formulation in deal.II

will be documented soon together with a "how to use" code and maybe even some benchmarks, meanwhile you can check out selective reduced integration that also alleviates volumetric locking
https://github.com/jfriedlein/selective_reduced_integration_SRI-dealii

## Background
see book "COMPUTATIONAL METHODS FOR PLASTICITY" by Neto

see papers "DESIGN OF SIMPLE LOW ORDER FINITE ELEMENTS FOR LARGE STRAIN ANALYSIS OF NEARLY INCOMPRESSIBLE SOLIDS_Neto_Dutko_Owen" and "F-bar-based linear triangles and tetrahedra for finite strain analysis of nearly incompressible solids. Part I - formulation and benchmarking_Neto_Owen"

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

Moreover, we compute the some derivative for the later linearisation of the residual. We compute these here, because these values are constant for each dof, so save substantial computation time if we do this only once for each QP.
```
      Tensor<4,dim> dFbar_dF  = Fbar::dFbar_dF <dim>( DefoGrad_compatible, DeformationGradient_c );
      Tensor<4,dim> dFbar_dFc = Fbar::dFbar_dFc<dim>( DefoGrad_compatible, DeformationGradient_c );
```
Now, you can call your material model with the modified deformation gradient `DeformationGradient`.
```
      {DeformationGradient, history_n} -> material model -> {stress_S_3D, Tangent_3D, history_tmp}
```
Because our routines utilise the second Piola-Kirchhoff stress (PK2) `stress_S_3D`, we also need to scale this stress tensor to account for the changing deformation gradient (details see book by Neto and his papers). This would not be necessary if we used the Cauchy stress.
```
      stress_scaler = Fbar::get_stress_scaler_PK2(DefoGrad_compatible,DeformationGradient_c);
      stress_S_3D *= stress_scaler;      
```
Next, we prepare some more contributions to the tangent, again as early as possible to avoid unnessary recomputations.
```
      Tensor<4,dim> S_dyadic_F_invT = Fbar::get_S_dyadic_F_invT( stress_S, DefoGrad_compatible );
      Tensor<4,dim> S_dyadic_F0_invT =Fbar::get_S_dyadic_F_invT( stress_S, DeformationGradient_c );
```
For the computation of the linearisation we also need the gradient of the shape function at the centre QP for the j-th dof.
```
      ...
      for(unsigned int i=0; i<dofs_per_cell; ++i) {
      ... computing the residual ...
         for(unsigned int j=0; j<dofs_per_cell; ++j) {
	    ...
            Tensor<2,dim> grad_X_N_u_j = fe_values_ref[u_fe].gradient(j,k);
	    Tensor<2,dim> grad_X_N_u_j_c = fe_values_ref_RI[u_fe].gradient(j,0);
```
This helps us to compute the linearisation of the deformation gradient, of the right Cauchy-Green tensor (RCG), and the second Piola Kirchhoff stress (S). Here, all the previously computed values come together.
```
            Tensor<2,dim> deltaF = Fbar::deltaFbar( dFbar_dF, dFbar_dFc, grad_X_N_u_j, grad_X_N_u_j_c );
            SymmetricTensor<2,dim> deltaRCG = 2 * symmetrize( transpose(DeformationGradient) * deltaF );
            SymmetricTensor<2,dim> deltaS = Fbar::deltaS( Tangent_3D, deltaRCG, stress_scaler, S_dyadic_F_invT, grad_X_N_u_j, S_dyadic_F0_invT, grad_X_N_u_j_c );
```
`deltaS` can then be used to compute the contribution to the tangent matrix `cell_matrix`.
```
            cell_matrix(i,j) += (
                                   symmetrize( transpose(grad_X_N_u_j) * grad_X_N_u_i ) * stress_S_3D
                                   +
                                   symmetrize( transpose(DefoGrad_compatible) * grad_X_N_u_i ) * deltaS
                                )
                                * JxW;
```
Please pay attention to where to use which deformation gradient. This you have to derive yourself for your case.


### ToDo
update the code and the docu, also including the linearisation, what defoGrad to use when and 2D,3D,axisym
