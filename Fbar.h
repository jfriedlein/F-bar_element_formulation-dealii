#ifndef FBAR_H
#define FBAR_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include "../MA-Code/auxiliary_functions/StandardTensors.h"

#include <iostream>

using namespace dealii;


/**
 * Based on "COMPUTATIONAL METHODS FOR PLASTICITY" by Neto and "Regularisation of gradient-enhanced damage coupled to finite plasticity" by Sprave&Menzel
 * @todo There still seems to be a problem with convergence
 * @todo Check for 2D and ax and then use Number template
 */
namespace Fbar
{
	/**
	 * For F-bar we only need the gradient at the center QP (one QP) to determine the volumetric part of deformation
	 * gradient at the center "c". Thus, we don't store the gradients and hardcode the number of
	 * quadrature points to 1
	 * @ToDo: avoid using the FEextractor maybe use fe_values[u_fe] as input
	 */
	template<int dim>
	void prepare_DefoGradC( FEValues<dim> &fe_values_ref_RI, const Vector<double> &current_solution,
							Tensor<2,dim> &DeformationGradient_c )
	 {
		 // We use a single QP at the center for Fbar
		 std::vector< Tensor<2,dim> > solution_grads_u_RI( 1 );
		 fe_values_ref_RI[(FEValuesExtractors::Vector) 0].get_function_gradients(current_solution,solution_grads_u_RI);

		 DeformationGradient_c = (Tensor<2, dim>(unit_symmetric_tensor<dim>()) + solution_grads_u_RI[0]);
	 }

	/**
	 * @brief Get ratio of the volumetric parts of the two given deformation gradients
	 * 
	 * @tparam dim 
	 * @param F 
	 * @param F_c 
	 * @return double 
	 */
	template <int dim>
	double get_vol_part_ratio ( const Tensor<2,dim> &F, const Tensor<2,dim> &F_c )
	{
		return /* vol_part_ratio=*/ std::pow( determinant(F_c) / determinant(F), 1./double(dim) );
	}

	/**
	 * @brief Get the ratio of the determinants of the two given deformation gradients
	 * 
	 * @tparam dim 
	 * @param F 
	 * @param F_c 
	 * @return double 
	 */
	template <int dim>
	double get_detF0_detF_ratio ( const Tensor<2,dim> &F, const Tensor<2,dim> &F_c )
	{
		return /* ratio=*/ determinant(F_c) / determinant(F);
	}
	/**
	 * @brief Return the fourth order tangent that results from the derivative of an unsymmetric tensor with itself
	 * @todo-optimize Yes, this can be done easier and is already available in deal.ii (but nice to see it explicitly)
	 * 
	 * @tparam dim 
	 * @return Tensor<4, dim> 
	 */
	template <int dim>
	Tensor<4, dim> I_unsym_unsym()
	{
		Tensor<4, dim> dF_dF;
		{
			for (unsigned int i = 0; i < dim; i++)
				for (unsigned int j = 0; j < dim; j++)
					for (unsigned int k = 0; k < dim; k++)
						for (unsigned int l = 0; l < dim; l++)
							if (i == k && j == l)
								dF_dF[i][j][k][l] = 1.;
		}
		return dF_dF;
	}
	/**
	 * @brief Derivative of Fbar with respect to F
	 * @todo-optimize Add option to insert the \a vol_part_ratio and F_inv as input arguments to save computation time
	 * @param F
	 * @param F_c
	 * @return * template <int dim>
	 */
		template <int dim>
		Tensor<4, dim> dFbar_dF(const Tensor<2, dim> &F, const Tensor<2, dim> &F_c)
	{
		const double vol_part_ratio = get_vol_part_ratio(F,F_c);

		return /*dFbar_dF=*/ - 1./double(dim) * vol_part_ratio * outer_product( F, transpose(invert(F)) )
							 + vol_part_ratio * I_unsym_unsym<dim>();
	}

	/**
	 * @brief Derivative of Fbar with respect to Fc
	 * @todo-optimize Add option to insert the \a vol_part_ratio and F_inv as input arguments to save computation time
	 * @tparam dim
	 * @param F
	 * @param F_c
	 * @return Tensor<4,dim>
	 */
	template <int dim>
	Tensor<4,dim> dFbar_dFc ( const Tensor<2,dim> &F, const Tensor<2,dim> &F_c )
	{
		const double vol_part_ratio = get_vol_part_ratio(F,F_c);

		return /*dFbar_dFc=*/ 1./double(dim) * vol_part_ratio * outer_product( F, transpose(invert(F_c)) );
	}

	/**
	 * @brief Compute the scaling factor that is necessary for the PK2 stress
	 * 
	 * @todo Choose dim=3 for 3D and axisym, whereas dim=2 for 2D-plane strain
	 * @param DefoGrad_compatible
	 * @param DeformationGradient_c
	 * @return
	 */
	template <int dim>
	double get_stress_scaler_PK2(const Tensor<2,dim> &DefoGrad_compatible, const Tensor<2,dim> &DeformationGradient_c )
	{
		 return std::pow( get_detF0_detF_ratio( DefoGrad_compatible, DeformationGradient_c ), 2./double(dim)-1. );
	}

	/**
	 * Compute the linearisation \f$ \Delta \bC \f$ of the right Cauchy-Green tensor \f$ \bC \f$
	 * taking the special setup of the deformation gradient for Fbar into account. (The "efficient" version)
	 * @todo-optimize Can we simply use deltaC=2(F^T*deltaF)^sym?
	 * @todo-optimize can we do some faster contraction, if we take the symmetric part in the end anyway?
	 * @todo Update docu
	 * @param grad_X_N_u_j
	 * @param grad_X_N_u_j_c
	 * @return
	 */
	template <int dim>
	SymmetricTensor<2,dim> deltaCbar_deltaRCG ( const Tensor<4,dim> &dCbar_dF, const Tensor<4,dim> &dCbar_dFc,
												const Tensor<2,dim> &grad_X_N_u_j, const Tensor<2,dim> &grad_X_N_u_j_c )
	{
		return symmetrize(
							double_contract<2,0,3,1>( dCbar_dF, grad_X_N_u_j )
							+
							double_contract<2,0,3,1>( dCbar_dFc,grad_X_N_u_j_c )
						 );
	}

	/**
	 * @BugFix At some point deltaFbar was symmetrized, which is obviously wrong. It it a non-symmetric second order tensor.
	 * @param dFbar_dF
	 * @param dFbar_dFc
	 * @param grad_X_N_u_j
	 * @param grad_X_N_u_j_c
	 * @return
	 */
	template <int dim>
	Tensor<2,dim> deltaFbar ( const Tensor<4,dim> &dFbar_dF, const Tensor<4,dim> &dFbar_dFc,
									   const Tensor<2,dim> &grad_X_N_u_j, const Tensor<2,dim> &grad_X_N_u_j_c )
	{
		return   (
					double_contract<2,0,3,1>( dFbar_dF, grad_X_N_u_j )
					+
					double_contract<2,0,3,1>( dFbar_dFc,grad_X_N_u_j_c )
				 );
	}

	template <int dim>
	SymmetricTensor<2,dim> deltaS ( const Tensor<4,dim> &dS_hat_dFbar, const Tensor<2,dim> &deltaF, const double &stress_scaler,
									const Tensor<4,dim> &S_dyadic_F_invT, const Tensor<2,dim> &grad_X_N_u_j,
									const Tensor<4,dim> &S_dyadic_F0_invT, const Tensor<2,dim> &grad_X_N_u_j_c )
	{
		  return /*deltaS=*/ symmetrize( stress_scaler * double_contract<2,0,3,1> (dS_hat_dFbar , deltaF)
										 + (2./double(dim) - 1.) * (
																	double_contract<2,0,3,1> ( S_dyadic_F0_invT, grad_X_N_u_j_c)
																	- double_contract<2,0,3,1> ( S_dyadic_F_invT, grad_X_N_u_j)
																   )
									  );
	}

	template <int dim>
	SymmetricTensor<2,dim> deltaS ( const SymmetricTensor<4,dim> &dS_hat_dCbar, const SymmetricTensor<2,dim> &deltaC, const double &stress_scaler,
									const Tensor<4,dim> &S_dyadic_F_invT, const Tensor<2,dim> &grad_X_N_u_j,
									const Tensor<4,dim> &S_dyadic_F0_invT, const Tensor<2,dim> &grad_X_N_u_j_c )
	{
		  return /*deltaS=*/ symmetrize( stress_scaler * (dS_hat_dCbar * deltaC)
										 + (2./double(dim) - 1.) * (
																	double_contract<2,0,3,1> ( S_dyadic_F0_invT, grad_X_N_u_j_c)
																	- double_contract<2,0,3,1> ( S_dyadic_F_invT, grad_X_N_u_j)
																   )
									  );
	}

	/**
	 * Two options: Either call with full 3D stress tensor (for 3D or 2D) or with 2D stress for 2D only
	 * @param PK2_stress
	 * @param defoGrad
	 * @return
	 */
	template <int dim>
	Tensor<4,dim> get_S_dyadic_F_invT  ( const SymmetricTensor<2,3> &PK2_stress, const Tensor<2,dim> &defoGrad )
	{
		return /*S_dyadic_F0_invT=*/ outer_product( Tensor<2,dim> (extract_dim<dim>(PK2_stress)), transpose(invert(defoGrad)) );
	}
	Tensor<4,2> get_S_dyadic_F_invT  ( const SymmetricTensor<2,2> &PK2_stress, const Tensor<2,2> &defoGrad )
	{
		return /*S_dyadic_F0_invT=*/ outer_product( Tensor<2,2> (PK2_stress), transpose(invert(defoGrad)) );
	}
}

#endif // FBAR_H
