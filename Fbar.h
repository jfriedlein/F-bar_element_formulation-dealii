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
	 * For F-bar we only need the gradient at the center QP (one QP) to determine the deformation
	 * gradient at the center "c". Thus, we don't store the gradients and hardcode the number of
	 * quadrature points to 1
	 * @ToDo: avoid using the FEextractor maybe use fe_values[u_fe] as input
	 */
	template<int dim>
	void prepare_DefoGradC( FEValues<dim> &fe_values_ref_RI, const Vector<double> &current_solution,
							Tensor<2,dim> &DeformationGradient_c )
	 {
		 // We use a single additional QP at the center for Fbar
		 std::vector< Tensor<2,dim> > solution_grads_u_RI( 1 );
		 fe_values_ref_RI[(FEValuesExtractors::Vector) 0].get_function_gradients(current_solution,solution_grads_u_RI);

		 DeformationGradient_c = (Tensor<2, dim>(StandardTensors::I<dim>()) + solution_grads_u_RI[0]);
	 }

	/**
	 */
	template <int dim>
	double get_vol_part_ratio ( const Tensor<2,dim> &F, const Tensor<2,dim> &F_c )
	{
		return /* vol_part_ratio=*/ std::pow( determinant(F_c) / determinant(F), 1./double(dim) );
	}

	/**
	 */
	template <int dim>
	double get_detF0_detF_ratio ( const Tensor<2,dim> &F, const Tensor<2,dim> &F_c )
	{
		return /* ratio=*/ determinant(F_c) / determinant(F);
	}

	/**
	 * @todo-optimize Add option to insert the \a vol_part_ratio and F_inv to save comp. time
	 */
	template <int dim>
	Tensor<4,dim> dFbar_dF ( const Tensor<2,dim> &F, const Tensor<2,dim> &F_c )
	{
		const double vol_part_ratio = get_vol_part_ratio(F,F_c);

		return /*dFbar_dF=*/ - 1./double(dim) * vol_part_ratio * outer_product( F, transpose(invert(F)) )
							 + vol_part_ratio * StandardTensors::I_unsym_unsym<dim>();
	}
	
	/**
	 * @todo-optimize Add option to insert the \a vol_part_ratio and F_inv to save comp. time
	 */
	template <int dim>
	Tensor<4,dim> dFbar_dFc ( const Tensor<2,dim> &F, const Tensor<2,dim> &F_c )
	{
		const double vol_part_ratio = get_vol_part_ratio(F,F_c);

		return /*dFbar_dFc=*/ 1./double(dim) * vol_part_ratio * outer_product( F, transpose(invert(F_c)) );
	}


//	/**
//	 * Compute the linearisation \f$ \Delta \bC \f$ of the right Cauchy-Green tensor \f$ \bC \f$
//	 * taking the special setup of the deformation gradient for Fbar into account.
//	 * @warning This variant is also slow, approximately a factor of 3 slower (overall computation time) than the "efficient version" below.
//	 * Because we do the operations for each dof i and j, not just each QP k.
//	 * @param dC_dF The fourth-order tensor describing the derivative of the RCG wrt to the deformation gradient. Can be computed via \code dC_dF = StandardTensors::dC_dF<3>(F) \endcode
//	 * @param F
//	 * @param F_c
//	 * @param grad_X_N_u_j
//	 * @param grad_X_N_u_j_c
//	 * @return
//	 */
//	template <int dim>
//	SymmetricTensor<2,dim> deltaCbar_deltaRCG ( const Tensor<4,dim> &dC_dF, const Tensor<2,3> &F, const Tensor<2,3> &F_c, const Tensor<2,dim> &grad_X_N_u_j, const Tensor<2,dim> &grad_X_N_u_j_c )
//	{
//		return symmetrize(
//							double_contract<2,0,3,1>( dC_dF,
//													  double_contract<2,0,3,1>( dFbar_dF<dim>(F,F_c),grad_X_N_u_j )
//													  +
//													  double_contract<2,0,3,1>( dFbar_dFc<dim>(F,F_c),grad_X_N_u_j_c )
//												     )
//						 );
//	}
//	/**
//	 * The lazy variant for \a dC_deltaRCG which does not require dC_dF which is however constant for the same QP_k. So, the lazy variant is much slower.
//	 * @warning By "much slower", I mean factors!
//	 * @warning Untested
//	 */
//	template <int dim>
//	SymmetricTensor<2,dim> deltaCbar_deltaRCG ( const Tensor<2,3> &F, const Tensor<2,3> &F_c, const Tensor<2,dim> &grad_X_N_u_j, const Tensor<2,dim> &grad_X_N_u_j_c )
//	{
//		AssertThrow(false, ExcMessage("dC_deltaRCG without dC_dF as argument is currently untested"));
//
//		Tensor<4,dim> dC_dF = extract_dim<dim> ( StandardTensors::dC_dF<3>( /*Fbar=*/ get_vol_part_ratio(F,F_c) * F ) );
//		return deltaCbar_deltaRCG( dC_dF, F, F_c, grad_X_N_u_j, grad_X_N_u_j_c );
//	}

	/**
	 * Compute the linearisation \f$ \Delta \bC \f$ of the right Cauchy-Green tensor \f$ \bC \f$
	 * taking the special setup of the deformation gradient for Fbar into account. (The "efficient" version)
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
}

#endif // FBAR_H
