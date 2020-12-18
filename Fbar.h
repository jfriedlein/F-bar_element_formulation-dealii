#ifndef FBAR_H
#define FBAR_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include "../MA-Code/auxiliary_functions/StandardTensors.h"

#include <iostream>

using namespace dealii;


/// @todo Check for 2D and ax and then use Number template
/**
 */
namespace Fbar
{
	/**
	 */
	double get_vol_part_ratio ( const Tensor<2,3> &F, const Tensor<2,3> &F_c )
	{
		return /* vol_part_ratio=*/ std::pow( determinant(F_c) / determinant(F), 1./3.);
	}

	/**
	 */
	template <int dim>
	Tensor<4,dim> dFbar_dF ( const Tensor<2,3> &F, const Tensor<2,3> &F_c )
	{
		const double vol_part_ratio = get_vol_part_ratio(F,F_c);

		return /*dFbar_dF=*/ - 1./3. * vol_part_ratio * outer_product( F, transpose(invert(F)) )
							 + vol_part_ratio * StandardTensors::I_unsym_unsym<dim>();
	}
	
	/**
	 */
	template <int dim>
	Tensor<4,dim> dFbar_dFc ( const Tensor<2,3> &F, const Tensor<2,3> &F_c )
	{
		const double vol_part_ratio = get_vol_part_ratio(F,F_c);

		return /*dFbar_dFc=*/ 1./3. * vol_part_ratio * outer_product( F, transpose(invert(F_c)) );
	}


	/**
	 * Compute the linearisation \f$ \Delta \bC \f$ of the right Cauchy-Green tensor \f$ \bC \f$
	 * taking the special setup of the deformation gradient for Fbar into account.
	 * @param dC_dF The fourth-order tensor describing the derivative of the RCG wrt to the deformation gradient. Can be computed via \code dC_dF = StandardTensors::dC_dF<3>(F) \endcode
	 * @param F
	 * @param F_c
	 * @param grad_X_N_u_j
	 * @param grad_X_N_u_j_c
	 * @return
	 */
	template <int dim>
	SymmetricTensor<2,dim> deltaCbar_deltaRCG ( const Tensor<4,dim> &dC_dF, const Tensor<2,3> &F, const Tensor<2,3> &F_c, const Tensor<2,3> &grad_X_N_u_j, const Tensor<2,3> &grad_X_N_u_j_c )
	{
		return symmetrize(
							double_contract<2,0,3,1>( dC_dF,
													  double_contract<2,0,3,1>( dFbar_dF<dim>(F,F_c),grad_X_N_u_j )
													  +
													  double_contract<2,0,3,1>( dFbar_dFc<dim>(F,F_c),grad_X_N_u_j_c )
												     )
						 );
	}
	/**
	 * The lazy variant for \a dC_deltaRCG which does not require dC_dF which is however constant for the same QP_k. So, the lazy variant is much slower.
	 * @warning Untested
	 */
	template <int dim>
	SymmetricTensor<2,dim> deltaCbar_deltaRCG ( const Tensor<2,3> &F, const Tensor<2,3> &F_c, const Tensor<2,3> &grad_X_N_u_j, const Tensor<2,3> &grad_X_N_u_j_c )
	{
		AssertThrow(false, ExcMessage("dC_deltaRCG without dC_dF as argument is currently untested"));

		Tensor<4,dim> dC_dF = StandardTensors::dC_dF<3>( /*Fbar=*/ get_vol_part_ratio(F,F_c) * F );
		return deltaCbar_deltaRCG( dC_dF, F, F_c, grad_X_N_u_j, grad_X_N_u_j_c );
	}
}

#endif // FBAR_H
