// RUN: heir-opt %s | FileCheck %s

#ntt_poly = #polynomial.int_polynomial<-1 + x**8>
!zp17 = !mod_arith.int<17:i32>
!zp97 = !mod_arith.int<97:i32>
!rns_coeff = !rns.rns<!zp17, !zp97>
#ring = #polynomial.ring<coefficientType=!rns_coeff, polynomialModulus=#ntt_poly>
!poly_ty = !polynomial.polynomial<ring=#ring>
!poly_eval_ty = !polynomial.polynomial<ring=#ring, form=eval>

#root17 = #mod_arith.value<2:!zp17>
#root97 = #mod_arith.value<33:!zp97>
#root_value = #rns.value<[#root17, #root97]>
#root = #polynomial.primitive_root<value=#root_value, degree=8:i32>

module {
  // CHECK: func.func @test_ntt
  func.func @test_ntt(%arg0 : !poly_ty) -> !poly_eval_ty {
    // CHECK: polynomial.ntt
    %0 = polynomial.ntt %arg0 {root=#root} : !poly_ty
    return %0 : !poly_eval_ty
  }

  // CHECK: func.func @test_intt
  func.func @test_intt(%arg0 : !poly_eval_ty) -> !poly_ty {
    // CHECK: polynomial.intt
    %0 = polynomial.intt %arg0 {root=#root} : !poly_eval_ty
    return %0 : !poly_ty
  }
}
