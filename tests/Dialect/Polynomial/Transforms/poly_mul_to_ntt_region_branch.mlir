// RUN: heir-opt --convert-polynomial-mul-to-ntt %s | FileCheck %s

!Zq0 = !mod_arith.int<1095233372161 : i64>
#ring_1 = #polynomial.ring<coefficientType = !rns.rns<!Zq0>, polynomialModulus = <1 + x**1024>>
!poly_ty_1 = !polynomial.polynomial<ring=#ring_1, form=coeff>
!ntt_poly_ty_1 = !polynomial.polynomial<ring=#ring_1, form=eval>

module {
  // Covers: an scf.for iter_arg that needs both forms -- coeff to match its
  // loop result (consumed by to_tensor, coeff-only) and eval to feed the
  // eval-only MulOp in the loop body. Since the loop's iter_arg and result
  // share one materialized type, but each has its own demand driven by a
  // different consumer, the solver settles the whole loop (argument,
  // iter_arg, and result) on eval -- so the loop body needs zero
  // conversions, no matter how many times it runs -- and pays the one
  // unavoidable coeff conversion exactly once, right after the loop.
  // CHECK: func.func @for_iter_arg_needs_both_forms([[x:%.+]]: [[ntt_poly_ty_1:![^ ]+]],
  // CHECK-SAME: -> tensor<1024x[[RNS:![^ ]+]]> {
  // CHECK: [[r:%.+]] = scf.for {{.*}} iter_args([[acc:%.+]] = [[x]]) -> ([[ntt_poly_ty_1]]) {
  // CHECK: [[sq:%.+]] = polynomial.mul [[acc]], [[acc]] : [[ntt_poly_ty_1]]
  // CHECK: scf.yield [[sq]] : [[ntt_poly_ty_1]]
  // CHECK: }
  // CHECK: [[rc:%.+]] = polynomial.intt [[r]] : [[ntt_poly_ty_1]]
  // CHECK: [[t:%.+]] = polynomial.to_tensor [[rc]] : [[poly_ty_1:![^ ]+]] -> tensor<1024x[[RNS]]>
  // CHECK: return [[t]] : tensor<1024x[[RNS]]>
  func.func @for_iter_arg_needs_both_forms(%x: !poly_ty_1, %lb: index, %ub: index, %step: index) -> tensor<1024x!rns.rns<!Zq0>> {
    %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %x) -> !poly_ty_1 {
      %sq = polynomial.mul %acc, %acc : !poly_ty_1
      scf.yield %sq : !poly_ty_1
    }
    %t = polynomial.to_tensor %r : !poly_ty_1 -> tensor<1024x!rns.rns<!Zq0>>
    return %t : tensor<1024x!rns.rns<!Zq0>>
  }

  // Covers: the same both-forms-needed iter_arg as above, but with a
  // statically-known trip count (4), so the solver isn't just falling back
  // to the "unknown trip count is assumed expensive" default -- it's
  // actually weighing "4 conversions inside the loop" against "1 conversion
  // outside it" and picking the latter because 1 < 4, confirming the
  // hoisting is genuinely cost-driven rather than an artifact of one
  // specific loop shape.
  // CHECK: func.func @for_iter_arg_static_trip_count([[x4:%.+]]: [[ntt_poly_ty_1]])
  // CHECK-SAME: -> tensor<1024x[[RNS]]> {
  // CHECK: [[r4:%.+]] = scf.for {{.*}} iter_args([[acc4:%.+]] = [[x4]]) -> ([[ntt_poly_ty_1]]) {
  // CHECK: [[sq4:%.+]] = polynomial.mul [[acc4]], [[acc4]] : [[ntt_poly_ty_1]]
  // CHECK: scf.yield [[sq4]] : [[ntt_poly_ty_1]]
  // CHECK: }
  // CHECK: [[r4c:%.+]] = polynomial.intt [[r4]] : [[ntt_poly_ty_1]]
  // CHECK: [[t4:%.+]] = polynomial.to_tensor [[r4c]] : [[poly_ty_1]] -> tensor<1024x[[RNS]]>
  // CHECK: return [[t4]] : tensor<1024x[[RNS]]>
  func.func @for_iter_arg_static_trip_count(%x: !poly_ty_1) -> tensor<1024x!rns.rns<!Zq0>> {
    %lb = arith.constant 0 : index
    %ub = arith.constant 4 : index
    %step = arith.constant 1 : index
    %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %x) -> !poly_ty_1 {
      %sq = polynomial.mul %acc, %acc : !poly_ty_1
      scf.yield %sq : !poly_ty_1
    }
    %t = polynomial.to_tensor %r : !poly_ty_1 -> tensor<1024x!rns.rns<!Zq0>>
    return %t : tensor<1024x!rns.rns<!Zq0>>
  }

  // Covers: two independent polynomial iter_args in the same scf.for, each
  // getting its own, independently solved form. %acc0 needs coeff (feeds
  // to_tensor after the loop) and eval (feeds the eval-only mul in the loop
  // body); like the previous test, the solver settles the whole %acc0/%r0
  // slot on eval and pays one intt after the loop rather than converting on
  // every iteration. %acc1 merely passes through unused, so nothing
  // constrains its form at all -- any choice costs zero conversions, and
  // which one the solver lands on here is a tie-break, not a guarantee.
  // CHECK: func.func @for_multiple_independent_iter_args([[a:%.+]]: [[ntt_poly_ty_1]], [[b:%.+]]: [[poly_ty_1]],
  // CHECK: [[loop:%.+]]:2 = scf.for {{.*}} iter_args([[acc0:%.+]] = [[a]], [[acc1:%.+]] = [[b]]) -> ([[ntt_poly_ty_1]], [[poly_ty_1]]) {
  // CHECK: [[sq0:%.+]] = polynomial.mul [[acc0]], [[acc0]] : [[ntt_poly_ty_1]]
  // CHECK: scf.yield [[sq0]], [[acc1]] : [[ntt_poly_ty_1]], [[poly_ty_1]]
  // CHECK: }
  // CHECK: [[r0c:%.+]] = polynomial.intt [[loop]]#0 : [[ntt_poly_ty_1]]
  // CHECK: [[t:%.+]] = polynomial.to_tensor [[r0c]] : [[poly_ty_1]] -> tensor<1024x[[RNS]]>
  // CHECK: return [[t]], [[loop]]#1 : tensor<1024x[[RNS]]>, [[poly_ty_1]]
  func.func @for_multiple_independent_iter_args(%a: !poly_ty_1, %b: !poly_ty_1, %lb: index, %ub: index, %step: index) -> (tensor<1024x!rns.rns<!Zq0>>, !poly_ty_1) {
    %r0, %r1 = scf.for %i = %lb to %ub step %step iter_args(%acc0 = %a, %acc1 = %b) -> (!poly_ty_1, !poly_ty_1) {
      %sq0 = polynomial.mul %acc0, %acc0 : !poly_ty_1
      scf.yield %sq0, %acc1 : !poly_ty_1, !poly_ty_1
    }
    %t = polynomial.to_tensor %r0 : !poly_ty_1 -> tensor<1024x!rns.rns<!Zq0>>
    return %t, %r1 : tensor<1024x!rns.rns<!Zq0>>, !poly_ty_1
  }

  // Covers: a tensor<poly> loop-carried value. Since the result is returned
  // directly with no coeff-only consumer, the whole loop (argument, iter_arg,
  // and result) settles on eval form for free, with no conversions inserted.
  // CHECK: func.func @for_tensor_iter_arg([[xt:%.+]]: tensor<2x[[ntt_poly_ty_1]]>,
  // CHECK: [[rt:%.+]] = scf.for {{.*}} iter_args([[acct:%.+]] = [[xt]]) -> (tensor<2x[[ntt_poly_ty_1]]>) {
  // CHECK-NOT: polynomial.ntt
  // CHECK-NOT: polynomial.intt
  // CHECK: [[sqt:%.+]] = polynomial.mul [[acct]], [[acct]] : tensor<2x[[ntt_poly_ty_1]]>
  // CHECK: scf.yield [[sqt]] : tensor<2x[[ntt_poly_ty_1]]>
  // CHECK: return [[rt]] : tensor<2x[[ntt_poly_ty_1]]>
  func.func @for_tensor_iter_arg(%x: tensor<2x!poly_ty_1>, %lb: index, %ub: index, %step: index) -> tensor<2x!poly_ty_1> {
    %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %x) -> tensor<2x!poly_ty_1> {
      %sq = polynomial.mul %acc, %acc : tensor<2x!poly_ty_1>
      scf.yield %sq : tensor<2x!poly_ty_1>
    }
    return %r : tensor<2x!poly_ty_1>
  }

  // Covers: scf.if yielding the same polynomial from both branches, with a
  // coeff-only consumer of the original value and an eval-only consumer of
  // the if's result. Since scf.if's regions take no block arguments, this
  // exercises only the "value returned to the parent" successor-input case
  // (not entry/backedge edges), forwarded from two different scf.yield ops
  // (one per branch) to the same result. x needs coeff (to_tensor) and the
  // if's own result needs eval (the mul after it); scf.if executes each
  // branch exactly once, so it costs exactly the same (one conversion) to
  // land that conversion on x before the if or on the if's result after it
  // -- which one the solver picks is a tie-break, not a guarantee.
  // CHECK: func.func @if_yields_both_branches([[cond:%.+]]: i1, [[x2:%.+]]: [[poly_ty_1:![^ ]+]])
  // CHECK: [[r2:%.+]] = scf.if [[cond]] -> ([[poly_ty_1]]) {
  // CHECK: scf.yield [[x2]] : [[poly_ty_1]]
  // CHECK: } else {
  // CHECK: scf.yield [[x2]] : [[poly_ty_1]]
  // CHECK: }
  // CHECK: [[r2e:%.+]] = polynomial.ntt [[r2]] : [[poly_ty_1]]
  // CHECK: [[t2:%.+]] = polynomial.to_tensor [[x2]] : [[poly_ty_1]] -> tensor<1024x[[RNS]]>
  // CHECK: [[m2:%.+]] = polynomial.mul [[r2e]], [[r2e]] : [[ntt_poly_ty_1:![^ ]+]]
  // CHECK: return [[t2]], [[m2]] : tensor<1024x[[RNS]]>, [[ntt_poly_ty_1]]
  func.func @if_yields_both_branches(%cond: i1, %x: !poly_ty_1) -> (tensor<1024x!rns.rns<!Zq0>>, !poly_ty_1) {
    %r = scf.if %cond -> !poly_ty_1 {
      scf.yield %x : !poly_ty_1
    } else {
      scf.yield %x : !poly_ty_1
    }
    %t = polynomial.to_tensor %x : !poly_ty_1 -> tensor<1024x!rns.rns<!Zq0>>
    %m = polynomial.mul %r, %r : !poly_ty_1
    return %t, %m : tensor<1024x!rns.rns<!Zq0>>, !poly_ty_1
  }

  // Covers: scf.while, whose scf.condition op forwards the very same
  // operand to two different successor inputs at once -- the "after"
  // region's block argument (if the loop continues) and the scf.while op's
  // own result (if it exits) -- exercising the case where a single physical
  // operand fans out to multiple successor inputs that must share one
  // resolved form (see NTTSolver::addFloatingMode). Both the "before"
  // region's argument and this after-arg/result slot settle on eval: the
  // before-arg needs eval anyway (the mul), and forcing the after-arg/result
  // slot to coeff instead would mean converting on every iteration (inside
  // the "after" region) rather than paying the one coeff conversion
  // to_tensor needs exactly once, after the whole loop.
  // CHECK: func.func @while_loop([[x3:%.+]]: [[ntt_poly_ty_1:![^ ]+]], [[cond3:%.+]]: i1) -> tensor<1024x[[RNS:![^ ]+]]> {
  // CHECK: [[r3:%.+]] = scf.while ([[acc3:%.+]] = [[x3]]) : ([[ntt_poly_ty_1]]) -> [[ntt_poly_ty_1]] {
  // CHECK: [[sq3:%.+]] = polynomial.mul [[acc3]], [[acc3]] : [[ntt_poly_ty_1]]
  // CHECK: scf.condition({{.*}}) [[sq3]] : [[ntt_poly_ty_1]]
  // CHECK: } do {
  // CHECK: ^bb0([[after3:%.+]]: [[ntt_poly_ty_1]]):
  // CHECK: scf.yield [[after3]] : [[ntt_poly_ty_1]]
  // CHECK: }
  // CHECK: [[r3c:%.+]] = polynomial.intt [[r3]] : [[ntt_poly_ty_1]]
  // CHECK: [[t3:%.+]] = polynomial.to_tensor [[r3c]] : [[poly_ty_1:![^ ]+]] -> tensor<1024x[[RNS]]>
  // CHECK: return [[t3]] : tensor<1024x[[RNS]]>
  func.func @while_loop(%x: !poly_ty_1, %cond0: i1) -> tensor<1024x!rns.rns<!Zq0>> {
    %r = scf.while (%acc = %x) : (!poly_ty_1) -> !poly_ty_1 {
      %sq = polynomial.mul %acc, %acc : !poly_ty_1
      %c = arith.constant true
      scf.condition(%c) %sq : !poly_ty_1
    } do {
    ^bb0(%arg: !poly_ty_1):
      scf.yield %arg : !poly_ty_1
    }
    %t = polynomial.to_tensor %r : !poly_ty_1 -> tensor<1024x!rns.rns<!Zq0>>
    return %t : tensor<1024x!rns.rns<!Zq0>>
  }
}
