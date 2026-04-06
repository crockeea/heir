!Zp = !mod_arith.int<3097973 : i26>
!Zp0 = !mod_arith.int<829 : i11>
!Zp1 = !mod_arith.int<101 : i11>
!Zp2 = !mod_arith.int<37 : i11>
!RNS = !rns.rns<!mod_arith.int<829 : i11>, !mod_arith.int<101 : i11>, !mod_arith.int<37 : i11>>

func.func public @test_lower_mod_switch_decompose() -> tensor<3xi11> {
  // 57543298 is -9565566
  %x = arith.constant 57543298 : i26
  %ex = mod_arith.encapsulate %x : i26 -> !Zp
  %mx = mod_arith.reduce %ex : !Zp
  %m1 = mod_arith.mod_switch %mx : !Zp to !RNS
  %r0 = rns.extract_single_slice %m1 {index = 0 : index} : !RNS -> !Zp0
  %r1 = rns.extract_single_slice %m1 {index = 1 : index} : !RNS -> !Zp1
  %r2 = rns.extract_single_slice %m1 {index = 2 : index} : !RNS -> !Zp2
  %i0 = mod_arith.extract %r0 : !Zp0 -> i11
  %i1 = mod_arith.extract %r1 : !Zp1 -> i11
  %i2 = mod_arith.extract %r2 : !Zp2 -> i11
  %out = tensor.from_elements %i0, %i1, %i2 : tensor<3xi11>
  return %out : tensor<3xi11>
}
