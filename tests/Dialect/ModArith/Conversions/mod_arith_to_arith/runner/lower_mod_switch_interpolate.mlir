!Zp = !mod_arith.int<3097973 : i26>
!Zp0 = !mod_arith.int<829 : i11>
!Zp1 = !mod_arith.int<101 : i11>
!Zp2 = !mod_arith.int<37 : i11>
!RNS = !rns.rns<!mod_arith.int<829 : i11>, !mod_arith.int<101 : i11>, !mod_arith.int<37 : i11>>

func.func public @test_lower_mod_switch_interpolate() -> i26 {
  %x = arith.constant dense<[798, 94, 23]> : tensor<3xi11>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %x0 = tensor.extract %x[%c0] : tensor<3xi11>
  %x1 = tensor.extract %x[%c1] : tensor<3xi11>
  %x2 = tensor.extract %x[%c2] : tensor<3xi11>
  %m0 = mod_arith.encapsulate %x0 : i11 -> !Zp0
  %m1 = mod_arith.encapsulate %x1 : i11 -> !Zp1
  %m2 = mod_arith.encapsulate %x2 : i11 -> !Zp2
  %ex = rns.pack %m0, %m1, %m2 : !Zp0, !Zp1, !Zp2
  %switched = mod_arith.mod_switch %ex : !RNS to !Zp
  %out = mod_arith.extract %switched : !Zp -> i26
  return %out : i26
}
