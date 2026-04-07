#ifndef LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_H_
#define LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_H_

#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "lib/Dialect/RNS/IR/RNSAttributes.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h.inc"
#include "lib/Dialect/Polynomial/IR/PolynomialEnums.h.inc"

namespace mlir {
namespace heir {
namespace polynomial {

LogicalResult getCoefficientAttrResidues(
    Type coefficientType, Attribute value,
    SmallVectorImpl<mod_arith::ModArithAttr>& residues,
    llvm::function_ref<InFlightDiagnostic()> emitError);

FailureOr<Attribute> buildCoefficientAttrFromResidues(
    Type coefficientType, ArrayRef<mod_arith::ModArithAttr> residues,
    llvm::function_ref<InFlightDiagnostic()> emitError);

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_H_
