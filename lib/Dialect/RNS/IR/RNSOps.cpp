#include "lib/Dialect/RNS/IR/RNSOps.h"

#include <cstdint>
#include <optional>

#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/RNS/IR/RNSOps.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Region.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {
namespace rns {

FailureOr<SmallVector<Value>> computeMixedRadixCoeffs(
    ImplicitLocOpBuilder& b, Value input, const ArrayAttr& qInvProds) {
  auto rnsTy = dyn_cast<RNSType>(getElementTypeOrSelf(input.getType()));
  if (!rnsTy) {
    emitError(b.getLoc()) << "expected an RNS value, got " << input.getType();
    return failure();
  }
  int64_t numLimbs = rnsTy.getBasisTypes().size();
  if (qInvProds.size() != numLimbs - 1) {
    emitError(b.getLoc()) << "expected " << (numLimbs - 1)
                          << " qInvProds for an RNS basis with " << numLimbs
                          << " limbs, got " << qInvProds.size();
    return failure();
  }

  ArrayRef<Attribute> qInvAttrs = qInvProds.getValue();
  SmallVector<Value> mrcs;

  // The first mixed-radix coefficient is easy
  Value c0Reduced = ExtractSingleSliceOp::create(b, input, b.getIndexAttr(0));
  Value c0Lifted = mod_arith::ExtractOp::create(b, c0Reduced);
  mrcs.push_back(c0Lifted);

  // Subsequent coefficients depend on prior coefficients
  // c_i = \parens*{x_i - \sum_{j=0}^{i-1} \bracks*{c_j}_{q_j}\cdot
  // Q_{j-1}}\cdot Q_{i-1}^{-1} \in\Z_{q_i} Here the Q_i's represent partial
  // products of the limb moduli. Rather than use these values directly, we
  // evaluate the sum using Horner's method.
  for (int i = 1; i < numLimbs; i++) {
    Value xi = ExtractSingleSliceOp::create(b, input, b.getIndexAttr(i));
    Type limbValueTy = xi.getType();
    auto limbTy =
        cast<mod_arith::ModArithType>(getElementTypeOrSelf(limbValueTy));
    auto shapedType = dyn_cast<ShapedType>(limbValueTy);
    Value temp = mod_arith::EncapsulateOp::create(b, limbValueTy, mrcs[i - 1]);
    for (int j = i - 2; j >= 0; j--) {
      Value reducedCj =
          mod_arith::EncapsulateOp::create(b, limbValueTy, mrcs[j]);
      auto qjTy = cast<mod_arith::ModArithType>(rnsTy.getBasisTypes()[j]);
      auto qjAttr =
          IntegerAttr::get(limbTy.getModulus().getType(),
                           qjTy.getModulus().getValue().zextOrTrunc(
                               limbTy.getModulus().getValue().getBitWidth()));
      TypedAttr qjValueAttr = qjAttr;
      if (shapedType) {
        qjValueAttr = DenseElementsAttr::get(
            cast<ShapedType>(shapedType.clone(limbTy.getModulus().getType())),
            qjAttr);
      }
      Value qjConst =
          mod_arith::ConstantOp::create(b, limbValueTy, qjValueAttr);
      temp = mod_arith::MacOp::create(b, temp, qjConst, reducedCj);
    }
    Value ci = mod_arith::SubOp::create(b, xi, temp);
    Attribute qiInvAttr = qInvAttrs[i - 1];
    auto maAttr = dyn_cast<mod_arith::ModArithAttr>(qiInvAttr);
    if (!maAttr) {
      emitError(b.getLoc())
          << "expected mod_arith attribute, got " << qiInvAttr;
      return failure();
    }
    if (maAttr.getType() != limbTy) {
      emitError(b.getLoc())
          << "expected qInv attribute type to match limb type " << limbTy
          << ", got " << maAttr.getType();
      return failure();
    }
    TypedAttr qInvValueAttr = maAttr.getValue();
    if (shapedType) {
      qInvValueAttr = DenseElementsAttr::get(
          cast<ShapedType>(shapedType.clone(limbTy.getModulus().getType())),
          maAttr.getValue());
    }
    Value qInvConst =
        mod_arith::ConstantOp::create(b, limbValueTy, qInvValueAttr);
    ci = mod_arith::MulOp::create(b, ci, qInvConst);
    Value liftedCi = mod_arith::ExtractOp::create(b, ci);
    mrcs.push_back(liftedCi);
  }
  return mrcs;
}

// We use Horner's method to avoid
// cs = [self.get(0).liftToIntegers(zqRepr)]
// for i in range(1, len(self)):
//     xi = self.get(i)
//     temp = cs[i - 1].reduce(self.qs[i], False)
//     for j in range(i - 2, -1, -1):
//         temp = temp * self.qs[j] + cs[j].reduce(self.qs[i], False)
//     ci = (xi - temp) * Qinvs[i]
//     cs.append(ci.liftToIntegers(zqRepr))

LogicalResult ExtractSliceOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type>& results) {
  ExtractSliceOpAdaptor op(operands, attrs, properties, regions);
  RNSType elementType =
      dyn_cast<RNSType>(getElementTypeOrSelf(op.getInput().getType()));
  if (!elementType) return failure();
  RNSType truncatedType =
      inferExtractSliceReturnTypes(context, &op, elementType);
  Type resultType = truncatedType;
  if (auto shapedType = dyn_cast<ShapedType>(op.getInput().getType())) {
    resultType = shapedType.clone(truncatedType);
  }
  results.push_back(resultType);
  return success();
}

LogicalResult ExtractSliceOp::verify() {
  auto rnsType = dyn_cast<RNSType>(getElementTypeOrSelf(getInput().getType()));
  if (!rnsType) {
    return failure();
  }
  int64_t start = getStart().getZExtValue();
  int64_t size = getSize().getZExtValue();

  return verifyExtractSliceOp(this, rnsType, start, size);
}

// verification for ExtractSingleSlice used in both verify and inferReturnType
static LogicalResult verifyExtractSingleSliceInput(std::optional<Location> loc,
                                                   Type coeffType,
                                                   APInt index) {
  RNSType rnsCoeffType = dyn_cast<RNSType>(getElementTypeOrSelf(coeffType));
  if (!rnsCoeffType) return failure();
  int64_t sliceIndex = index.getSExtValue();

  int64_t numLimbs = rnsCoeffType.getBasisTypes().size();
  if (sliceIndex < 0 || sliceIndex >= numLimbs) {
    return emitOptionalError(
        loc, "'rns.extract_single_slice' index ", sliceIndex,
        " is out of bounds for an RNS type with ", numLimbs, " limbs");
  }

  auto limbCoeffType = dyn_cast<mod_arith::ModArithType>(
      rnsCoeffType.getBasisTypes()[sliceIndex]);
  if (!limbCoeffType) {
    return emitOptionalError(loc,
                             "'rns.extract_single_slice' requires the selected "
                             "RNS limb to have ModArith type, but found ",
                             rnsCoeffType.getBasisTypes()[sliceIndex]);
  }

  return success();
}

LogicalResult ExtractSingleSliceOp::verify() {
  return verifyExtractSingleSliceInput(getLoc(), getInput().getType(),
                                       getIndex());
}

LogicalResult ExtractSingleSliceOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type>& results) {
  ExtractSingleSliceOpAdaptor op(operands, attrs, properties, regions);
  Type ty = op.getInput().getType();
  APInt index = op.getIndex();
  if (failed(verifyExtractSingleSliceInput(loc, ty, index))) {
    return failure();
  }
  int64_t sliceIndex = index.getSExtValue();
  RNSType rnsCoeffType = cast<RNSType>(getElementTypeOrSelf(ty));
  auto truncatedType =
      cast<mod_arith::ModArithType>(rnsCoeffType.getBasisTypes()[sliceIndex]);

  Type resultType = truncatedType;
  if (auto shapedType = dyn_cast<ShapedType>(ty)) {
    resultType = shapedType.clone(truncatedType);
  }
  results.push_back(resultType);
  return success();
}

LogicalResult PackOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type>& results) {
  PackOpAdaptor op(operands, attrs, properties, regions);
  ValueRange input = op.getInput();
  // There must be at least one item in the list to form an RNS component
  if (input.empty()) {
    return emitOptionalError(loc, "'rns.pack' requires at least one input");
  }

  SmallVector<Type> basisTypes;
  basisTypes.reserve(input.size());
  for (Value operand : input) {
    auto maTy = dyn_cast<mod_arith::ModArithType>(operand.getType());
    if (!maTy) {
      return emitOptionalError(loc, "'rns.pack' got input with type ",
                               operand.getType());
    }
    basisTypes.push_back(maTy);
  }
  results.push_back(rns::RNSType::get(context, basisTypes));
  return success();
}

}  // namespace rns
}  // namespace heir
}  // namespace mlir
