#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.h"

#include <cstddef>

#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "llvm/include/llvm/ADT/APInt.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/ArrayRef.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {

using mod_arith::ModArithDialect;
using mod_arith::ModArithType;
using mod_arith::ModQTypeInterface;

namespace rns {

struct ModArithRNSBasisTypeInterface
    : public RNSBasisTypeInterface::ExternalModel<ModArithRNSBasisTypeInterface,
                                                  ModArithType> {
  bool isCompatibleWith(Type type, Type otherRnsBasisType) const {
    auto thisType = mlir::dyn_cast<ModArithType>(type);
    if (!thisType) {
      return false;
    }

    auto other = mlir::dyn_cast<ModArithType>(otherRnsBasisType);
    if (!other) {
      return false;
    }

    auto thisStorageType = thisType.getModulus().getType();
    auto otherStorageType = other.getModulus().getType();
    APInt thisModulus = thisType.getModulus().getValue();
    APInt otherModulus = other.getModulus().getValue();

    // require same storage type
    if (thisStorageType != otherStorageType) {
      return false;
    }

    // coprime test
    return llvm::APIntOps::GreatestCommonDivisor(thisModulus, otherModulus) ==
           1;
  }
};

struct RNSModQTypeInterface
    : public ModQTypeInterface::ExternalModel<RNSModQTypeInterface, RNSType> {
  Type getStorageType(Type type) const {
    auto rnsType = mlir::dyn_cast<RNSType>(type);
    if (!rnsType || rnsType.getBasisTypes().empty()) {
      return Type();
    }

    auto firstLimb = mlir::dyn_cast<ModArithType>(rnsType.getBasisTypes()[0]);
    if (!firstLimb) {
      return Type();
    }
    return firstLimb.getStorageType();
  }

  unsigned getNumResidues(Type type) const {
    auto rnsType = mlir::dyn_cast<RNSType>(type);
    if (!rnsType) {
      return 0;
    }
    return rnsType.getBasisTypes().size();
  }

  Type getResidueType(Type type, unsigned index) const {
    auto rnsType = mlir::dyn_cast<RNSType>(type);
    if (!rnsType || index >= rnsType.getBasisTypes().size()) {
      return Type();
    }
    return rnsType.getBasisTypes()[index];
  }

  bool isCompatibleWith(Type type, Type other) const {
    auto rnsType = mlir::dyn_cast<RNSType>(type);
    auto otherRnsType = mlir::dyn_cast<RNSType>(other);
    if (!rnsType || !otherRnsType) {
      return false;
    }
    return rnsType.getBasisTypes() == otherRnsType.getBasisTypes();
  }
};

void registerExternalRNSTypeInterfaces(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, ModArithDialect* dialect) {
    ModArithType::attachInterface<ModArithRNSBasisTypeInterface>(*ctx);
  });
  registry.addExtension(+[](MLIRContext* ctx, RNSDialect* dialect) {
    RNSType::attachInterface<RNSModQTypeInterface>(*ctx);
  });
}

}  // namespace rns
}  // namespace heir
}  // namespace mlir
