#ifndef LIB_DIALECT_MODARITH_IR_MODARITHTYPEINTERFACES_TD_
#define LIB_DIALECT_MODARITH_IR_MODARITHTYPEINTERFACES_TD_

include "mlir/IR/BuiltinTypeInterfaces.td"

def ModQTypeInterface : TypeInterface<"ModQTypeInterface"> {
  let cppNamespace = "::mlir::heir::mod_arith";
  let description = [{
    Interface for types that support modular arithmetic semantics.
  }];
  let methods = [
    InterfaceMethod<
        "Returns the scalar storage type used by this modular type.",
        "::mlir::Type", "getStorageType", (ins)>,
    InterfaceMethod<
        "Returns the number of residue components represented by this type.",
        "unsigned", "getNumResidues", (ins)>,
    InterfaceMethod<
        "Returns the residue type at the given index.",
        "::mlir::Type", "getResidueType", (ins "unsigned":$index)>,
    InterfaceMethod<
        "Returns true when this type is compatible with another modular type.",
        "bool", "isCompatibleWith", (ins "::mlir::Type":$other)>
  ];
}

#endif  // LIB_DIALECT_MODARITH_IR_MODARITHTYPEINTERFACES_TD_
