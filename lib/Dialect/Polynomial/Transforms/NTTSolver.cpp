#include "lib/Dialect/Polynomial/Transforms/NTTSolver.h"

#include <cassert>

#include "lib/Dialect/ModArith/IR/ModArithTypeInterfaces.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "ortools/sat/cp_model.h"         // from @com_google_ortools
#include "ortools/sat/cp_model_solver.h"  // from @com_google_ortools

namespace mlir {
namespace heir {
namespace polynomial {

using namespace operations_research;
using namespace operations_research::sat;

CPSATSolution::CPSATSolution(const NTTSolver& solver,
                             const CpSolverResponse& soln)
    : solver(solver), soln(soln) {}

bool CPSATSolution::needsForm(const Value& v, Form form) const {
  const BoolVar& bv = solver.vars.at(v).getVarForm(form);
  return SolutionBooleanValue(soln, bv);
}

bool CPSATSolution::needsConversion(const Value& v) const {
  const NTTSolver::RepVars& vs = solver.vars.at(v);
  return SolutionBooleanValue(soln, vs.conv);
}

Form CPSATSolution::getMode(const Value& v) const {
  const NTTSolver::RepVars& vs = solver.vars.at(v);
  return SolutionBooleanValue(soln, vs.mode) ? Form::EVAL : Form::COEFF;
}

bool CPSATSolution::isValid() const {
  return soln.status() == CpSolverStatus::OPTIMAL ||
         soln.status() == CpSolverStatus::FEASIBLE;
}

int getConversionCost(const Value& v) {
  Type t = v.getType();
  int multiplicity = 1;

  if (auto shaped = dyn_cast<ShapedType>(t)) {
    multiplicity = shaped.getNumElements();
    t = shaped.getElementType();
  }

  auto polyTy = dyn_cast<PolynomialType>(t);
  if (!polyTy) return 0;

  auto coeffTy = dyn_cast<mod_arith::ModQTypeInterface>(
      polyTy.getRing().getCoefficientType());
  assert(coeffTy &&
         "polynomial coefficient type must implement ModQTypeInterface");

  return coeffTy.getNumResidues() * multiplicity;
}

NTTSolver::RepVars& NTTSolver::getOrCreateVars(const Value& v) {
  auto it = vars.find(v);
  if (it != vars.end()) {
    return it->second;
  }
  int64_t convCost = getConversionCost(v);
  auto multiplierIt = conversionCostMultiplier.find(v);
  if (multiplierIt != conversionCostMultiplier.end()) {
    convCost *= multiplierIt->second;
  }
  RepVars repVars{/*c=*/model.NewBoolVar(),
                  /*e=*/model.NewBoolVar(),
                  /*conv=*/model.NewBoolVar(),
                  /*mode=*/BoolVar()};

  objective += LinearExpr::Term(repVars.conv, convCost);
  vars[v] = repVars;
  return vars[v];
}

void NTTSolver::setConversionCostMultiplier(const Value& v,
                                            int64_t multiplier) {
  assert(multiplier >= 1 && "conversion cost multiplier must be positive");
  assert(!vars.contains(v) &&
         "conversion cost multiplier must be set before value is modeled");
  conversionCostMultiplier[v] = multiplier;
}

const BoolVar& NTTSolver::RepVars::getVarForm(Form form) const {
  return form == Form::COEFF ? c : e;
}

void NTTSolver::forceDemandEitherForm(const Value& v) {
  RepVars& vs = getOrCreateVars(v);
  model.AddBoolOr({vs.c, vs.e});
}

void NTTSolver::forceDemandFixedForm(const Value& v, Form form) {
  RepVars& vs = getOrCreateVars(v);
  model.AddEquality(vs.getVarForm(form), 1);
}

void NTTSolver::implyForm(const Value& v, Form a, Form b) {
  RepVars& vs = getOrCreateVars(v);
  model.AddImplication(vs.getVarForm(a), vs.getVarForm(b));
}

void NTTSolver::prohibitBothForms(const Value& v) {
  RepVars& vs = getOrCreateVars(v);
  model.AddBoolOr(
      {vs.getVarForm(Form::COEFF).Not(), vs.getVarForm(Form::EVAL).Not()});
}

void NTTSolver::implyUse(const Value& out, const Value& in, Form form) {
  RepVars& outs = getOrCreateVars(out);
  RepVars& ins = getOrCreateVars(in);
  model.AddImplication(outs.getVarForm(form), ins.getVarForm(form));
}

void NTTSolver::implyMode(const Value& out, const Value& in) {
  RepVars& outs = getOrCreateVars(out);
  RepVars& ins = getOrCreateVars(in);
  // if mode = 0 and output in any form is needed, force inputs to coeff
  // ¬v_mode ^ v_c => u_c
  model.AddBoolOr({outs.mode, outs.c.Not(), ins.c});
  // ¬v_mode ^ v_e => u_c
  model.AddBoolOr({outs.mode, outs.e.Not(), ins.c});

  // if mode = 1 and output in any form is needed, force inputs to eval
  // v_mode ^ v_c => u_c
  model.AddBoolOr({outs.mode.Not(), outs.c.Not(), ins.e});
  // v_mode ^ v_e => u_c
  model.AddBoolOr({outs.mode.Not(), outs.e.Not(), ins.e});
}

void NTTSolver::addConversionCostForForm(const Value& v, Form form) {
  RepVars& vs = getOrCreateVars(v);
  model.AddEquality(vs.conv, form == Form::COEFF ? vs.c : vs.e);
}

BoolVar NTTSolver::addFloatingMode(llvm::ArrayRef<Value> members) {
  BoolVar mode = model.NewBoolVar();
  for (const Value& v : members) {
    getOrCreateVars(v).mode = mode;
  }
  return mode;
}

void NTTSolver::addModeConversionCost(const Value& member,
                                      const BoolVar& mode) {
  RepVars& vs = getOrCreateVars(member);
  // conv == (mode ^ c) v (!mode ^ e), i.e. a conversion is needed at
  // member's own materialization site iff member's own demand disagrees
  // with the group's shared mode.
  model.AddBoolOr({mode.Not(), vs.c.Not(), vs.conv});  // mode ^ c => conv
  model.AddBoolOr({mode, vs.e.Not(), vs.conv});        // !mode ^ e => conv
  model.AddBoolOr({vs.conv.Not(), mode.Not(), vs.c});  // conv ^ mode => c
  model.AddBoolOr({vs.conv.Not(), mode, vs.e});        // conv ^ !mode => e
}

void NTTSolver::setZeroConversionCost(const Value& v) {
  RepVars& vs = getOrCreateVars(v);
  model.AddEquality(vs.conv, 0);
  // Since there is no conversion cost, we add a cost to
  // materialized forms so that we don't end up materializing
  // dead code. Brief proof of correctness:
  // - if some op needs vs.c/vs.e, there is an implication
  //   that forces the variable to 1, meaning adding a cost
  //   here can't cause the solver to set it to 0 to minimize
  //   cost
  // - if no op needs vs.c/vs.e, then the minimal solution
  //   sets the variable to 0
  objective += vs.c;
  objective += vs.e;
}

CPSATSolution NTTSolver::solve() {
  model.Minimize(objective);
  const CpSolverResponse resp = Solve(model.Build());
  return CPSATSolution(*this, resp);
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
