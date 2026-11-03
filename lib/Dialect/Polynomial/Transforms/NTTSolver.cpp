#include "lib/Dialect/Polynomial/Transforms/NTTSolver.h"

#include <cassert>
#include <cstdint>

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
  // Only a proven optimum is accepted; a merely FEASIBLE solution may use
  // more conversions than necessary.
  return soln.status() == CpSolverStatus::OPTIMAL;
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
  assert(multiplier >= 0 && "conversion cost multiplier must be non-negative");
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
  modeVars.push_back(mode);
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
  // `objective` is the total number of per-limb NTTs+iNTTs for a fixed
  // assignment to the set of variables. Multiple assignments can result in the
  // same objective, making deterministic tests fragile. We *partially* address
  // this problem by adding a secondary objective `tieBreak` which is the total
  // number of modes set to EVAL. This ensures that, among all possible
  // solutions that minimize the primary objective, we pick one which
  // (arbitrarily) minimizes the number of EVAL modes. `primaryScale` is a
  // constant (i.e., independent of the mode assignments) which is always larger
  // than `tieBreak`, therefore `objective * primaryScale + tieBreak` results in
  // a two-tiered optimization. The tie-break term is at most modeVars.size(),
  // so scaling the primary objective by modeVars.size() + 1 means saving one
  // unit of conversion cost always outweighs any change in the tie-break term.
  //
  // The current tests all happen to have a unique solution under this
  // condition, but in general, multiple solutions may still exist. For example:
  //  - f(a, b) has the same objective cost when exactly one of a/b is EVAL.
  //      Both solutions also clearly have the same *secondary* objective cost,
  //      so a non-deterministic solver (either from randomness or from
  //      multi-threading) or different solver version is free to choose a
  //      solution different from the one chosen by the current solver.
  //  - Regions that run zero times (e.g., a loop with a static trip count of 0)
  //      can include conversions (or not) that don't change the objective,
  //      since the region never runs, but which changes the IR.
  //  - Returned constants, which have no conversion cost, can be output in
  //  either form
  //  - Zero-length tensors have arbitrary form
  //
  // Currently, non-determinism isn't hurting anything, so we finding a robust,
  // cheap way to produce a globally-unique solution to the CPSAT instance as an
  // open problem. But if you update the solver version, run this on a very big
  // machine with lots of cores, or make a seemingly innocuous change in
  // PolyMulToNTT that shouldn't affect the optimal solution, be aware that the
  // concrete IR output by this pass can change!
  LinearExpr tieBreak;
  for (const BoolVar& mode : modeVars) {
    tieBreak += mode;
  }
  const int64_t primaryScale = static_cast<int64_t>(modeVars.size()) + 1;
  model.Minimize(objective * primaryScale + tieBreak);
  const CpSolverResponse resp = Solve(model.Build());
  return CPSATSolution(*this, resp);
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
