#include "./RegisterTritonDialects.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"

#include <iostream>
#include <memory>
#include <vector>

using namespace mlir;
using namespace llvm;
using namespace mlir::triton;

class InstCostEstimator {
public:
  virtual ~InstCostEstimator() {}
  virtual int64_t cost(mlir::Operation *op);
};

class HopperCostEstimator : public InstCostEstimator {
public:
  HopperCostEstimator() {}
  ~HopperCostEstimator() {}
  int64_t cost(mlir::Operation *op) override { return 0; }
};

class BlackwellCostEstimator : public InstCostEstimator {
public:
  BlackwellCostEstimator() {}
  ~BlackwellCostEstimator() {}
  int64_t cost(mlir::Operation *op) override {
    assert(false);
    return 0;
  }
};

std::string node_name(const mlir::Value &value, mlir::AsmState &asm_state) {
  std::string str;
  llvm::raw_string_ostream os(str);
  value.printAsOperand(os, asm_state);
  if (auto defOp = value.getDefiningOp()) {
    auto producer = defOp->getName();
    os << "_" << producer << " ";
  } else {
    os << "_blockarg ";
  }
  std::string result = "\"" + os.str() + "\"";
  return result;
}

void dump_dot_graph(
    llvm::DenseMap<mlir::Value, llvm::DenseSet<mlir::Value>> &dependence_graph,
    mlir::AsmState &asm_state) {

  std::cout << "digraph G {" << std::endl;
  for (const auto &[source, sinks] : dependence_graph) {
    for (const auto &sink : sinks) {
      std::cout << node_name(source, asm_state) << " -> "
                << node_name(sink, asm_state) << ";\n";
    }
  }
  std::cout << "}" << std::endl;
}

int main(int argc, char **argv) {
  // Parse our command line operations.
  static cl::opt<std::string> inputFilename(cl::Positional,
                                            cl::desc("<input file>"));
  cl::ParseCommandLineOptions(argc, argv);

  // Exit early if the user didn't provide an input file.
  if (inputFilename.empty()) {
    llvm::outs() << "Please provide an input filename!\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  registerTritonDialects(registry);

  // Load the input and output files.
  std::string errorMessage;
  auto inputFile = mlir::openInputFile(inputFilename.getValue(), &errorMessage);
  if (!inputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(inputFile), mlir::SMLoc());

  mlir::MLIRContext ctx(registry, mlir::MLIRContext::Threading::DISABLED);
  mlir::FallbackAsmResourceMap fallbackResourceMap;
  mlir::ParserConfig parseConfig(&ctx, /*verifyAfterParse=*/true,
                                 &fallbackResourceMap);
  mlir::OwningOpRef<mlir::Operation *> op =
      mlir::parseSourceFile(sourceMgr, parseConfig);
  if (!op) {
    llvm::errs() << "Error parsing input program.\n";
    return 1;
  }

  // Run a few simplification passes on the IR. This is taken
  // from the TTIR construction done in
  // `third_party/nvidia/backend/compiler.py`.
  mlir::PassManager pm(op.get()->getName(),
                       mlir::PassManager::Nesting::Explicit);
  {
    pm.addPass(mlir::createInlinerPass());
    // This is done by the TTIR construction pass, but results
    // in a large amount of code expansion that isn't really
    // relevant for what we're trying to do.
    // pm.addPass(triton::createTritonRewriteTensorPointer());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(triton::createTritonLoopAwareCSE());
    pm.addPass(triton::createTritonCombineOps());
    pm.addPass(triton::createTritonReorderBroadcast());
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(mlir::createCanonicalizerPass());
  }
  if (mlir::failed(pm.run(*op))) {
    llvm::errs() << "Failed running pre-processing pass pipeline.\n";
    return 1;
  }

  std::unique_ptr<InstCostEstimator> estimator =
      std::make_unique<HopperCostEstimator>();

  llvm::DenseMap<mlir::Value, llvm::DenseSet<mlir::Value>> dependence_graph;
  // We now have op, which is an mlir::ModuleOp. As part of a normal
  // compiler, this logic would be extracted into a pass, but we can
  // do the manipulation inline here.
  op->walk([&](scf::ForOp forOp) {
    for (auto &op : forOp.getOps()) {
      for (auto result : op.getOperands()) {
        if (dependence_graph.find(result) == dependence_graph.end()) {
          dependence_graph[result] = llvm::DenseSet<mlir::Value>();
        }

        for (auto user : result.getUsers()) {
          for (auto userResult : user->getResults()) {
            dependence_graph[result].insert(userResult);
          }
        }

        // TODO: Make sure there aren't any other backedges
        if (op.getName().getStringRef() == "scf.yield") {
          mlir::Value yield_var = op.getOperand(0);
          mlir::Value loop_carried_var = forOp.getRegionIterArg(0);
          dependence_graph[yield_var].insert(loop_carried_var);
        }
      }
    }
  });

  // Print the entire module.
  mlir::AsmState asm_state(op.get(), mlir::OpPrintingFlags(),
                           /*locationMap=*/nullptr, &fallbackResourceMap);
  dump_dot_graph(dependence_graph, asm_state);
  op.get()->print(llvm::outs(), asm_state);

  return 0;
}
