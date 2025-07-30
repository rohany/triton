#include "./RegisterTritonDialects.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <iostream>
#include <memory>

using namespace mlir;
using namespace llvm;
using namespace mlir::triton;

class InstCostEstimator {
 public:
  virtual ~InstCostEstimator() {}
  virtual int64_t cost(mlir::Operation* op);
};

class HopperCostEstimator : public InstCostEstimator {
public:
  HopperCostEstimator() {}
  ~HopperCostEstimator() {}
  int64_t cost(mlir::Operation* op) override {
    return 0;
  }
};

class BlackwellCostEstimator : public InstCostEstimator {
public:
  BlackwellCostEstimator() {}
  ~BlackwellCostEstimator() {}
  int64_t cost(mlir::Operation* op) override {
    assert(false);
    return 0;
  }
};

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

  // We now have op, which is an mlir::ModuleOp. As part of a normal
  // compiler, this logic would be extracted into a pass, but we can
  // do the manipulation inline here.
  op->walk([&](scf::ForOp forOp) {
    // Dump the for loop.
    // forOp->dump();

    // Iterate through all operations in the for loop.
    // for (auto& op : forOp.getOps()) {
    //   op.dump();
    // }
  });

  // Print the entire module.
  mlir::AsmState asmState(op.get(), mlir::OpPrintingFlags(),
                          /*locationMap=*/nullptr, &fallbackResourceMap);
  op.get()->print(llvm::outs(), asmState);

  return 0;
}
