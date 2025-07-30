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
  virtual int64_t cost(mlir::Operation *op);
};

class HopperCostEstimator : public InstCostEstimator {
public:
  HopperCostEstimator() {}
  ~HopperCostEstimator() {}
  int64_t cost(mlir::Operation *op) override {
    return llvm::TypeSwitch<mlir::Operation *, int64_t>(op)
        // Matmul operations.
        .Case<triton::DotOpInterface>([&](auto dotop) { 
          return 1;
        })
        // Reductions.
        .Case<triton::ReduceOp>([&](auto redop) {
          // TODO (rohany): ...
          return 0;
        })
        // Arithmetic operations. These need cases internally
        // about whether they are operating on tensors or scalars.
        // If scalars. we can ignore them.
        .Case<arith::AddFOp>([&](auto addop) {
          // TODO (rohany): ...
          return 0;
        })
        .Case<arith::MulFOp>([&](auto mulop) {
          // TODO (rohany): ...
          return 0;
        })
        .Case<arith::MaxNumFOp>([&]( auto maxop) {
          // TODO (rohany): ...
          return 0;
        })
        .Case<arith::SubFOp>([&](auto subop) {
          // TODO (rohany): ...
          return 0;
        })
        .Case<arith::TruncFOp>([&](auto truncop) {
          // TODO (rohany): ...
          return 0;
        })
        // Special math functions.
        .Case<math::Exp2Op>([&](auto expop) {
          // TODO (rohany): ...
          return 0;
        })
        // GMEM -> SMEM loads, or SMEM- > GMEM stores.
        .Case<triton::LoadOp, triton::StoreOp>([&](auto memop) {
          // TODO (rohany): Not sure what to do yet with loads. Based on the
          //  model of the machine that we've been discussing, it might be
          //  feasible to pretend that loads actually have no latency, since
          //  we're going to put them into a separate warp anyway. Maybe we
          //  can pretend zero latency loads if we know the loads have no
          //  dependencies from tensor operations inside the loop? The same
          //  logic also goes for stores, where if there's nothing that depends
          //  on the store finishing we might be able to pretend it to not have
          //  any latency either...
          return 0;
        })
        // Triton operations that can be thought of as having 0 cost.
        // TODO (rohany): Not sure about splat here, because technically this
        //  counts as doing some register moves, or maybe a copy into a memory
        //  like TMEM.
        .Case<triton::AdvanceOp, triton::SplatOp, triton::ExpandDimsOp, triton::BroadcastOp>([&](auto op) {
          return 0;
        })
        .Default([&](mlir::Operation * op) {
          llvm::errs() << "Unhandled operation in cost estimator: " << op->getName().getStringRef() << "\n";
          assert(false);
          return 0;
        });
  }
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
      // op.dump();

      // Test that enough cases in the cost estimator are handled.
      // if (!llvm::isa<scf::YieldOp>(op)) {
      //   auto cost = estimator->cost(&op);
      //   llvm::outs() << op.getName().getStringRef() << " ==> " << cost << "\n";
      // }
    // }
  });

  // Print the entire module.
  mlir::AsmState asmState(op.get(), mlir::OpPrintingFlags(),
                          /*locationMap=*/nullptr, &fallbackResourceMap);
  op.get()->print(llvm::outs(), asmState);

  return 0;
}
