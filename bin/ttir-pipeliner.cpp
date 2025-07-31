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

    auto gemm_tops = [](mlir::Type typ) -> int64_t {
      // * FP8 with * accumulate: 2000 TFlops
      // * FP16 with * accumulate: 1000 TFlops
      // * FP32 with * accumulate: 500 TFlops
      // * FP64 with * accumulate: 60 TFlops
      return llvm::TypeSwitch<mlir::Type, int64_t>(typ)
          .Case<mlir::IntegerType>([&](auto intty) -> int64_t {
            assert(false);
            return 0;
          })
          .Case<mlir::FloatType>([&](auto fty) -> int64_t {
            // TODO (rohany): This isn't enough for types like e5m3 or
            // whatever...
            switch (fty.getWidth()) {
            case 8:
              return 2000;
            case 16:
              return 1000;
            case 32:
              return 500;
            case 64:
              return 60;
            default: {
              assert(false);
              return 0;
            }
            }
          });
    };
    auto simt_tops = [](mlir::Type typ) -> int64_t {
      // Notes for non-tensor ops:
      // * FP16: 120 TFLOPS
      // * FP32: 60 TFLOPS
      // * FP64: 30 TFLOPS
      return llvm::TypeSwitch<mlir::Type, int64_t>(typ)
          .Case<mlir::IntegerType>([&](auto intty) -> int64_t {
            assert(false);
            return 0;
          })
          .Case<mlir::FloatType>([&](auto fty) -> int64_t {
            switch (fty.getWidth()) {
            // The SM can't do fp8 non-tensor core operations, or it
            // emulates them with fp16.
            case 8: // [fallthrough]
            case 16:
              return 120;
            case 32:
              return 60;
            case 64:
              return 30;
            default: {
              assert(false);
              return 0;
            }
            }
          });
    };
    auto sfu_tops = [](mlir::Type typ) -> int64_t {
      auto fty = llvm::dyn_cast<mlir::FloatType>(typ);
      assert(fty);
      // TODO (rohany): I wasn't able to derive these from scratch. I took
      //  the logic described in the FA3 paper, but I couldn't find the
      //  documentation that they referenced.
      switch (fty.getWidth()) {
      case 16:
        return 8;
      case 32:
        return 4;
      case 64:
        return 2;
      default: {
        assert(false);
        return 0;
      }
      }
    };

    return llvm::TypeSwitch<mlir::Operation *, int64_t>(op)
        // Matmul operations.
        .Case<triton::DotOpInterface>([&](auto dotop) {
          auto Aty = llvm::dyn_cast<mlir::TensorType>(dotop.getA().getType());
          auto Bty = llvm::dyn_cast<mlir::TensorType>(dotop.getB().getType());
          auto elemTy = Aty.getElementType();
          assert(elemTy == Bty.getElementType());
          assert(Aty.hasStaticShape() && Bty.hasStaticShape());
          int64_t M = Aty.getShape()[0];
          int64_t K = Aty.getShape()[1];
          int64_t N = Bty.getShape()[1];
          int64_t flops = 2 * M * N * K;
          return flops / gemm_tops(Aty.getElementType());
        })

        // Reductions.
        .Case<triton::ReduceOp>([&](auto redop) {
          // We can assume that the reduction here is a unit-cost arithmetic
          // operation, similarly costed to the operations below.
          assert(redop.getOperands().size() == 1);
          auto inputTy =
              llvm::dyn_cast<mlir::TensorType>(redop.getOperand(0).getType());
          assert(inputTy);
          int64_t axis = redop.getAxis();
          // A reduction on axis i with dimension n performs n-1 flops for
          // each element in the remaining dimensions.
          int64_t flops = 1;
          auto shape = inputTy.getShape();
          for (size_t i = 0; i < shape.size(); i++) {
            flops *= i == axis ? shape[i] - 1 : shape[i];
          }
          return flops / simt_tops(inputTy.getElementType());
        })
        // Arithmetic operations. These need cases internally
        // about whether they are operating on tensors or scalars.
        // If scalars. we can ignore them.
        .Case<arith::AddFOp, arith::MulFOp, arith::MaxNumFOp, arith::SubFOp,
              arith::TruncFOp, triton::FpToFpOp>([&](auto aop) -> int64_t {
          mlir::TensorType output =
              llvm::dyn_cast<mlir::TensorType>(aop.getResult().getType());
          if (!output) {
            return 0;
          }

          assert(output.hasStaticShape());
          int64_t elems = output.getNumElements();
          return elems / simt_tops(output.getElementType());
        })
        // Special math functions.
        .Case<math::Exp2Op>([&](auto expop) -> int64_t {
          mlir::TensorType output =
              llvm::dyn_cast<mlir::TensorType>(expop.getResult().getType());
          if (!output) {
            return 0;
          }
          assert(output.hasStaticShape());
          int64_t elems = output.getNumElements();
          return elems / sfu_tops(output.getElementType());
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
        .Case<triton::AdvanceOp, triton::SplatOp, triton::ExpandDimsOp,
              triton::BroadcastOp>([&](auto op) { return 0; })
        .Default([&](mlir::Operation *op) {
          llvm::errs() << "Unhandled operation in cost estimator: "
                       << op->getName().getStringRef() << "\n";
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
    for (auto &op : forOp.getOps()) {
      // op.dump();

      // Test that enough cases in the cost estimator are handled.
      if (!llvm::isa<scf::YieldOp>(op)) {
        auto cost = estimator->cost(&op);
        llvm::outs() << op.getName().getStringRef() << " ==> " << cost << "\n";
      }
    }
  });

  // Print the entire module.
  mlir::AsmState asmState(op.get(), mlir::OpPrintingFlags(),
                          /*locationMap=*/nullptr, &fallbackResourceMap);
  op.get()->print(llvm::outs(), asmState);

  return 0;
}
