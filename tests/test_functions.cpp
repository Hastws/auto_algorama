#include <gtest/gtest.h>
#include <cmath>
#include "common/types.h"
#include "functions/sigmoid.h"

namespace Autoalg {
namespace test {

// ============================================================================
// Sigmoid Function Tests
// ============================================================================
TEST(SigmoidTest, BasicValues) {
    // Sigmoid(0) should be 0.5
    EXPECT_NEAR(Sigmoid(0.0), 0.5, 1e-10);
    
    // Sigmoid(large positive) should approach 1
    EXPECT_NEAR(Sigmoid(10.0), 1.0, 1e-4);
    
    // Sigmoid(large negative) should approach 0
    EXPECT_NEAR(Sigmoid(-10.0), 0.0, 1e-4);
}

TEST(SigmoidTest, Symmetry) {
    // Sigmoid(x) + Sigmoid(-x) = 1
    for (double x = -5.0; x <= 5.0; x += 0.5) {
        EXPECT_NEAR(Sigmoid(x) + Sigmoid(-x), 1.0, 1e-10);
    }
}

TEST(SigmoidTest, Monotonicity) {
    // Sigmoid is strictly increasing
    for (double x = -5.0; x < 5.0; x += 0.5) {
        EXPECT_LT(Sigmoid(x), Sigmoid(x + 0.1));
    }
}

// ============================================================================
// Tanh Function Tests
// ============================================================================
TEST(TanhTest, BasicValues) {
    // Tanh(0) should be 0
    EXPECT_NEAR(Tanh(0.0), 0.0, 1e-10);
    
    // Tanh(large positive) should approach 1
    EXPECT_NEAR(Tanh(10.0), 1.0, 1e-4);
    
    // Tanh(large negative) should approach -1
    EXPECT_NEAR(Tanh(-10.0), -1.0, 1e-4);
}

TEST(TanhTest, OddFunction) {
    // Tanh(-x) = -Tanh(x)
    for (double x = -5.0; x <= 5.0; x += 0.5) {
        EXPECT_NEAR(Tanh(-x), -Tanh(x), 1e-10);
    }
}

TEST(TanhTest, Bounds) {
    // Tanh is bounded between -1 and 1
    for (double x = -10.0; x <= 10.0; x += 0.5) {
        EXPECT_GE(Tanh(x), -1.0);
        EXPECT_LE(Tanh(x), 1.0);
    }
}

// ============================================================================
// Probit Function Tests
// ============================================================================
TEST(ProbitTest, BasicValues) {
    // Probit(0) should be 0.5
    EXPECT_NEAR(Probit(0.0), 0.5, 1e-10);
    
    // Probit(large positive) should approach 1
    EXPECT_NEAR(Probit(5.0), 1.0, 1e-4);
    
    // Probit(large negative) should approach 0
    EXPECT_NEAR(Probit(-5.0), 0.0, 1e-4);
}

// ============================================================================
// ScaledTanh Function Tests
// ============================================================================
TEST(ScaledTanhTest, ZeroInput) {
    // ScaledTanh(0) should be 0
    EXPECT_NEAR(ScaledTanh(0.0), 0.0, 1e-10);
}

// ============================================================================
// BimodalSigmoid Function Tests
// ============================================================================
TEST(BimodalSigmoidTest, Bounds) {
    // BimodalSigmoid should be bounded between 0 and 1
    for (double x = -10.0; x <= 10.0; x += 0.5) {
        Real val = BimodalSigmoid(x);
        EXPECT_GE(val, 0.0);
        EXPECT_LE(val, 1.0);
    }
}

}  // namespace test
}  // namespace Autoalg
