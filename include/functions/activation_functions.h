#ifndef INCLUDE_AUTOALG_ACTIVATION_FUNCTIONS_H
#define INCLUDE_AUTOALG_ACTIVATION_FUNCTIONS_H

#include "common/types.h"

namespace Autoalg {

// ============================================================================
// 1. SIGMOID AND RELATED FUNCTIONS
// ============================================================================

// Standard Sigmoid (Logistic)
inline Real Sigmoid(const Real &z) { return 1.0 / (1.0 + std::exp(-z)); }

// Hard Sigmoid
inline Real HardSigmoid(const Real &z) {
  if (z <= -2.5) return 0.0;
  if (z >= 2.5) return 1.0;
  return 0.2 * z + 0.5;
}

// Swish / SiLU (Sigmoid Linear Unit)
inline Real Swish(const Real &z) { return z * Sigmoid(z); }

// Hard Swish
inline Real HardSwish(const Real &z) {
  if (z <= -3.0) return 0.0;
  if (z >= 3.0) return z;
  return z * (z + 3.0) / 6.0;
}

// E-Swish
inline Real ESwish(const Real &z) {
  const Real beta = 1.25;
  return beta * z * Sigmoid(z);
}

// Flatten-T Swish
inline Real FlattenTSwish(const Real &z) {
  const Real T = 1.0;
  if (z < 0) return 0.0;
  return z * Sigmoid(z) + T;
}

// dSiLU (derivative of SiLU used as activation)
inline Real dSiLU(const Real &z) {
  Real sig = Sigmoid(z);
  return sig * (1.0 + z * (1.0 - sig));
}

// Parametric Swish
inline Real ParametricSwish(const Real &z) {
  const Real beta = 1.5;
  return z * Sigmoid(beta * z);
}

// ============================================================================
// 2. HYPERBOLIC TANGENT AND VARIANTS
// ============================================================================

// Standard Tanh
inline Real Tanh(const Real &z) { return std::tanh(z); }

// Hard Tanh
inline Real HardTanh(const Real &z) {
  if (z < -1.0) return -1.0;
  if (z > 1.0) return 1.0;
  return z;
}

// LeCun Tanh
inline Real LeCunTanh(const Real &z) {
  return 1.7159 * std::tanh(0.6667 * z);
}

// Scaled Tanh
inline Real ScaledTanh(const Real &z) {
  const Real a = 1.7159;
  const Real b = 2.0 / 3.0;
  return a * std::tanh(b * z);
}

// Penalized Tanh
inline Real PenalizedTanh(const Real &z) {
  const Real a = 0.25;
  if (z >= 0) return std::tanh(z);
  return a * std::tanh(z);
}

// TanhShrink
inline Real TanhShrink(const Real &z) { return z - std::tanh(z); }

// Arctanh (inverse hyperbolic tangent)
inline Real ArcTanh(const Real &z) {
  Real clamped = std::max(-0.999, std::min(0.999, z));
  return 0.5 * std::log((1.0 + clamped) / (1.0 - clamped));
}

// ============================================================================
// 3. ReLU FAMILY
// ============================================================================

// ReLU (Rectified Linear Unit)
inline Real ReLU(const Real &z) { return std::max(0.0, z); }

// Leaky ReLU
inline Real LeakyReLU(const Real &z) {
  const Real alpha = 0.01;
  return z > 0 ? z : alpha * z;
}

// Parametric ReLU (PReLU)
inline Real PReLU(const Real &z) {
  const Real alpha = 0.25;
  return z > 0 ? z : alpha * z;
}

// ELU (Exponential Linear Unit)
inline Real ELU(const Real &z) {
  const Real alpha = 1.0;
  return z > 0 ? z : alpha * (std::exp(z) - 1);
}

// SELU (Scaled Exponential Linear Unit)
inline Real SELU(const Real &z) {
  const Real lambda = 1.0507;
  const Real alpha = 1.6733;
  return z > 0 ? lambda * z : lambda * alpha * (std::exp(z) - 1);
}

// CELU (Continuously Differentiable ELU)
inline Real CELU(const Real &z) {
  const Real alpha = 1.0;
  return std::max(0.0, z) + std::min(0.0, alpha * (std::exp(z / alpha) - 1));
}

// GELU (Gaussian Error Linear Unit)
inline Real GELU(const Real &z) {
  return 0.5 * z * (1.0 + std::erf(z / std::sqrt(2.0)));
}

// GELU Approximation (Tanh)
inline Real GELUTanh(const Real &z) {
  const Real c = std::sqrt(2.0 / M_PI);
  return 0.5 * z * (1.0 + std::tanh(c * (z + 0.044715 * z * z * z)));
}

// SoftPlus
inline Real SoftPlus(const Real &z) { return std::log(1.0 + std::exp(z)); }

// Mish
inline Real Mish(const Real &z) { return z * std::tanh(SoftPlus(z)); }

// ReLU6
inline Real ReLU6(const Real &z) { return std::min(std::max(0.0, z), 6.0); }

// Thresholded ReLU
inline Real ThresholdedReLU(const Real &z) {
  const Real theta = 1.0;
  return z > theta ? z : 0.0;
}

// RReLU (Randomized ReLU - using fixed alpha for visualization)
inline Real RReLU(const Real &z) {
  const Real alpha = 0.125;  // midpoint of typical [0.1, 0.3] range
  return z > 0 ? z : alpha * z;
}

// Softplus Beta variant
inline Real SoftplusBeta(const Real &z) {
  const Real beta = 2.0;
  return std::log(1.0 + std::exp(beta * z)) / beta;
}

// Softmax Linear Unit (SoLU)
inline Real SoLU(const Real &z) {
  return z * std::exp(z) / (1.0 + std::exp(z));
}

// ============================================================================
// 4. EXPONENTIAL AND LOGARITHMIC FUNCTIONS
// ============================================================================

// Exponential Linear Squashing (ELiSH)
inline Real ELiSH(const Real &z) {
  if (z >= 0) return z / (1.0 + std::exp(-z));
  return (std::exp(z) - 1) / (1.0 + std::exp(-z));
}

// Hard ELiSH
inline Real HardELiSH(const Real &z) {
  if (z >= 0) return z * std::max(0.0, std::min(1.0, (z + 1) / 2.0));
  return (std::exp(z) - 1) * std::max(0.0, std::min(1.0, (z + 1) / 2.0));
}

// Soft Exponential
inline Real SoftExponential(const Real &z) {
  const Real alpha = 0.5;
  if (alpha < 0) return -std::log(1.0 - alpha * (z + alpha)) / alpha;
  if (alpha == 0) return z;
  return (std::exp(alpha * z) - 1) / alpha + alpha;
}

// LogSigmoid
inline Real LogSigmoid(const Real &z) { return std::log(Sigmoid(z)); }

// Log1p (log(1+x))
inline Real Log1p(const Real &z) { return std::log1p(std::max(-0.999, z)); }

// Exponential
inline Real Exponential(const Real &z) { return std::exp(z); }

// ============================================================================
// 5. GAUSSIAN AND RADIAL BASIS FUNCTIONS
// ============================================================================

// Gaussian
inline Real Gaussian(const Real &z) { return std::exp(-z * z); }

// Gaussian Error Linear Unit (approximation)
inline Real GaussianELU(const Real &z) {
  return z * std::exp(-z * z / 2.0);
}

// Growing Cosine Unit (GCU)
inline Real GCU(const Real &z) { return z * std::cos(z); }

// Sine Activation
inline Real SineActivation(const Real &z) { return std::sin(z); }

// Cosine Activation
inline Real CosineActivation(const Real &z) { return std::cos(z); }

// SiLU with Gaussian
inline Real GaussianSiLU(const Real &z) {
  return z * std::exp(-z * z) + Sigmoid(z);
}

// Double Gaussian
inline Real DoubleGaussian(const Real &z) {
  return std::exp(-z * z) - std::exp(-z * z / 2.0);
}

// ============================================================================
// 6. ADAPTIVE AND PARAMETRIC FUNCTIONS
// ============================================================================

// Softmax (for single value, acts as identity shifted)
inline Real SoftmaxSingle(const Real &z) { return std::exp(z); }

// Softsign
inline Real Softsign(const Real &z) { return z / (1.0 + std::abs(z)); }

// Bent Identity
inline Real BentIdentity(const Real &z) {
  return (std::sqrt(z * z + 1) - 1) / 2.0 + z;
}

// ArcTan
inline Real ArcTan(const Real &z) { return std::atan(z); }

// ArcSinh
inline Real ArcSinh(const Real &z) {
  return std::log(z + std::sqrt(z * z + 1));
}

// Sinc
inline Real Sinc(const Real &z) {
  if (std::abs(z) < 1e-10) return 1.0;
  return std::sin(z) / z;
}

// Sech (hyperbolic secant)
inline Real Sech(const Real &z) {
  return 2.0 / (std::exp(z) + std::exp(-z));
}

// ============================================================================
// 7. PIECE-WISE LINEAR FUNCTIONS
// ============================================================================

// Identity
inline Real Identity(const Real &z) { return z; }

// Binary Step
inline Real BinaryStep(const Real &z) { return z >= 0 ? 1.0 : 0.0; }

// Sign
inline Real Sign(const Real &z) {
  if (z > 0) return 1.0;
  if (z < 0) return -1.0;
  return 0.0;
}

// Absolute Value
inline Real AbsoluteValue(const Real &z) { return std::abs(z); }

// Maxout (simplified for 1D)
inline Real Maxout(const Real &z) {
  return std::max(z, 0.5 * z + 0.25);
}

// Symmetric Saturating Linear
inline Real SymmetricSaturating(const Real &z) {
  if (z < -1) return -1;
  if (z > 1) return 1;
  return z;
}

// ============================================================================
// 8. SMOOTH APPROXIMATIONS
// ============================================================================

// Smooth ReLU (SmoothReLU / Softplus variant)
inline Real SmoothReLU(const Real &z) {
  const Real beta = 1.0;
  return std::log(1.0 + std::exp(beta * z)) / beta;
}

// Smooth Absolute
inline Real SmoothAbs(const Real &z) {
  const Real eps = 0.1;
  return std::sqrt(z * z + eps);
}

// SoftShrink
inline Real SoftShrink(const Real &z) {
  const Real lambda = 0.5;
  if (z > lambda) return z - lambda;
  if (z < -lambda) return z + lambda;
  return 0.0;
}

// HardShrink
inline Real HardShrink(const Real &z) {
  const Real lambda = 0.5;
  if (z > lambda || z < -lambda) return z;
  return 0.0;
}

// SquarePlus
inline Real SquarePlus(const Real &z) {
  const Real b = 1.0;
  return (z + std::sqrt(z * z + b)) / 2.0;
}

// Smelu (Smooth Modulated ELU)
inline Real Smelu(const Real &z) {
  const Real beta = 0.5;
  if (z >= beta) return z;
  if (z <= -beta) return 0.0;
  return (z + beta) * (z + beta) / (4.0 * beta);
}

// ============================================================================
// 9. SPECIAL FUNCTIONS FROM LITERATURE
// ============================================================================

// Probit (inverse CDF of standard normal)
inline Real Probit(const Real &z) {
  return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
}

// Complementary Log-Log
inline Real CLogLog(const Real &z) {
  return 1.0 - std::exp(-std::exp(z));
}

// Log-Log
inline Real LogLog(const Real &z) { return std::exp(-std::exp(-z)); }

// Bi-modal Sigmoid
inline Real BimodalSigmoid(const Real &z) {
  const Real b = 2.0;
  return 0.5 * (Sigmoid(z) + Sigmoid(z - b));
}

// Shifted Scaled Sigmoid
inline Real ShiftedScaledSigmoid(const Real &z) {
  const Real a = 0.2;
  const Real b = 6.0;
  return 1.0 / (1.0 + std::exp(-a * (z - b)));
}

// Variant Sigmoid Function
inline Real VariantSigmoid(const Real &z) {
  const Real a = 1.0;
  const Real b = 5.0;
  const Real c = 0.5;
  return a / (1.0 + std::exp(-b * z)) - c;
}

// Bipolar Sigmoid
inline Real BipolarSigmoid(const Real &z) {
  return (1.0 - std::exp(-z)) / (1.0 + std::exp(-z));
}

// Gompertz Function
inline Real Gompertz(const Real &z) {
  const Real a = 1.0, b = 1.0, c = 1.0;
  return a * std::exp(-b * std::exp(-c * z));
}

// ============================================================================
// 10. MODERN ACTIVATION FUNCTIONS
// ============================================================================

// SiLU (same as Swish)
inline Real SiLU(const Real &z) { return z * Sigmoid(z); }

// LiSHT (Linearly Scaled Hyperbolic Tangent)
inline Real LiSHT(const Real &z) { return z * std::tanh(z); }

// Logit (inverse sigmoid)
inline Real Logit(const Real &z) {
  Real p = std::max(1e-10, std::min(1.0 - 1e-10, z));
  return std::log(p / (1.0 - p));
}

// Phish
inline Real Phish(const Real &z) {
  return z * std::tanh(GELU(z));
}

// SQNL (Square Nonlinearity)
inline Real SQNL(const Real &z) {
  if (z > 2.0) return 1.0;
  if (z >= 0.0) return z - z * z / 4.0;
  if (z >= -2.0) return z + z * z / 4.0;
  return -1.0;
}

// ISRU (Inverse Square Root Unit)
inline Real ISRU(const Real &z) {
  const Real alpha = 1.0;
  return z / std::sqrt(1.0 + alpha * z * z);
}

// ISRLU (Inverse Square Root Linear Unit)
inline Real ISRLU(const Real &z) {
  const Real alpha = 1.0;
  if (z >= 0) return z;
  return z / std::sqrt(1.0 + alpha * z * z);
}

// SReLU (S-shaped ReLU)
inline Real SReLU(const Real &z) {
  const Real tl = -0.5, tr = 0.5;
  const Real al = 0.01, ar = 0.01;
  if (z <= tl) return tl + al * (z - tl);
  if (z >= tr) return tr + ar * (z - tr);
  return z;
}

// BReLU (Bipolar ReLU)
inline Real BReLU(const Real &z) {
  if (z >= 0) return z;
  return z;  // Simplified version
}

// APL (Adaptive Piecewise Linear)
inline Real APL(const Real &z) {
  const Real a1 = 0.5, b1 = 1.0;
  return std::max(0.0, z) + a1 * std::max(0.0, -z + b1);
}

// Smish
inline Real Smish(const Real &z) {
  return z * std::tanh(std::log(1.0 + Sigmoid(z)));
}

// Logish
inline Real Logish(const Real &z) {
  return z * std::log(1.0 + Sigmoid(z));
}

// TanhExp
inline Real TanhExp(const Real &z) {
  return z * std::tanh(std::exp(z));
}

// ============================================================================
// 11. ATTENTION AND TRANSFORMER RELATED
// ============================================================================

// QuickGELU
inline Real QuickGELU(const Real &z) {
  return z * Sigmoid(1.702 * z);
}

// GEGLU (simplified for 1D)
inline Real GEGLU(const Real &z) {
  return z * GELU(z);
}

// ReGLU (simplified for 1D)
inline Real ReGLU(const Real &z) {
  return z * ReLU(z);
}

// SwiGLU (simplified for 1D)
inline Real SwiGLU(const Real &z) {
  return z * Swish(z);
}

// Laplace activation
inline Real Laplace(const Real &z) {
  const Real mu = 0.707107, sigma = 0.282095;
  return 0.5 * (1.0 + std::erf((z - mu) / (sigma * std::sqrt(2.0))));
}

// ============================================================================
// 12. ADDITIONAL SPECIALIZED FUNCTIONS
// ============================================================================

// Elliott
inline Real Elliott(const Real &z) {
  return z / (1.0 + std::abs(z));
}

// SoftClipping
inline Real SoftClipping(const Real &z) {
  const Real alpha = 0.5;
  return (1.0 / alpha) * std::log((1.0 + std::exp(alpha * z)) / 
                                   (1.0 + std::exp(alpha * (z - 1.0))));
}

// Hexpo (Hyperbolic Exponential)
inline Real Hexpo(const Real &z) {
  if (z >= 0) return -std::exp(-z) + 1;
  return std::exp(z) - 1;
}

// NCU (Non-monotonic Cubic Unit)
inline Real NCU(const Real &z) {
  return z - z * z * z;
}

// DSU (Decaying Sine Unit)
inline Real DSU(const Real &z) {
  const Real pi = M_PI;
  return pi / 2.0 * (Sinc(z - pi) - Sinc(z + pi));
}

// SERLU (Scaled Exponential Rectified Linear Unit)
inline Real SERLU(const Real &z) {
  const Real lambda = 1.07862;
  const Real alpha = 2.90427;
  if (z >= 0) return lambda * z;
  return lambda * alpha * (std::exp(z) - 1);
}

// PAU (Padé Activation Unit) - simplified
inline Real PAU(const Real &z) {
  // Approximation using rational function
  Real num = z + 0.5 * z * z;
  Real den = 1.0 + std::abs(z) + 0.5 * z * z;
  return num / den;
}

// Snake activation
inline Real Snake(const Real &z) {
  const Real a = 1.0;
  return z + (1.0 / a) * std::sin(a * z) * std::sin(a * z);
}

// ============================================================================
// 13. POLYNOMIAL ACTIVATIONS
// ============================================================================

// Cube
inline Real Cube(const Real &z) { return z * z * z; }

// Square
inline Real Square(const Real &z) { return z * z; }

// CReLU (Concatenated ReLU - simplified)
inline Real CReLU(const Real &z) {
  return std::max(0.0, z);
}

// Quartic
inline Real Quartic(const Real &z) { return z * z * z * z; }

// ============================================================================
// 14. PROBABILISTIC FUNCTIONS
// ============================================================================

// Logistic CDF (same as Sigmoid)
inline Real LogisticCDF(const Real &z) {
  return 1.0 / (1.0 + std::exp(-z));
}

// Normal CDF (Probit function)
inline Real NormalCDF(const Real &z) {
  return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
}

// Cauchy CDF
inline Real CauchyCDF(const Real &z) {
  return std::atan(z) / M_PI + 0.5;
}

// Gumbel CDF
inline Real GumbelCDF(const Real &z) {
  return std::exp(-std::exp(-z));
}

// Weibull-like
inline Real WeibullLike(const Real &z) {
  if (z < 0) return 0.0;
  const Real k = 2.0;
  return 1.0 - std::exp(-std::pow(z, k));
}

// ============================================================================
// 15. RECENTLY PROPOSED FUNCTIONS
// ============================================================================

// FReLU (Flexible ReLU)
inline Real FReLU(const Real &z) {
  return std::max(z, std::tanh(z));
}

// StarReLU
inline Real StarReLU(const Real &z) {
  const Real s = 0.8944, b = -0.4472;
  return s * ReLU(z) * ReLU(z) + b;
}

// Serf
inline Real Serf(const Real &z) {
  return z * std::erf(SoftPlus(z));
}

// ACON-C
inline Real ACONC(const Real &z) {
  const Real p1 = 1.0, p2 = 0.0;
  Real sig = Sigmoid(z);
  return (p1 - p2) * z * sig + p2 * z;
}

// MetaACON
inline Real MetaACON(const Real &z) {
  const Real beta = 1.5;
  Real sig = Sigmoid(beta * z);
  return z * (sig + (1.0 - sig) * 0.25);
}

// ELish Swish variant
inline Real ELishSwish(const Real &z) {
  if (z >= 0) return Swish(z);
  return (std::exp(z) - 1) * Sigmoid(z);
}

// Shifted ReLU
inline Real ShiftedReLU(const Real &z) {
  const Real shift = -0.5;
  return std::max(0.0, z + shift);
}

// Maxsig
inline Real Maxsig(const Real &z) {
  return std::max(z, Sigmoid(z));
}

// ============================================================================
// 16. DERIVATIVE APPROXIMATION HELPER
// ============================================================================

// Numerical derivative (for visualization)
inline Real NumericalDerivative(const std::function<Real(Real)>& func, 
                                const Real &z, Real h = 1e-5) {
  return (func(z + h) - func(z - h)) / (2.0 * h);
}

}  // namespace Autoalg

#endif  // INCLUDE_AUTOALG_ACTIVATION_FUNCTIONS_H
