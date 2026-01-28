#include "windows/main_window.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>

#include "common/log.h"
#include "functions/functions.h"
#include "imgui_internal.h"

namespace Autoalg {

// ============================================================================
// Global State Variables
// ============================================================================
static float amplitude = 1.0f;
static float frequency = 1.0f;
static int func_type = 0;

// View state
static float x_min = -5.0f;
static float x_max = 5.0f;
static float y_min = -2.0f;
static float y_max = 2.0f;
static int x_ticks = 10;
static int y_ticks = 10;

// Interaction state
static bool is_dragging = false;
static ImVec2 drag_start_pos;
static float drag_start_x_min, drag_start_x_max;
static float drag_start_y_min, drag_start_y_max;

// Display options
static bool show_derivative = false;
static bool show_grid = true;
static bool show_axis_labels = true;
static bool show_function_points = false;
static int samples = 500;
static float line_thickness = 2.0f;

// Comparison mode
static std::set<int> comparison_functions;
static bool comparison_mode = false;

// Color palette for multiple functions
static const ImU32 function_colors[] = {
    IM_COL32(255, 120, 120, 255),  // Red
    IM_COL32(120, 255, 120, 255),  // Green
    IM_COL32(120, 120, 255, 255),  // Blue
    IM_COL32(255, 255, 120, 255),  // Yellow
    IM_COL32(255, 120, 255, 255),  // Magenta
    IM_COL32(120, 255, 255, 255),  // Cyan
    IM_COL32(255, 180, 120, 255),  // Orange
    IM_COL32(180, 120, 255, 255),  // Purple
    IM_COL32(120, 255, 180, 255),  // Mint
    IM_COL32(255, 120, 180, 255),  // Pink
};
static const int num_colors = sizeof(function_colors) / sizeof(function_colors[0]);

// Category filter
static int current_category = 0;

// ============================================================================
// Register All Activation Functions
// ============================================================================

// Sigmoid Family
REGISTER_FUNCTION_WITH_CATEGORY(Sigmoid, ::Autoalg::Sigmoid,
                                FunctionCategory::SIGMOID_FAMILY, "σ(x) = 1/(1+e^(-x))");
REGISTER_FUNCTION_WITH_CATEGORY(HardSigmoid, ::Autoalg::HardSigmoid,
                                FunctionCategory::SIGMOID_FAMILY, "Linear approximation of sigmoid");
REGISTER_FUNCTION_WITH_CATEGORY(Swish, ::Autoalg::Swish,
                                FunctionCategory::SIGMOID_FAMILY, "x·σ(x), also known as SiLU");
REGISTER_FUNCTION_WITH_CATEGORY(HardSwish, ::Autoalg::HardSwish,
                                FunctionCategory::SIGMOID_FAMILY, "Efficient approximation of Swish");
REGISTER_FUNCTION_WITH_CATEGORY(ESwish, ::Autoalg::ESwish,
                                FunctionCategory::SIGMOID_FAMILY, "β·x·σ(x)");
REGISTER_FUNCTION_WITH_CATEGORY(LogSigmoid, ::Autoalg::LogSigmoid,
                                FunctionCategory::SIGMOID_FAMILY, "log(σ(x))");

// Tanh Family
REGISTER_FUNCTION_WITH_CATEGORY(Tanh, ::Autoalg::Tanh,
                                FunctionCategory::TANH_FAMILY, "Hyperbolic tangent");
REGISTER_FUNCTION_WITH_CATEGORY(HardTanh, ::Autoalg::HardTanh,
                                FunctionCategory::TANH_FAMILY, "Clamped linear: max(-1, min(1, x))");
REGISTER_FUNCTION_WITH_CATEGORY(LeCunTanh, ::Autoalg::LeCunTanh,
                                FunctionCategory::TANH_FAMILY, "1.7159·tanh(0.6667·x)");
REGISTER_FUNCTION_WITH_CATEGORY(ScaledTanh, ::Autoalg::ScaledTanh,
                                FunctionCategory::TANH_FAMILY, "a·tanh(b·x)");
REGISTER_FUNCTION_WITH_CATEGORY(PenalizedTanh, ::Autoalg::PenalizedTanh,
                                FunctionCategory::TANH_FAMILY, "Asymmetric tanh");
REGISTER_FUNCTION_WITH_CATEGORY(TanhShrink, ::Autoalg::TanhShrink,
                                FunctionCategory::TANH_FAMILY, "x - tanh(x)");
REGISTER_FUNCTION_WITH_CATEGORY(LiSHT, ::Autoalg::LiSHT,
                                FunctionCategory::TANH_FAMILY, "x·tanh(x)");

// ReLU Family
REGISTER_FUNCTION_WITH_CATEGORY(ReLU, ::Autoalg::ReLU,
                                FunctionCategory::RELU_FAMILY, "max(0, x)");
REGISTER_FUNCTION_WITH_CATEGORY(LeakyReLU, ::Autoalg::LeakyReLU,
                                FunctionCategory::RELU_FAMILY, "max(αx, x), α=0.01");
REGISTER_FUNCTION_WITH_CATEGORY(PReLU, ::Autoalg::PReLU,
                                FunctionCategory::RELU_FAMILY, "Parametric ReLU, α=0.25");
REGISTER_FUNCTION_WITH_CATEGORY(ELU, ::Autoalg::ELU,
                                FunctionCategory::RELU_FAMILY, "Exponential Linear Unit");
REGISTER_FUNCTION_WITH_CATEGORY(SELU, ::Autoalg::SELU,
                                FunctionCategory::RELU_FAMILY, "Scaled ELU for self-normalization");
REGISTER_FUNCTION_WITH_CATEGORY(CELU, ::Autoalg::CELU,
                                FunctionCategory::RELU_FAMILY, "Continuously Differentiable ELU");
REGISTER_FUNCTION_WITH_CATEGORY(GELU, ::Autoalg::GELU,
                                FunctionCategory::RELU_FAMILY, "Gaussian Error Linear Unit");
REGISTER_FUNCTION_WITH_CATEGORY(GELUTanh, ::Autoalg::GELUTanh,
                                FunctionCategory::RELU_FAMILY, "GELU tanh approximation");
REGISTER_FUNCTION_WITH_CATEGORY(SoftPlus, ::Autoalg::SoftPlus,
                                FunctionCategory::RELU_FAMILY, "log(1 + e^x)");
REGISTER_FUNCTION_WITH_CATEGORY(Mish, ::Autoalg::Mish,
                                FunctionCategory::RELU_FAMILY, "x·tanh(softplus(x))");
REGISTER_FUNCTION_WITH_CATEGORY(ReLU6, ::Autoalg::ReLU6,
                                FunctionCategory::RELU_FAMILY, "min(max(0, x), 6)");
REGISTER_FUNCTION_WITH_CATEGORY(ThresholdedReLU, ::Autoalg::ThresholdedReLU,
                                FunctionCategory::RELU_FAMILY, "x if x>θ else 0");
REGISTER_FUNCTION_WITH_CATEGORY(SReLU, ::Autoalg::SReLU,
                                FunctionCategory::RELU_FAMILY, "S-shaped ReLU");
REGISTER_FUNCTION_WITH_CATEGORY(ISRU, ::Autoalg::ISRU,
                                FunctionCategory::RELU_FAMILY, "x/√(1+αx²)");
REGISTER_FUNCTION_WITH_CATEGORY(ISRLU, ::Autoalg::ISRLU,
                                FunctionCategory::RELU_FAMILY, "Inverse Square Root Linear Unit");
REGISTER_FUNCTION_WITH_CATEGORY(SERLU, ::Autoalg::SERLU,
                                FunctionCategory::RELU_FAMILY, "Scaled Exponential Rectified Linear");

// Exponential Functions
REGISTER_FUNCTION_WITH_CATEGORY(ELiSH, ::Autoalg::ELiSH,
                                FunctionCategory::EXPONENTIAL, "Exponential Linear Squashing");
REGISTER_FUNCTION_WITH_CATEGORY(HardELiSH, ::Autoalg::HardELiSH,
                                FunctionCategory::EXPONENTIAL, "Hard ELiSH");
REGISTER_FUNCTION_WITH_CATEGORY(SoftExponential, ::Autoalg::SoftExponential,
                                FunctionCategory::EXPONENTIAL, "Parametric soft exponential");
REGISTER_FUNCTION_WITH_CATEGORY(Hexpo, ::Autoalg::Hexpo,
                                FunctionCategory::EXPONENTIAL, "Hyperbolic Exponential");

// Gaussian/Radial Functions
REGISTER_FUNCTION_WITH_CATEGORY(Gaussian, ::Autoalg::Gaussian,
                                FunctionCategory::GAUSSIAN, "e^(-x²)");
REGISTER_FUNCTION_WITH_CATEGORY(GaussianELU, ::Autoalg::GaussianELU,
                                FunctionCategory::GAUSSIAN, "x·e^(-x²/2)");
REGISTER_FUNCTION_WITH_CATEGORY(GCU, ::Autoalg::GCU,
                                FunctionCategory::GAUSSIAN, "x·cos(x)");
REGISTER_FUNCTION_WITH_CATEGORY(SineActivation, ::Autoalg::SineActivation,
                                FunctionCategory::GAUSSIAN, "sin(x)");
REGISTER_FUNCTION_WITH_CATEGORY(CosineActivation, ::Autoalg::CosineActivation,
                                FunctionCategory::GAUSSIAN, "cos(x)");
REGISTER_FUNCTION_WITH_CATEGORY(Sinc, ::Autoalg::Sinc,
                                FunctionCategory::GAUSSIAN, "sin(x)/x");

// Adaptive Functions
REGISTER_FUNCTION_WITH_CATEGORY(Softsign, ::Autoalg::Softsign,
                                FunctionCategory::ADAPTIVE, "x/(1+|x|)");
REGISTER_FUNCTION_WITH_CATEGORY(BentIdentity, ::Autoalg::BentIdentity,
                                FunctionCategory::ADAPTIVE, "(√(x²+1)-1)/2 + x");
REGISTER_FUNCTION_WITH_CATEGORY(ArcTan, ::Autoalg::ArcTan,
                                FunctionCategory::ADAPTIVE, "arctan(x)");
REGISTER_FUNCTION_WITH_CATEGORY(ArcSinh, ::Autoalg::ArcSinh,
                                FunctionCategory::ADAPTIVE, "asinh(x)");
REGISTER_FUNCTION_WITH_CATEGORY(Elliott, ::Autoalg::Elliott,
                                FunctionCategory::ADAPTIVE, "x/(1+|x|)");
REGISTER_FUNCTION_WITH_CATEGORY(SQNL, ::Autoalg::SQNL,
                                FunctionCategory::ADAPTIVE, "Square Nonlinearity");
REGISTER_FUNCTION_WITH_CATEGORY(APL, ::Autoalg::APL,
                                FunctionCategory::ADAPTIVE, "Adaptive Piecewise Linear");

// Piecewise Linear
REGISTER_FUNCTION_WITH_CATEGORY(Identity, ::Autoalg::Identity,
                                FunctionCategory::PIECEWISE, "f(x) = x");
REGISTER_FUNCTION_WITH_CATEGORY(BinaryStep, ::Autoalg::BinaryStep,
                                FunctionCategory::PIECEWISE, "0 if x<0 else 1");
REGISTER_FUNCTION_WITH_CATEGORY(Sign, ::Autoalg::Sign,
                                FunctionCategory::PIECEWISE, "Sign function");
REGISTER_FUNCTION_WITH_CATEGORY(AbsoluteValue, ::Autoalg::AbsoluteValue,
                                FunctionCategory::PIECEWISE, "|x|");
REGISTER_FUNCTION_WITH_CATEGORY(Maxout, ::Autoalg::Maxout,
                                FunctionCategory::PIECEWISE, "max(x, 0.5x+0.25)");

// Smooth Approximations
REGISTER_FUNCTION_WITH_CATEGORY(SmoothReLU, ::Autoalg::SmoothReLU,
                                FunctionCategory::SMOOTH, "Smooth approximation of ReLU");
REGISTER_FUNCTION_WITH_CATEGORY(SmoothAbs, ::Autoalg::SmoothAbs,
                                FunctionCategory::SMOOTH, "√(x²+ε)");
REGISTER_FUNCTION_WITH_CATEGORY(SoftShrink, ::Autoalg::SoftShrink,
                                FunctionCategory::SMOOTH, "Soft shrinkage");
REGISTER_FUNCTION_WITH_CATEGORY(HardShrink, ::Autoalg::HardShrink,
                                FunctionCategory::SMOOTH, "Hard shrinkage");
REGISTER_FUNCTION_WITH_CATEGORY(SquarePlus, ::Autoalg::SquarePlus,
                                FunctionCategory::SMOOTH, "(x+√(x²+b))/2");
REGISTER_FUNCTION_WITH_CATEGORY(Smelu, ::Autoalg::Smelu,
                                FunctionCategory::SMOOTH, "Smooth Modulated ELU");

// Special Functions
REGISTER_FUNCTION_WITH_CATEGORY(Probit, ::Autoalg::Probit,
                                FunctionCategory::SPECIAL, "Inverse normal CDF");
REGISTER_FUNCTION_WITH_CATEGORY(CLogLog, ::Autoalg::CLogLog,
                                FunctionCategory::SPECIAL, "1 - e^(-e^x)");
REGISTER_FUNCTION_WITH_CATEGORY(LogLog, ::Autoalg::LogLog,
                                FunctionCategory::SPECIAL, "e^(-e^(-x))");
REGISTER_FUNCTION_WITH_CATEGORY(BimodalSigmoid, ::Autoalg::BimodalSigmoid,
                                FunctionCategory::SPECIAL, "Bi-modal sigmoid");
REGISTER_FUNCTION_WITH_CATEGORY(ShiftedScaledSigmoid, ::Autoalg::ShiftedScaledSigmoid,
                                FunctionCategory::SPECIAL, "Shifted and scaled sigmoid");
REGISTER_FUNCTION_WITH_CATEGORY(VariantSigmoid, ::Autoalg::VariantSigmoid,
                                FunctionCategory::SPECIAL, "Variant sigmoid function");
REGISTER_FUNCTION_WITH_CATEGORY(SoftClipping, ::Autoalg::SoftClipping,
                                FunctionCategory::SPECIAL, "Soft clipping function");
REGISTER_FUNCTION_WITH_CATEGORY(BipolarSigmoid, ::Autoalg::BipolarSigmoid,
                                FunctionCategory::SPECIAL, "(1-e^-x)/(1+e^-x)");
REGISTER_FUNCTION_WITH_CATEGORY(Gompertz, ::Autoalg::Gompertz,
                                FunctionCategory::SPECIAL, "Gompertz growth function");

// Modern Functions
REGISTER_FUNCTION_WITH_CATEGORY(SiLU, ::Autoalg::SiLU,
                                FunctionCategory::MODERN, "Sigmoid Linear Unit");
REGISTER_FUNCTION_WITH_CATEGORY(Phish, ::Autoalg::Phish,
                                FunctionCategory::MODERN, "x·tanh(GELU(x))");
REGISTER_FUNCTION_WITH_CATEGORY(NCU, ::Autoalg::NCU,
                                FunctionCategory::MODERN, "x - x³");
REGISTER_FUNCTION_WITH_CATEGORY(DSU, ::Autoalg::DSU,
                                FunctionCategory::MODERN, "Decaying Sine Unit");
REGISTER_FUNCTION_WITH_CATEGORY(Smish, ::Autoalg::Smish,
                                FunctionCategory::MODERN, "x·tanh(log(1+σ(x)))");
REGISTER_FUNCTION_WITH_CATEGORY(Logish, ::Autoalg::Logish,
                                FunctionCategory::MODERN, "x·log(1+σ(x))");
REGISTER_FUNCTION_WITH_CATEGORY(TanhExp, ::Autoalg::TanhExp,
                                FunctionCategory::MODERN, "x·tanh(e^x)");
REGISTER_FUNCTION_WITH_CATEGORY(Snake, ::Autoalg::Snake,
                                FunctionCategory::MODERN, "x + sin²(ax)/a");
REGISTER_FUNCTION_WITH_CATEGORY(PAU, ::Autoalg::PAU,
                                FunctionCategory::MODERN, "Padé Activation Unit");
REGISTER_FUNCTION_WITH_CATEGORY(FReLU, ::Autoalg::FReLU,
                                FunctionCategory::MODERN, "max(x, tanh(x))");
REGISTER_FUNCTION_WITH_CATEGORY(StarReLU, ::Autoalg::StarReLU,
                                FunctionCategory::MODERN, "s·ReLU²(x) + b");
REGISTER_FUNCTION_WITH_CATEGORY(Serf, ::Autoalg::Serf,
                                FunctionCategory::MODERN, "x·erf(softplus(x))");
REGISTER_FUNCTION_WITH_CATEGORY(ACONC, ::Autoalg::ACONC,
                                FunctionCategory::MODERN, "ACON-C activation");
REGISTER_FUNCTION_WITH_CATEGORY(MetaACON, ::Autoalg::MetaACON,
                                FunctionCategory::MODERN, "Meta-ACON activation");
REGISTER_FUNCTION_WITH_CATEGORY(Maxsig, ::Autoalg::Maxsig,
                                FunctionCategory::MODERN, "max(x, σ(x))");

// Attention/Transformer
REGISTER_FUNCTION_WITH_CATEGORY(QuickGELU, ::Autoalg::QuickGELU,
                                FunctionCategory::ATTENTION, "x·σ(1.702x)");
REGISTER_FUNCTION_WITH_CATEGORY(GEGLU, ::Autoalg::GEGLU,
                                FunctionCategory::ATTENTION, "x·GELU(x)");
REGISTER_FUNCTION_WITH_CATEGORY(ReGLU, ::Autoalg::ReGLU,
                                FunctionCategory::ATTENTION, "x·ReLU(x)");
REGISTER_FUNCTION_WITH_CATEGORY(SwiGLU, ::Autoalg::SwiGLU,
                                FunctionCategory::ATTENTION, "x·Swish(x)");
REGISTER_FUNCTION_WITH_CATEGORY(Laplace, ::Autoalg::Laplace,
                                FunctionCategory::ATTENTION, "Laplace activation");

// Polynomial
REGISTER_FUNCTION_WITH_CATEGORY(Cube, ::Autoalg::Cube,
                                FunctionCategory::POLYNOMIAL, "x³");
REGISTER_FUNCTION_WITH_CATEGORY(Square, ::Autoalg::Square,
                                FunctionCategory::POLYNOMIAL, "x²");
REGISTER_FUNCTION_WITH_CATEGORY(Quartic, ::Autoalg::Quartic,
                                FunctionCategory::POLYNOMIAL, "x⁴");

// Probabilistic
REGISTER_FUNCTION_WITH_CATEGORY(LogisticCDF, ::Autoalg::LogisticCDF,
                                FunctionCategory::PROBABILISTIC, "Logistic CDF");
REGISTER_FUNCTION_WITH_CATEGORY(NormalCDF, ::Autoalg::NormalCDF,
                                FunctionCategory::PROBABILISTIC, "Normal CDF (Φ)");
REGISTER_FUNCTION_WITH_CATEGORY(CauchyCDF, ::Autoalg::CauchyCDF,
                                FunctionCategory::PROBABILISTIC, "Cauchy CDF");
REGISTER_FUNCTION_WITH_CATEGORY(GumbelCDF, ::Autoalg::GumbelCDF,
                                FunctionCategory::PROBABILISTIC, "Gumbel CDF");
REGISTER_FUNCTION_WITH_CATEGORY(WeibullLike, ::Autoalg::WeibullLike,
                                FunctionCategory::PROBABILISTIC, "Weibull-like CDF");

// Additional ReLU variants
REGISTER_FUNCTION_WITH_CATEGORY(RReLU, ::Autoalg::RReLU,
                                FunctionCategory::RELU_FAMILY, "Randomized ReLU");
REGISTER_FUNCTION_WITH_CATEGORY(SoftplusBeta, ::Autoalg::SoftplusBeta,
                                FunctionCategory::RELU_FAMILY, "Softplus with β=2");
REGISTER_FUNCTION_WITH_CATEGORY(SoLU, ::Autoalg::SoLU,
                                FunctionCategory::RELU_FAMILY, "Softmax Linear Unit");
REGISTER_FUNCTION_WITH_CATEGORY(ShiftedReLU, ::Autoalg::ShiftedReLU,
                                FunctionCategory::RELU_FAMILY, "ReLU with shift");
REGISTER_FUNCTION_WITH_CATEGORY(ELishSwish, ::Autoalg::ELishSwish,
                                FunctionCategory::RELU_FAMILY, "ELiSH-Swish hybrid");

// Additional Sigmoid variants
REGISTER_FUNCTION_WITH_CATEGORY(dSiLU, ::Autoalg::dSiLU,
                                FunctionCategory::SIGMOID_FAMILY, "Derivative of SiLU");
REGISTER_FUNCTION_WITH_CATEGORY(ParametricSwish, ::Autoalg::ParametricSwish,
                                FunctionCategory::SIGMOID_FAMILY, "x·σ(βx), β=1.5");

// Additional Gaussian
REGISTER_FUNCTION_WITH_CATEGORY(GaussianSiLU, ::Autoalg::GaussianSiLU,
                                FunctionCategory::GAUSSIAN, "Gaussian + SiLU hybrid");
REGISTER_FUNCTION_WITH_CATEGORY(DoubleGaussian, ::Autoalg::DoubleGaussian,
                                FunctionCategory::GAUSSIAN, "Difference of Gaussians");
REGISTER_FUNCTION_WITH_CATEGORY(Sech, ::Autoalg::Sech,
                                FunctionCategory::GAUSSIAN, "Hyperbolic secant");

// Additional Piecewise
REGISTER_FUNCTION_WITH_CATEGORY(SymmetricSaturating, ::Autoalg::SymmetricSaturating,
                                FunctionCategory::PIECEWISE, "Symmetric saturating linear");

// Additional Special
REGISTER_FUNCTION_WITH_CATEGORY(Log1p, ::Autoalg::Log1p,
                                FunctionCategory::EXPONENTIAL, "log(1+x)");
REGISTER_FUNCTION_WITH_CATEGORY(Exponential, ::Autoalg::Exponential,
                                FunctionCategory::EXPONENTIAL, "e^x");

// ============================================================================
// Helper Functions
// ============================================================================

double EvaluateFunction(int func, float x, float amp, float freq) {
  auto name = FunctionsManager::Instance().GetFunctionName(func);
  return amp * FunctionsManager::Instance().Call(name, freq * x);
}

double EvaluateDerivative(int func, float x, float amp, float freq) {
  const float h = 0.0001f;
  double f_plus = EvaluateFunction(func, x + h, amp, freq);
  double f_minus = EvaluateFunction(func, x - h, amp, freq);
  return (f_plus - f_minus) / (2.0 * h);
}

void ResetView() {
  x_min = -5.0f;
  x_max = 5.0f;
  y_min = -2.0f;
  y_max = 2.0f;
}

void ZoomView(float factor, float center_x, float center_y) {
  float x_range = x_max - x_min;
  float y_range = y_max - y_min;
  
  float new_x_range = x_range * factor;
  float new_y_range = y_range * factor;
  
  // Zoom centered on the given point
  float x_ratio = (center_x - x_min) / x_range;
  float y_ratio = (center_y - y_min) / y_range;
  
  x_min = center_x - x_ratio * new_x_range;
  x_max = center_x + (1.0f - x_ratio) * new_x_range;
  y_min = center_y - y_ratio * new_y_range;
  y_max = center_y + (1.0f - y_ratio) * new_y_range;
}

// ============================================================================
// MainWindow Implementation
// ============================================================================

MainWindow::MainWindow(const int width, const int height)
    : width_(width), height_(height), window_(nullptr) {
  DEBUG(MainWindow) << "initialized.";
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_EVENTS) != 0) {
    ERROR(MainWindow) << "SDL_Init failed: " << SDL_GetError();
  } else {
    const char *glsl_version = "#version 150";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                        SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

    window_ = SDL_CreateWindow(
        "Activation Functions Visualizer - 400 Functions Survey", 
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width_,
        height_,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    gl_context_ = SDL_GL_CreateContext(window_);
    SDL_GL_MakeCurrent(window_, gl_context_);
    SDL_GL_SetSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui_ImplSDL2_InitForOpenGL(window_, gl_context_);
    ImGui_ImplOpenGL3_Init(glsl_version);
    
    // Set dark theme
    ImGui::StyleColorsDark();
  }
}

int MainWindow::Process() const {
  DEBUG(MainWindow) << "processing.";
  if (!window_) {
    return 1;
  }
  constexpr ImVec4 clear_color = ImVec4(0.08f, 0.08f, 0.12f, 1.0f);
  bool done = false;
  
  while (!done) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      ImGui_ImplSDL2_ProcessEvent(&event);
      if (event.type == SDL_QUIT) done = true;
      if (event.type == SDL_WINDOWEVENT &&
          event.window.event == SDL_WINDOWEVENT_CLOSE &&
          event.window.windowID == SDL_GetWindowID(window_))
        done = true;
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    ImGuiIO &io = ImGui::GetIO();
    ImDrawList *bg = ImGui::GetBackgroundDrawList();
    ImGuiViewport *vp = ImGui::GetMainViewport();
    ImVec2 p0 = vp->Pos;
    ImVec2 p1 = ImVec2(vp->Pos.x + vp->Size.x, vp->Pos.y + vp->Size.y);
    ImVec2 dispSize = vp->Size;

    float x_range = x_max - x_min;
    float y_range = y_max - y_min;

    // Background
    bg->AddRectFilled(p0, p1, IM_COL32(15, 15, 25, 255));

    // Grid
    if (show_grid) {
      for (int i = 0; i <= x_ticks; ++i) {
        float t = (float)i / x_ticks;
        float px = p0.x + t * dispSize.x;
        bg->AddLine(ImVec2(px, p0.y), ImVec2(px, p1.y),
                    IM_COL32(40, 40, 60, 100));
      }
      for (int i = 0; i <= y_ticks; ++i) {
        float t = (float)i / y_ticks;
        float py = p0.y + t * dispSize.y;
        bg->AddLine(ImVec2(p0.x, py), ImVec2(p1.x, py),
                    IM_COL32(40, 40, 60, 100));
      }
    }

    // Coordinate axes
    float x0 = p0.x + dispSize.x * (-x_min / x_range);
    float y0 = p0.y + dispSize.y * (y_max / y_range);
    
    // Only draw axes if they're visible
    if (x_min <= 0 && x_max >= 0) {
      bg->AddLine(ImVec2(x0, p0.y), ImVec2(x0, p1.y),
                  IM_COL32(200, 200, 200, 180), 1.5f);
    }
    if (y_min <= 0 && y_max >= 0) {
      bg->AddLine(ImVec2(p0.x, y0), ImVec2(p1.x, y0),
                  IM_COL32(200, 200, 200, 180), 1.5f);
    }

    // Axis labels
    if (show_axis_labels) {
      for (int i = 0; i <= x_ticks; ++i) {
        float t = (float)i / x_ticks;
        float px = p0.x + t * dispSize.x;
        float data_x = x_min + t * x_range;
        
        if (y_min <= 0 && y_max >= 0) {
          bg->AddLine(ImVec2(px, y0 - 4), ImVec2(px, y0 + 4),
                      IM_COL32(200, 200, 200, 150));
        }
        
        char buf[32];
        snprintf(buf, 32, "%.1f", data_x);
        float label_y = (y_min <= 0 && y_max >= 0) ? y0 + 5 : p1.y - 15;
        bg->AddText(ImVec2(px + 2, label_y), IM_COL32(180, 180, 180, 200), buf);
      }
      
      for (int i = 0; i <= y_ticks; ++i) {
        float t = (float)i / y_ticks;
        float py = p0.y + t * dispSize.y;
        float data_y = y_max - t * y_range;
        
        if (x_min <= 0 && x_max >= 0) {
          bg->AddLine(ImVec2(x0 - 4, py), ImVec2(x0 + 4, py),
                      IM_COL32(200, 200, 200, 150));
        }
        
        char buf[32];
        snprintf(buf, 32, "%.1f", data_y);
        float label_x = (x_min <= 0 && x_max >= 0) ? x0 + 6 : p0.x + 5;
        bg->AddText(ImVec2(label_x, py - 6), IM_COL32(180, 180, 180, 200), buf);
      }
    }

    // Lambda to convert data coordinates to screen coordinates
    auto dataToScreen = [&](float x, float y) -> ImVec2 {
      float px = p0.x + (x - x_min) / x_range * dispSize.x;
      float py = p0.y + (y_max - y) / y_range * dispSize.y;
      return ImVec2(px, py);
    };

    // Lambda to convert screen coordinates to data coordinates
    auto screenToData = [&](ImVec2 screen) -> std::pair<float, float> {
      float x = x_min + (screen.x - p0.x) / dispSize.x * x_range;
      float y = y_max - (screen.y - p0.y) / dispSize.y * y_range;
      return {x, y};
    };

    // Store curve points for hover detection
    struct CurveData {
      std::vector<ImVec2> points;
      std::vector<float> x_values;
      int func_id;
      ImU32 color;
    };
    std::vector<CurveData> all_curves;

    // Draw function curves
    auto drawFunction = [&](int func_id, ImU32 color, bool is_derivative = false) {
      CurveData curve;
      curve.func_id = func_id;
      curve.color = color;
      
      ImVec2 prev;
      bool firstPt = true;
      
      for (int i = 0; i <= samples; ++i) {
        float t = (float)i / samples;
        float x = x_min + t * x_range;
        float y;
        
        if (is_derivative) {
          y = EvaluateDerivative(func_id, x, amplitude, frequency);
        } else {
          y = EvaluateFunction(func_id, x, amplitude, frequency);
        }
        
        // Handle NaN/Inf
        if (std::isnan(y) || std::isinf(y)) {
          firstPt = true;
          continue;
        }
        
        ImVec2 pt = dataToScreen(x, y);
        
        if (!is_derivative) {
          curve.points.push_back(pt);
          curve.x_values.push_back(x);
        }
        
        // Only draw if point is somewhat visible
        if (pt.y >= p0.y - 100 && pt.y <= p1.y + 100) {
          if (!firstPt) {
            bg->AddLine(prev, pt, color, line_thickness);
          }
          prev = pt;
          firstPt = false;
        } else {
          firstPt = true;
        }
      }
      
      if (!is_derivative) {
        all_curves.push_back(curve);
      }
    };

    // Draw main function or comparison functions
    if (comparison_mode && !comparison_functions.empty()) {
      int color_idx = 0;
      for (int func_id : comparison_functions) {
        ImU32 color = function_colors[color_idx % num_colors];
        drawFunction(func_id, color);
        if (show_derivative) {
          ImU32 deriv_color = (color & 0x00FFFFFF) | 0x80000000; // Semi-transparent
          drawFunction(func_id, deriv_color, true);
        }
        color_idx++;
      }
    } else {
      drawFunction(func_type, function_colors[0]);
      if (show_derivative) {
        drawFunction(func_type, IM_COL32(120, 200, 255, 180), true);
      }
    }

    // Draw function sample points
    if (show_function_points && !all_curves.empty()) {
      for (const auto& curve : all_curves) {
        for (size_t i = 0; i < curve.points.size(); i += 10) {
          bg->AddCircleFilled(curve.points[i], 2.0f, curve.color);
        }
      }
    }

    // Mouse interaction
    ImVec2 mouse_pos = io.MousePos;
    bool mouse_in_rect = (mouse_pos.x >= p0.x && mouse_pos.x <= p1.x &&
                          mouse_pos.y >= p0.y && mouse_pos.y <= p1.y);
    bool control_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow);

    // Mouse wheel zoom
    if (mouse_in_rect && !control_hovered && io.MouseWheel != 0) {
      auto [data_x, data_y] = screenToData(mouse_pos);
      float zoom_factor = io.MouseWheel > 0 ? 0.9f : 1.1f;
      ZoomView(zoom_factor, data_x, data_y);
    }

    // Mouse drag for panning
    if (mouse_in_rect && !control_hovered) {
      if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        is_dragging = true;
        drag_start_pos = mouse_pos;
        drag_start_x_min = x_min;
        drag_start_x_max = x_max;
        drag_start_y_min = y_min;
        drag_start_y_max = y_max;
      }
    }
    
    if (is_dragging) {
      if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        float dx = (mouse_pos.x - drag_start_pos.x) / dispSize.x * x_range;
        float dy = (mouse_pos.y - drag_start_pos.y) / dispSize.y * y_range;
        x_min = drag_start_x_min - dx;
        x_max = drag_start_x_max - dx;
        y_min = drag_start_y_min + dy;
        y_max = drag_start_y_max + dy;
      } else {
        is_dragging = false;
      }
    }

    // Hover detection and crosshair
    if (mouse_in_rect && !control_hovered && !is_dragging) {
      auto [mouse_data_x, mouse_data_y] = screenToData(mouse_pos);

      // Draw crosshair
      bg->AddLine(ImVec2(p0.x, mouse_pos.y), ImVec2(p1.x, mouse_pos.y),
                  IM_COL32(100, 100, 150, 120), 1.0f);
      bg->AddLine(ImVec2(mouse_pos.x, p0.y), ImVec2(mouse_pos.x, p1.y),
                  IM_COL32(100, 100, 150, 120), 1.0f);

      // Find closest point on any curve
      bool hoveredOnCurve = false;
      float hoverX = 0.0f, hoverY = 0.0f;
      int hoverFuncId = -1;
      ImU32 hoverColor = 0;
      const float hoverRadius = 15.0f;

      for (const auto& curve : all_curves) {
        float minDist = FLT_MAX;
        int closestIdx = -1;
        
        for (size_t i = 0; i < curve.points.size(); i += 2) {
          float dx = mouse_pos.x - curve.points[i].x;
          float dy = mouse_pos.y - curve.points[i].y;
          float dist = dx * dx + dy * dy;
          if (dist < minDist) {
            minDist = dist;
            closestIdx = i;
          }
        }
        
        if (closestIdx >= 0 && minDist < hoverRadius * hoverRadius) {
          hoveredOnCurve = true;
          hoverX = curve.x_values[closestIdx];
          hoverY = EvaluateFunction(curve.func_id, hoverX, amplitude, frequency);
          hoverFuncId = curve.func_id;
          hoverColor = curve.color;
          break;
        }
      }

      // Build tooltip text
      std::stringstream ss;
      ss.precision(4);
      ss << std::fixed;
      
      if (hoveredOnCurve) {
        std::string func_name = FunctionsManager::Instance().GetFunctionName(hoverFuncId);
        ss << func_name << "\n";
        ss << "x = " << hoverX << "\n";
        ss << "f(x) = " << hoverY << "\n";
        
        float deriv = EvaluateDerivative(hoverFuncId, hoverX, amplitude, frequency);
        ss << "f'(x) = " << deriv;
        
        // Highlight point on curve
        ImVec2 highlight_pt = dataToScreen(hoverX, hoverY);
        bg->AddCircleFilled(highlight_pt, 6.0f, hoverColor);
        bg->AddCircle(highlight_pt, 8.0f, IM_COL32(255, 255, 255, 200), 0, 2.0f);
        
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
      } else {
        ss << "x = " << mouse_data_x << "\n";
        ss << "y = " << mouse_data_y;
      }

      // Draw tooltip
      ImVec2 tooltip_pos = ImVec2(mouse_pos.x + 15, mouse_pos.y - 60);
      if (tooltip_pos.x + 150 > p1.x) tooltip_pos.x = mouse_pos.x - 165;
      if (tooltip_pos.y < p0.y) tooltip_pos.y = mouse_pos.y + 15;
      
      std::string tooltip_text = ss.str();
      ImVec2 text_size = ImGui::CalcTextSize(tooltip_text.c_str());
      
      bg->AddRectFilled(
          ImVec2(tooltip_pos.x - 5, tooltip_pos.y - 5),
          ImVec2(tooltip_pos.x + text_size.x + 10, tooltip_pos.y + text_size.y + 10),
          IM_COL32(30, 30, 40, 220), 4.0f);
      bg->AddRect(
          ImVec2(tooltip_pos.x - 5, tooltip_pos.y - 5),
          ImVec2(tooltip_pos.x + text_size.x + 10, tooltip_pos.y + text_size.y + 10),
          IM_COL32(100, 100, 150, 200), 4.0f);
      bg->AddText(tooltip_pos, IM_COL32(255, 255, 255, 255), tooltip_text.c_str());
    }

    // ========================================================================
    // Control Panel Window
    // ========================================================================
    ImGui::SetNextWindowBgAlpha(0.92f);
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320, 600), ImGuiCond_FirstUseEver);
    
    ImGui::Begin("Activation Functions Control", nullptr, ImGuiWindowFlags_None);
    
    // Title
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Neural Network Activation Functions");
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "400 Functions Survey Visualization");
    ImGui::Separator();

    // View Controls
    if (ImGui::CollapsingHeader("View Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (ImGui::Button("Reset View")) {
        ResetView();
      }
      ImGui::SameLine();
      if (ImGui::Button("Zoom In")) {
        ZoomView(0.8f, (x_min + x_max) / 2, (y_min + y_max) / 2);
      }
      ImGui::SameLine();
      if (ImGui::Button("Zoom Out")) {
        ZoomView(1.25f, (x_min + x_max) / 2, (y_min + y_max) / 2);
      }
      
      ImGui::DragFloatRange2("X Range", &x_min, &x_max, 0.1f, -100.0f, 100.0f);
      ImGui::DragFloatRange2("Y Range", &y_min, &y_max, 0.1f, -100.0f, 100.0f);
      
      ImGui::SliderInt("X Ticks", &x_ticks, 5, 50);
      ImGui::SliderInt("Y Ticks", &y_ticks, 5, 50);
      ImGui::SliderInt("Samples", &samples, 100, 2000);
      ImGui::SliderFloat("Line Width", &line_thickness, 1.0f, 5.0f);
    }

    // Display Options
    if (ImGui::CollapsingHeader("Display Options", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Checkbox("Show Grid", &show_grid);
      ImGui::Checkbox("Show Axis Labels", &show_axis_labels);
      ImGui::Checkbox("Show Derivative", &show_derivative);
      ImGui::Checkbox("Show Sample Points", &show_function_points);
      ImGui::Checkbox("Comparison Mode", &comparison_mode);
    }

    // Function Parameters
    if (ImGui::CollapsingHeader("Function Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::SliderFloat("Amplitude", &amplitude, 0.1f, 5.0f);
      ImGui::SliderFloat("Frequency", &frequency, 0.1f, 5.0f);
      
      if (ImGui::Button("Reset Parameters")) {
        amplitude = 1.0f;
        frequency = 1.0f;
      }
    }

    // Function Selection
    if (ImGui::CollapsingHeader("Function Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
      // Category filter
      const char* categories[] = {
        "All Functions", "Sigmoid Family", "Tanh Family", "ReLU Family",
        "Exponential", "Gaussian/Radial", "Adaptive", "Piecewise Linear",
        "Smooth Approx", "Special", "Modern", "Attention/Transformer",
        "Polynomial", "Probabilistic"
      };
      ImGui::Combo("Category", &current_category, categories, IM_ARRAYSIZE(categories));
      
      // Search box
      static char search_buf[64] = "";
      ImGui::InputText("Search", search_buf, IM_ARRAYSIZE(search_buf));
      
      // Function list
      ImGui::BeginChild("Function List", ImVec2(0, 200), true);
      
      FunctionCategory filter_cat = static_cast<FunctionCategory>(current_category);
      
      for (size_t i = 0; i < FunctionsManager::Instance().GetNumberOfFunctions(); ++i) {
        std::string name = FunctionsManager::Instance().GetFunctionName(i);
        FunctionCategory cat = FunctionsManager::Instance().GetCategory(name);
        
        // Filter by category
        if (filter_cat != FunctionCategory::ALL && cat != filter_cat) {
          continue;
        }
        
        // Filter by search
        if (strlen(search_buf) > 0) {
          std::string lower_name = ToLower(name);
          std::string lower_search = ToLower(search_buf);
          if (lower_name.find(lower_search) == std::string::npos) {
            continue;
          }
        }
        
        bool is_selected = (func_type == static_cast<int>(i));
        bool is_compared = comparison_functions.count(i) > 0;
        
        // Color coding for compared functions
        if (is_compared) {
          ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 1.0f, 0.5f, 1.0f));
        }
        
        std::string display_name = name;
        if (comparison_mode && is_compared) {
          display_name = "[*] " + name;
        }
        
        if (ImGui::Selectable(display_name.c_str(), is_selected)) {
          if (comparison_mode) {
            if (is_compared) {
              comparison_functions.erase(i);
            } else {
              comparison_functions.insert(i);
            }
          } else {
            func_type = i;
          }
        }
        
        // Tooltip with description
        if (ImGui::IsItemHovered()) {
          std::string desc = FunctionsManager::Instance().GetDescription(name);
          if (!desc.empty()) {
            ImGui::SetTooltip("%s", desc.c_str());
          }
        }
        
        if (is_compared) {
          ImGui::PopStyleColor();
        }
      }
      ImGui::EndChild();
      
      if (comparison_mode) {
        if (ImGui::Button("Clear Comparison")) {
          comparison_functions.clear();
        }
        ImGui::SameLine();
        ImGui::Text("Selected: %zu", comparison_functions.size());
      }
    }

    // Current Function Info
    if (ImGui::CollapsingHeader("Current Function Info", ImGuiTreeNodeFlags_DefaultOpen)) {
      std::string current_name = FunctionsManager::Instance().GetFunctionName(func_type);
      std::string current_desc = FunctionsManager::Instance().GetDescription(current_name);
      
      ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "Name: %s", current_name.c_str());
      if (!current_desc.empty()) {
        ImGui::TextWrapped("Formula: %s", current_desc.c_str());
      }
      
      // Show some key values
      ImGui::Separator();
      ImGui::Text("Key Values:");
      float test_values[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
      for (float x : test_values) {
        float y = EvaluateFunction(func_type, x, amplitude, frequency);
        ImGui::Text("  f(%.1f) = %.4f", x, y);
      }
    }

    // Legend for comparison mode
    if (comparison_mode && !comparison_functions.empty()) {
      if (ImGui::CollapsingHeader("Legend", ImGuiTreeNodeFlags_DefaultOpen)) {
        int color_idx = 0;
        for (int func_id : comparison_functions) {
          ImU32 color = function_colors[color_idx % num_colors];
          ImVec4 col_vec = ImGui::ColorConvertU32ToFloat4(color);
          
          std::string name = FunctionsManager::Instance().GetFunctionName(func_id);
          ImGui::ColorButton(("##color" + std::to_string(func_id)).c_str(), col_vec, 
                            ImGuiColorEditFlags_NoTooltip, ImVec2(20, 20));
          ImGui::SameLine();
          ImGui::Text("%s", name.c_str());
          
          color_idx++;
        }
      }
    }

    // Instructions
    if (ImGui::CollapsingHeader("Instructions")) {
      ImGui::BulletText("Mouse wheel: Zoom in/out");
      ImGui::BulletText("Right-click drag: Pan view");
      ImGui::BulletText("Hover on curve: Show exact values");
      ImGui::BulletText("Enable Comparison Mode to select");
      ImGui::BulletText("multiple functions");
    }

    // Statistics
    ImGui::Separator();
    ImGui::Text("Total Functions: %zu", FunctionsManager::Instance().GetNumberOfFunctions());
    ImGui::Text("FPS: %.1f", io.Framerate);
    
    ImGui::End();

    // Render
    ImGui::Render();
    SDL_GL_MakeCurrent(window_, gl_context_);
    glViewport((int)vp->Pos.x, (int)vp->Pos.y, (int)vp->Size.x,
               (int)vp->Size.y);
    glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(window_);
  }
  return 0;
}

MainWindow::~MainWindow() {
  DEBUG(MainWindow) << "destroyed.";
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();
  SDL_GL_DeleteContext(gl_context_);
  SDL_DestroyWindow(window_);
  SDL_Quit();
}

}  // namespace Autoalg
