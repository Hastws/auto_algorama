#ifndef INCLUDE_AUTOALG_FUNCTIONS_H
#define INCLUDE_AUTOALG_FUNCTIONS_H

#include "activation_functions.h"
#include "common/types.h"

namespace Autoalg {
using Function = std::function<Real(Real)>;

inline std::string ToLower(const std::string &s) {
  std::string result = s;
  std::transform(result.begin(), result.end(), result.begin(), ::tolower);
  return result;
}

// Function category enumeration
enum class FunctionCategory {
  ALL,
  SIGMOID_FAMILY,
  TANH_FAMILY,
  RELU_FAMILY,
  EXPONENTIAL,
  GAUSSIAN,
  ADAPTIVE,
  PIECEWISE,
  SMOOTH,
  SPECIAL,
  MODERN,
  ATTENTION,
  POLYNOMIAL,
  PROBABILISTIC
};

inline const char* CategoryToString(FunctionCategory cat) {
  switch (cat) {
    case FunctionCategory::ALL: return "All Functions";
    case FunctionCategory::SIGMOID_FAMILY: return "Sigmoid Family";
    case FunctionCategory::TANH_FAMILY: return "Tanh Family";
    case FunctionCategory::RELU_FAMILY: return "ReLU Family";
    case FunctionCategory::EXPONENTIAL: return "Exponential";
    case FunctionCategory::GAUSSIAN: return "Gaussian/Radial";
    case FunctionCategory::ADAPTIVE: return "Adaptive";
    case FunctionCategory::PIECEWISE: return "Piecewise Linear";
    case FunctionCategory::SMOOTH: return "Smooth Approx";
    case FunctionCategory::SPECIAL: return "Special";
    case FunctionCategory::MODERN: return "Modern";
    case FunctionCategory::ATTENTION: return "Attention/Transformer";
    case FunctionCategory::POLYNOMIAL: return "Polynomial";
    case FunctionCategory::PROBABILISTIC: return "Probabilistic";
    default: return "Unknown";
  }
}

class FunctionsManager {
 public:
  struct FunctionInfo {
    Function func;
    FunctionCategory category;
    std::string description;
  };

  static FunctionsManager &Instance() {
    static FunctionsManager instance;
    return instance;
  }

  void RegisterFunction(const std::string &name, const Function &func,
                        FunctionCategory category = FunctionCategory::ALL,
                        const std::string &description = "") {
    const std::string key = ToLower(name);
    if (function_map_.count(key)) return;

    const int id = static_cast<int>(id_to_name_.size());
    function_map_[key] = {func, category, description};
    name_to_id_[key] = id;
    id_to_name_.push_back(key);

    cache_valid_ = false;
  }

  Real Call(const std::string &name, const Real &input) const {
    const auto it = function_map_.find(ToLower(name));
    if (it != function_map_.end()) {
      return it->second.func(input);
    }
    throw std::runtime_error("Function not found: " + name);
  }

  Real CallById(int id, const Real &input) const {
    if (id >= 0 && id < static_cast<int>(id_to_name_.size())) {
      return Call(id_to_name_[id], input);
    }
    throw std::runtime_error("Invalid function ID: " + std::to_string(id));
  }

  Function GetFunction(const std::string &name) const {
    const auto it = function_map_.find(ToLower(name));
    if (it != function_map_.end()) {
      return it->second.func;
    }
    return nullptr;
  }

  Function GetFunctionById(int id) const {
    if (id >= 0 && id < static_cast<int>(id_to_name_.size())) {
      return GetFunction(id_to_name_[id]);
    }
    return nullptr;
  }

  FunctionCategory GetCategory(const std::string &name) const {
    const auto it = function_map_.find(ToLower(name));
    if (it != function_map_.end()) {
      return it->second.category;
    }
    return FunctionCategory::ALL;
  }

  std::string GetDescription(const std::string &name) const {
    const auto it = function_map_.find(ToLower(name));
    if (it != function_map_.end()) {
      return it->second.description;
    }
    return "";
  }

  std::vector<std::string> GetAllFunctionNames() const {
    std::vector<std::string> names;
    for (const auto &pair : function_map_) {
      names.push_back(pair.first);
    }
    return names;
  }

  std::vector<int> GetFunctionsByCategory(FunctionCategory category) const {
    std::vector<int> ids;
    for (size_t i = 0; i < id_to_name_.size(); ++i) {
      if (category == FunctionCategory::ALL ||
          GetCategory(id_to_name_[i]) == category) {
        ids.push_back(static_cast<int>(i));
      }
    }
    return ids;
  }

  std::size_t GetNumberOfFunctions() const { return function_map_.size(); }

  int GetFunctionId(const std::string &name) const {
    const auto it = name_to_id_.find(ToLower(name));
    if (it != name_to_id_.end()) return it->second;
    throw std::runtime_error("Function ID not found for: " + name);
  }

  std::string GetFunctionName(const int id) const {
    if (id >= 0 && id < static_cast<int>(id_to_name_.size()))
      return id_to_name_[id];
    throw std::runtime_error("Invalid function ID: " + std::to_string(id));
  }

  const char *GetZeroSeparatedFunctionNames() {
    if (!cache_valid_) {
      function_names_zero_separated_.clear();
      for (const auto &name : id_to_name_) {
        function_names_zero_separated_.append(name);
        function_names_zero_separated_.push_back('\0');
      }
      cache_valid_ = true;
    }
    return function_names_zero_separated_.c_str();
  }

 private:
  bool cache_valid_ = false;
  std::string function_names_zero_separated_;
  std::unordered_map<std::string, FunctionInfo> function_map_;
  std::unordered_map<std::string, int> name_to_id_;
  std::vector<std::string> id_to_name_;
  FunctionsManager() = default;
};
}  // namespace Autoalg

#define REGISTER_FUNCTION(name, func)                                      \
  static bool Registered_##name = []() {                                   \
    ::Autoalg::FunctionsManager::Instance().RegisterFunction(#name, func); \
    return true;                                                           \
  }()

#define REGISTER_FUNCTION_WITH_CATEGORY(name, func, category, desc)           \
  static bool Registered_##name = []() {                                      \
    ::Autoalg::FunctionsManager::Instance().RegisterFunction(#name, func,     \
                                                             category, desc); \
    return true;                                                              \
  }()

#endif  // INCLUDE_AUTOALG_FUNCTIONS_H
