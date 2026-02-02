#include <iostream>
#include <ostream>

// Prevent SDL from redefining main on Windows
#define SDL_MAIN_HANDLED
#include <SDL.h>

#include "common/log.h"
#include "windows/main_window.h"

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;
  
  // Initialize SDL (required when using SDL_MAIN_HANDLED)
  SDL_SetMainReady();
  
  Autoalg::MainWindow window(800, 800);
  const int result = window.Process();
  DEBUG(MAIN) << "Processed [" << result << "] results";
  return 0;
}
