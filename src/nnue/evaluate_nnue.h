/*
  Wevel 1.0 - NNUE evaluation public interface.

  Evaluates a position using the loaded HalfKP-256-32-32-1 neural network.
  Falls back gracefully when no network is loaded (Eval::useNNUE stays false).
*/

#ifndef EVALUATE_NNUE_H_INCLUDED
#define EVALUATE_NNUE_H_INCLUDED

#include <string>
#include "../types.h"

class Position;

namespace Eval {

// Set to true when a valid NNUE network is loaded and verified.
extern bool useNNUE;

namespace NNUE {

// Initialize NNUE (sets up empty weight arrays). Called once at startup.
void init();

// Load a .nnue weight file. Sets Eval::useNNUE on success.
// Accepts the string "<internal>" to use embedded weights (if compiled in).
bool load_eval(const std::string& fname);

// Evaluate the position using the loaded network.
// Returns the score from the side-to-move's perspective, in centipawns.
// Precondition: useNNUE == true and the position is not in check.
Value evaluate(const Position& pos);

// Returns true if a network is loaded and the sizes look correct.
bool verify_net();

} // namespace NNUE
} // namespace Eval

#endif // EVALUATE_NNUE_H_INCLUDED
