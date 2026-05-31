/*
  Wevel 1.0 - HalfKP feature set for NNUE evaluation.

  HalfKP maps (king_sq, piece_type, piece_sq, perspective) to a feature index.
  Features are computed independently for each side (perspective), using the
  king square of that side as the "anchor".

  Feature layout per perspective:
    - 64 possible king squares (oriented: black's squares flipped vertically)
    - 641 possible (piece_type, piece_sq) combinations:
        friendly pieces: (pt-PAWN)*64 + oriented_sq  (offset 0..319)
        enemy pieces:    (pt-PAWN)*64 + oriented_sq + 320  (offset 320..639)
    - Total: 64 * 641 = 41024 features per side

  The feature index is:
    oriented_ksq * PS_NB + piece_sq_offset
*/

#ifndef HALF_KP_H_INCLUDED
#define HALF_KP_H_INCLUDED

#include "../../types.h"
#include "../nnue_common.h"

class Position;

namespace Eval::NNUE::Features {

// Returns the square oriented from the given perspective.
// Black's perspective flips the board vertically (rank 1 <-> rank 8).
inline Square orient(Color perspective, Square sq) {
    return perspective == WHITE ? sq : Square(sq ^ 56);
}

// Returns the piece-square feature offset for a piece on a given square
// from a given perspective.
// pt:          piece type (PAWN..QUEEN, not KING)
// piece_color: color of the piece
// perspective: the side whose accumulator we're updating
// sq:          piece square (NOT yet oriented — orient first)
inline int piece_sq_offset(PieceType pt, Color piece_color, Color perspective) {
    // Friendly = same color as perspective; enemy = opposite
    int relative_color = (piece_color != perspective) ? 1 : 0;
    return relative_color * 5 * 64 + (pt - PAWN) * 64;
}

// Computes the full feature index for a non-king piece from a given perspective.
// Returns -1 if the piece is a king (kings are excluded from HalfKP features).
inline int make_index(Color perspective, Square ksq, Square piece_sq,
                      PieceType pt, Color piece_color) {
    if (pt == KING) return -1;
    Square oriented_ksq  = orient(perspective, ksq);
    Square oriented_psq  = orient(perspective, piece_sq);
    int    psq_offset    = piece_sq_offset(pt, piece_color, perspective);
    return oriented_ksq * PS_NB + psq_offset + oriented_psq;
}

// Collect all active feature indices for a given perspective into `indices`.
// Returns number of features added.
int collect_features(Color perspective, const Position& pos, int* indices);

} // namespace Eval::NNUE::Features

#endif // HALF_KP_H_INCLUDED
