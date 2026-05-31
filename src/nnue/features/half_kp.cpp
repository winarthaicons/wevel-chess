/*
  Wevel 1.0 - HalfKP feature collection implementation.
*/

#include "half_kp.h"
#include "../../position.h"

namespace Eval::NNUE::Features {

int collect_features(Color perspective, const Position& pos, int* indices) {
    int count = 0;
    Square ksq = pos.square<KING>(perspective);

    for (PieceType pt : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN }) {
        for (Color c : { WHITE, BLACK }) {
            Bitboard bb = pos.pieces(c, pt);
            while (bb) {
                Square sq = pop_lsb(&bb);
                indices[count++] = make_index(perspective, ksq, sq, pt, c);
            }
        }
    }
    return count;
}

} // namespace Eval::NNUE::Features
