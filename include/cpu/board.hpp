#ifndef MCTS_CHECKERS_INCLUDE_CPU_CPU_BOARD_HPP_
#define MCTS_CHECKERS_INCLUDE_CPU_CPU_BOARD_HPP_

#include "common/checkers_defines.hpp"

namespace checkers::cpu
{
struct Board {
    public:
    board_t white;
    board_t black;
    board_t kings;

    Board() : white(0), black(0), kings(0) {}

    bool operator==(const Board& other) const
    {
        return (white == other.white) && (black == other.black) && (kings == other.kings);
    }

    bool operator!=(const Board& other) const { return !(*this == other); }

    /**
     * @brief Helper to set a piece.
     *        'W' -> white, 'B' -> black, 'K' -> king flag.
     */
    void setPieceAt(board_index_t idx, char pieceType)
    {
        switch (pieceType) {
            case 'W':
                white |= (static_cast<board_t>(1) << idx);
                break;
            case 'B':
                black |= (static_cast<board_t>(1) << idx);
                break;
            case 'K':
                kings |= (static_cast<board_t>(1) << idx);
                break;
            default:
                break;
        }
    }
};

}  // namespace checkers::cpu

#endif  // MCTS_CHECKERS_INCLUDE_CPU_CPU_BOARD_HPP_
