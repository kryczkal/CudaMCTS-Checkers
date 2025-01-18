#ifndef MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_TPP_
#define MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_TPP_

#include "board_helpers.cuh"

namespace checkers::mcts::gpu {
    static constexpr u8 kIsPieceOnBoardFlagIndex = 0;

    template<Turn turn>
    void GenerateMoves(
            // Board States
            const board_t *d_whites,
            const board_t *d_blacks,
            const board_t *d_kings,
            // Moves
            move_t *d_moves,
            u8* d_move_counts,
            u8* d_move_flags
    ) {
        // TODO: Use shmem
        board_t board_index  = 0;
        u8  figure_index = 0;

        u16 flags = 0;
        board_t pieces = turn == Turn::kWhite ? d_whites[board_index] : d_blacks[board_index];

        /////////////////////////////// Regular Pieces ///////////////////////////////
        board_t no_kings = pieces & ~d_kings[board_index];
        flags |= ( (no_kings >> figure_index) & 1 ) << kIsPieceOnBoardFlagIndex;

        board_index_t upper_left_index;
        board_index_t upper_right_index;

        board_index_t lower_left_index;
        board_index_t lower_right_index;
    }
}

#endif // MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_TPP_