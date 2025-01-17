#ifndef CUDA_MCTS_CHECKERS_INCLUDE_CHECKERS_ENGINE_TPP_
#define CUDA_MCTS_CHECKERS_INCLUDE_CHECKERS_ENGINE_TPP_

namespace CudaMctsCheckers
{

template <ApplyMoveType type>
bool CheckersEngine::ApplyMove(Board::IndexType from_idx, Board::IndexType to_idx)
{
    // Generate moves for the current player
    auto moves_output      = GetPrecomputedMovesOrGenerate();
    has_precomputed_moves_ = false;

    if (moves_output.no_moves) {
        return false;
    }
    bool capture_possible =
        moves_output.capture_moves_bitmask[MoveGenerationOutput::CaptureFlagIndex];

    if (type == ApplyMoveType::kValidate && !IsMoveValid(from_idx, to_idx, moves_output)) {
        return false;
    }

    PlayMove(from_idx, to_idx, capture_possible);
    UpdateTimeFromNonReversibleMove(to_idx, capture_possible);
    PromoteAll();
    SwitchTurnIfNoChainCapture(capture_possible);

    return true;
}

}  // namespace CudaMctsCheckers

#endif
