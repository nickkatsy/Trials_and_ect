def grim_trigger(opponent_moves):
    if 'D' in opponent_moves:
        return 'D'
    else:
        return 'C' 
def tit_for_tat(opponent_moves):
    if not opponent_moves:
        return 'D'
    else:
        return opponent_moves[-1]  


def play_game(player1_strategy, player2_strategy, num_rounds):
    player1_moves = []
    player2_moves = []
    for _ in range(num_rounds):
        p1_move = player1_strategy(player2_moves)
        p2_move = player2_strategy(player1_moves)
        player1_moves.append(p1_move)
        player2_moves.append(p2_move)
    return player1_moves, player2_moves


player1_moves, player2_moves = play_game(tit_for_tat, grim_trigger, 48)


print("Player 1 :", player1_moves)
print("Player 2 :", player2_moves)