#include <iostream>

const int payoff_prisoner_dilemma[2][2] = {{-1, -3}, {0, -2}};
const double discount_rate = 0.20;

void print_payoff_matrix() {
    std::cout << "Payoff Matrix for Prisoner's Dilemma:" << std::endl;
    std::cout << "        | Cooperate | Defect" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    std::cout << "Cooperate|     " << payoff_prisoner_dilemma[0][0] << "     |     " << payoff_prisoner_dilemma[0][1] << std::endl;
    std::cout << "Defect   |     " << payoff_prisoner_dilemma[1][0] << "     |     " << payoff_prisoner_dilemma[1][1] << std::endl;
}

void prisoner_dilemma(bool player1_cooperate, bool player2_cooperate) {
    int player1_payoff = (player1_cooperate && player2_cooperate) ? payoff_prisoner_dilemma[0][0] :
                         (player1_cooperate && !player2_cooperate) ? payoff_prisoner_dilemma[0][1] :
                         (!player1_cooperate && player2_cooperate) ? payoff_prisoner_dilemma[1][0] :
                                                                      payoff_prisoner_dilemma[1][1];
    int player2_payoff = (player1_cooperate && player2_cooperate) ? payoff_prisoner_dilemma[0][0] :
                         (player1_cooperate && !player2_cooperate) ? payoff_prisoner_dilemma[1][0] :
                         (!player1_cooperate && player2_cooperate) ? payoff_prisoner_dilemma[0][1] :
                                                                      payoff_prisoner_dilemma[1][1];


    player1_payoff -= (int)(discount_rate * player1_payoff);
    player2_payoff -= (int)(discount_rate * player2_payoff);

    std::cout << "Player 1's Payoff: " << player1_payoff << std::endl;
    std::cout << "Player 2's Payoff: " << player2_payoff << std::endl;
}

int main() {
    print_payoff_matrix();
    
    std::cout << std::endl << "round 1:" << std::endl;
    std::cout << "Both players cooperate:" << std::endl;
    prisoner_dilemma(true, true);

    std::cout << std::endl << "Player 1 cooperates, Player 2 defects:" << std::endl;
    prisoner_dilemma(true, false);

    std::cout << std::endl << "Player 1 defects, Player 2 cooperates:" << std::endl;
    prisoner_dilemma(false, true);

    std::cout << std::endl << "both defect"<< std::endl;
    prisoner_dilemma(false, false);

    return 0;
}