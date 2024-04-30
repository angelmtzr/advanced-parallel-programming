#include <iostream>
#include <vector>
#include <ctime>
#include <string>

#define BOARD_SIZE 9

// Function to check if a number is valid in a row/column/block
bool isValidPlacement(const std::vector<std::vector<int>>& board, int row, int col, int num) {

  // Check row
  for (int i = 0; i < BOARD_SIZE; ++i) {
    if (board[row][i] == num) return false;
  }

  // Check column
  for (int i = 0; i < BOARD_SIZE; ++i) {
    if (board[i][col] == num) return false;
  }

  // Check block (3x3 subgrid)
  int blockRowStart = (row / 3) * 3;
  int blockColStart = (col / 3) * 3;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (board[blockRowStart + i][blockColStart + j] == num) return false;
    }
  }

  return true;
}

// Function to verify if a Sudoku board is valid
bool verifySudokuBoard(const std::vector<std::vector<int>>& board) {

  // Check each cell for valid placement
  for (int i = 0; i < BOARD_SIZE; ++i) {
    for (int j = 0; j < BOARD_SIZE; ++j) {
      if (board[i][j] != 0 && !isValidPlacement(board, i, j, board[i][j])) {
        return false;
      }
    }
  }

  return true;
}

// Recursive function to solve the Sudoku board (backtracking)
bool solveSudoku(std::vector<std::vector<int>>& board, int row = 0, int col = 0) {

  // Base case: reached the end of the board
  if (row == BOARD_SIZE) return true;

  // Skip already filled cells
  if (board[row][col] != 0) {
    return solveSudoku(board, row + (col + 1) / BOARD_SIZE, (col + 1) % BOARD_SIZE);
  }

  // Try all possible numbers (1-9)
  for (int num = 1; num <= BOARD_SIZE; ++num) {
    if (isValidPlacement(board, row, col, num)) {
      board[row][col] = num;
      if (solveSudoku(board, row + (col + 1) / BOARD_SIZE, (col + 1) % BOARD_SIZE)) {
        return true; // Solution found, return true
      }
      board[row][col] = 0; // Backtrack if placement doesn't lead to solution
    }
  }

  // No solution found for this cell
  return false;
}

// Function to print the Sudoku board in a formatted way
void printSudokuBoard(const std::vector<std::vector<int>>& board) {

  const std::string separator = "+-------+-------+-------+";

  std::cout << separator << std::endl;

  for (int i = 0; i < BOARD_SIZE; ++i) {
    std::cout << "| ";
    for (int j = 0; j < BOARD_SIZE; ++j) {
      std::cout << (board[i][j] == 0 ? "." : std::to_string(board[i][j])) << " ";
      if (j % 3 == 2) {
        std::cout << "| ";
      }
    }

    std::cout << std::endl;
    if (i % 3 == 2) {
      std::cout << separator << std::endl;
    }
  }
}

int main() {
    // Sample Sudoku board (modify as needed)
    std::vector<std::vector<int>> board = {
        {6, 0, 0, 0, 2, 3, 0, 7, 9},
        {0, 0, 4, 5, 8, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 5, 3, 0},
        {0, 0, 1, 0, 9, 0, 0, 2, 0},
        {9, 0, 0, 0, 0, 7, 0, 0, 5},
        {4, 0, 0, 0, 0, 5, 8, 0, 0},
        {5, 6, 0, 0, 7, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 1, 6, 8, 0},
        {0, 9, 0, 0, 0, 8, 2, 0, 0}
    };

    // Check if the initial board is valid
    if (!verifySudokuBoard(board)) {
        std::cout << "The initial Sudoku board is not valid.\n" << std::endl;
        return 1;
    }

    clock_t startTime = clock();

    // Attempt to solve the Sudoku board
    if (solveSudoku(board)) {
        clock_t endTime = clock();
        double timeTaken = double(endTime - startTime) / CLOCKS_PER_SEC;

        std::cout << "Sudoku solved in " << timeTaken << " seconds.\n" << std::endl;
        printSudokuBoard(board);
    } else {
        std::cout << "No solution found for the Sudoku board.\n" << std::endl;
    }

    return 0;
}
