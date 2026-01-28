# Engineering Degree - Hand Gesture Recognition Game

This project implements a hand gesture recognition system and a game controlled by these gestures. It was developed as part of an Engineering Degree.

## Features
- Real-time hand gesture detection using MediaPipe.
- Support for multiple classification models (MLP, CNN, k-NN).
- Custom Neural Network implementation from scratch.
- Game controlled by hand gestures.
- Data collection and training scripts.

## Project Structure
- `Game/`: Contains the game source code, assets (sounds, sprites, fonts), and levels.
- `gestdetect.py`, `hand.py`: Core logic for hand landmark detection using MediaPipe.
- `create_data.py`, `get_data.py`: Scripts for collecting gesture data.
- `*_train_test.py`: Training and testing scripts for various Machine Learning models.
- `main.py`: Main entry point for the gesture detection demo.
- `Game/game.py`: Main entry point for the gesture-controlled game.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- MediaPipe
- PyTorch (for torch-based models)
- Pygame/SDL2 (for the game component)
- NumPy

## How to Run

### Real-time Gesture Detection Demo
To run the live camera feed with gesture recognition:
```bash
python main.py
```

### Running the Game
To start the gesture-controlled game:
1. Navigate to the `Game` directory.
2. Run the game script:
```bash
python game.py
```

### Collecting Data & Training
1. Run `get_data.py` to capture landmarks for different gestures.
2. Run any of the training scripts (e.g., `mlp_train_test.py`) to train the model on the collected data.
