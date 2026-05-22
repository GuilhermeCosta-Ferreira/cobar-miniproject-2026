# Miniproject for BIOENG-456: Controlling behavior in animals and robots

Welcome to the Miniproject for BIOENG-456!

## Setup
Run `uv sync` to make sure you have all the dependencies installed
```bash
uv sync
```

## Usage

To explore the levels interactively with the keyboard, run the `run_interactive.py` script. Then you can use the WASD keys to control the fly (Q to stop and ESC to exit).

```bash
uv run miniproject/run_interactive.py --level <level> --seed <seed>
```

Replace `<level>` with the desired level number (0 to 4 for the 5 levels) and `<seed>` with the random seed for reproducibility.

If you want to see the fly's vision as well, add the `--render-fly-vision` argument.

> If you have an issue with the rendering (black screen - mainly seen on linux), you can try to add the argument `--dont-use-pygame-rendering` to fallback to opencv rendering instead of pygame. Note: first you will have to run `uv pip install pynput` to get the pynput library.

The `run_simulation.ipynb` notebook contains code that will be used to evaluate the controller.