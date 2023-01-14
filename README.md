# GenerativeGI

Genetic algorithm / novelty search for discovering generative artwork via evolving a grammar-based solution.

## Usage

I used Python 3.8, though it should work for most current versions of Python.

1. Install required libraries: `python3 -m pip install -r requirements.txt`

2. Run: `python3 deap_main.py [args]`

  * Arguments:

      * `"--gens", type=int, default=100, help="Number of generations to run evolution for."`
      * `"--pop_size", type=int, default=100, help="Population size for evolution."`
      * `"--treatment", type=int, default=0, help="Run Number"`
      * `"--run_num", type=int, default=0, help="Run Number"`
      * `"--output_path", type=str, default="./", help="Output path."`
      * `"--lexicase",action="store_true",help="Whether to do normal or Lexicase selection."`
      * `"--shuffle", action="store_true", help="Shuffle the fitness indicies per selection event."`
      * `"--tourn_size", type=int, default=4, help="What tournament size should we go with?"`
    

## Grammar construction

TBD

## Adding a new technique

To add a technique for drawing you just have to create a self-contained function that accepts a PIL image as input and either updates that same image or returns a new one (up to you, I have some mixed ways to handle that I think).

Techniques are called via the grammar (in `settings.py`).  Basically, you would just add a new rule to the technique rule (with a colon after - I use that for splitting parameters).

Then if you wanted parameters, you can add them below as a new rule.  Currently the only one that accepts parameters is pixel sort, however I'll expand the flow field in the near future to be different.  Note that, if you want to call a different rule, you need to surround it with a pound sign (so for instance, #technique# would be filled with any of the rules within the technique rule) - Tracery in Python isn't too bad, here's a good ref if you want to play with it at all: https://www.brettwitty.net/tracery-in-python.html

So for instance, if I wanted to add circle packing without parameters (just for testing), I'd do something like this:

In `techniques.py`:

Add:

```python
def circlePacking(img):
  # perform circle packing on img directly
```

In `settings.py`:

Update the grammar rules:
```python
rules = {
...
technique = ['stippled:',
             ...
             'circlePacking:'],
}
```

Then in `main.py` within the `evaluate` function:

```python
def evaluate(g): 
   ...
   elif _technique[0] == 'circlePacking':
       circlePacking(g.image)
```

## Techniques

This section outlines the implemented techniques, their parameters, and how to call them via the grammar.

### Circle Packing

**Parameters**

* Palette
* Limit

#### Grammar Specification

TBD

### Dithering

TBD

#### Grammar Specification

TBD

### Drunkard's Walk

TBD

#### Grammar Specification

TBD

### Flow Field

**Parameters**

* Particle size
* Particle lifetime
* Zoom level (multX, multY)
* Palette
* Style
  * Edgy
  * Flowy

#### Grammar Specification

TBD

### Flow Field v2

**Parameters**

* Palette
* Style
  * Edgy
  * Flowy
* Noise Scale
* Resolution

#### Grammar Specification

TBD

### Pixel Sort

**Parameters**

* Angle
* Interval
  * Random
  * Edges
  * Threshold
  * Waves
  * None
* Sorting
  * Lightness
  * Hue
  * Saturation
  * Intensity
  * Minimum
* Lower Threshold
* Upper Threshold

#### Grammar Specification

TBD

### Stipple

TBD

#### Grammar Specification

TBD

### Wolfram Cellular Automata

TBD

#### Grammar Specification

TBD
