from PIL import Image, ImageDraw
import opensimplex
import random
import math
from techniques import *

class GenerativeObject:
  _id = 0 # Global genome identifier

  @classmethod
  def __get_new_id(cls):
      cls._id += 1
      return cls._id

  def __init__(self, dim, grammar, background="black", idx=-1):
    self.id = idx
    self.dim = dim
    self.grammar = grammar
    self.image = Image.new("RGBA", dim, background)
    self.isEvaluated = False
    self.fitness_internal = 0.0

  def setFitness(self, val):
      self.fitness_internal = val
  def getFitness(self):
      return self.fitness_internal

  def get_new_id(self):
    """ Get a new id for the genome. """
    self._id = GenerativeObject.__get_new_id()