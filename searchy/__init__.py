"""Searchy will find it!"""

__version__ = '0.1'

import fire
from searchyclient import SearchyClient

if __name__ == '__main__':
  fire.Fire(SearchyClient)