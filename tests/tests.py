import os
import pytest

os.chdir(os.getcwd())

pytest.main(["-s"])
