from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize([
        "OutputProfile.pyx",
        "ParameterConstraint.pyx",
        "Channel.pyx",
        "Compartment.pyx",
        "Model.pyx",
        "ModelRunner.pyx",
        "model_runner_script.py"
    ])
)