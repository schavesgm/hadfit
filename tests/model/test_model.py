# -- Import some third-party modules
import pytest
import numpy as np
import sympy as sy

# -- Import some user-defined modules
from hadfit import Model
from hadfit import generate_cosh_model

@pytest.fixture(scope = "session")
def generate_line_model():
    """ Generate a simple 1-d line model. """
    yield Model('A * x + B', 'x', 'A, B')

@pytest.fixture(scope = "session")
def generate_multi_cosh():
    """ Generate a simple multistate model. """
    yield generate_cosh_model(64, 2)

def generate_regressors():
    """ Generate some regressors to be used in the evaluation. """
    return np.linspace(0, 10), np.arange(64)

class TestModel:

    def test_initialiser(self, generate_line_model, generate_multi_cosh):
        """ Test basic behaviour of the Model class. """

        # -- Assert some conditions on the symbols
        assert generate_line_model.symbols == sy.symbols('x, A, B')
        assert generate_multi_cosh.symbols == sy.symbols('t, s0A, s0M, s1A, s1M')

        # -- Assert some conditions on the expression
        assert str(generate_line_model.expr).replace(' ', '') == 'A*x+B'
        assert str(generate_multi_cosh.expr).replace(' ', '') == 's0A*cosh(s0M*(t-32.0))+s1A*cosh(s1M*(t-32.0))'

    def test_evaluate(self, generate_line_model, generate_multi_cosh):
        """ Test the evaluation of the model. """

        # Regressors to be used in the models
        x, t = generate_regressors()

        # -- Evaluate the models
        assert np.array_equal(
            generate_line_model(x, 10, -2), 
            10 * x - 2
        )
        assert np.array_equal(
            generate_multi_cosh(t, 1.0, 0.05, 0.2, 0.1), 
            np.cosh(0.05 * (t - 64 // 2)) + 0.2 * np.cosh(0.1 * (t - 64 // 2))
        )

    def test_jacobian(self, generate_line_model):
        """ Test the jacobian construction of a simple linear model. """

        # Generate the jacobian of the model
        jac_expr = generate_line_model.show_jacobian()

        # Assert the jacobian is correct
        assert str(jac_expr[0]) == 'x' and str(jac_expr[1]) == '1'

        # Regressors to be used in the models
        x, _ = generate_regressors()

        # Evaluate the jacobian of the linear model
        jac_ev = generate_line_model.eval_jacobian(x = x, A = 10, B = -2)

        # Evaluate the jacobian with respect to A
        assert np.array_equal(jac_ev[0], x)
        assert np.array_equal(jac_ev[1], np.ones(x.size))
