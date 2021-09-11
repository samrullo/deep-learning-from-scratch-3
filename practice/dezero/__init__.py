is_simple_core = False

if is_simple_core:
    from practice.dezero.core_simple import Variable
    from practice.dezero.core_simple import Function
    from practice.dezero.core_simple import using_config
    from practice.dezero.core_simple import no_grad
    from practice.dezero.core_simple import as_array
    from practice.dezero.core_simple import as_variable
    from practice.dezero.core_simple import setup_variable
    from practice.dezero.core_simple import add
    from practice.dezero.core_simple import mul
    from practice.dezero.core_simple import sub
    from practice.dezero.core_simple import rsub
    from practice.dezero.core_simple import div
    from practice.dezero.core_simple import pow
else:
    from practice.dezero.core import Variable
    from practice.dezero.core import Parameter
    from practice.dezero.core import Function
    from practice.dezero.core import using_config
    from practice.dezero.core import no_grad
    from practice.dezero.core import as_array
    from practice.dezero.core import as_variable
    from practice.dezero.core import setup_variable
    from practice.dezero.core import add
    from practice.dezero.core import mul
    from practice.dezero.core import sub
    from practice.dezero.core import rsub
    from practice.dezero.core import div
    from practice.dezero.core import pow

    from practice.dezero.layers import Model
    from practice.dezero.layers import Layer

    import practice.dezero.functions
    import practice.dezero.utils

setup_variable()
version = "__1.0__"
