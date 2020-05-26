import os


def pytest_configure(config):
    import traceback
    t = traceback.extract_stack()
    if 'pytestworker.py' in t[0][0]:
        import matplotlib
        matplotlib.use('template')  # suppress plots when Spyder unit-testing


def pytest_generate_tests(metafunc):
    os.environ["TF_KERAS"] = os.environ.get("TF_KERAS", '1')
