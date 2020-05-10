import os

if not os.environ.get('IS_MAIN', '0') == '1':
    import matplotlib
    matplotlib.use('template')  # suppress figures for spyder unit testing
