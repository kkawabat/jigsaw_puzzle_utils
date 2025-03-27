from os.path import join, dirname, abspath

import jigsaw_puzzle_utils
from jigsaw_puzzle_utils.edge_finder import EdgeFinder

if __name__ == '__main__':
    DATA_DIR = join(dirname(dirname(abspath(jigsaw_puzzle_utils.__file__))), "tests", 'data')
    _input_path = join(DATA_DIR, 'sample_pieces.jpg')
    _output_path = join(DATA_DIR, 'edge_highlights.jpg')

    # ef = EdgeFinder(_input_path)
    # ef.run()
    # ef.save(DATA_DIR)

    ef = EdgeFinder.load(DATA_DIR)
    ef.refine_contours()
    ef.process_jigsaws()
    pass