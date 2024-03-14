import sys

from src import MLPipeline, parse_eval_args

args = parse_eval_args(sys.argv[1:])

test_pipeline = MLPipeline.for_evaluation(args)
test_pipeline.run()
