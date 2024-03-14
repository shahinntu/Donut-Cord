import sys

from src import MLPipeline, parse_train_args

args = parse_train_args(sys.argv[1:])

train_pipeline = MLPipeline.for_training(args)
train_pipeline.run()
