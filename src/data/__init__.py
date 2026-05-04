from .finetuning_datasets import *
from .evaluation_datasets import *


#def get_finetuning_dataset(args, preprocess):
	#if args.negative_images == "off":
		#return FineTuningDataset(args, preprocess)
	#elif args.negative_images == "on":
		#return FineTuningDatasetWithNegatives(args, preprocess)
	#elif args.negative_images == "on+":
		#return FineTuningDatasetWithNegatives(args, preprocess)
		
def get_finetuning_dataset(args, preprocess):
	if args.negative_images == "off":
		return FineTuningDataset(transform=preprocess)
	elif args.negative_images == "on":
		return FineTuningDatasetWithNegatives(transform=preprocess)
	elif args.negative_images == "on+":
		return FineTuningDatasetWithNegatives(transform=preprocess)
	elif args.negative_images == "original":
		# original 模式也使用 FineTuningDataset（返回 image, caption, negative_caption）
		# 但 OriginalLoss 只使用 L1 + L2，不使用 L3
		return FineTuningDataset(transform=preprocess)
	elif args.negative_images == "finetuneclip":
		# finetuneclip 模式只使用正确描述，不使用否定描述
		# 返回 (image, caption)
		return OriginalFineTuningDataset(transform=preprocess)


	