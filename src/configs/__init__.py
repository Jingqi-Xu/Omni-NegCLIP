from types import SimpleNamespace


configs = SimpleNamespace(**{})


## CC-Neg (for fine-tuning and evaluation)
#configs.ccneg_root_folder = "D:/download/CoN-CLIP/ccneg_dataset"         ## change this accordingly                                    
#configs.finetuning_dataset_path = f"{configs.ccneg_root_folder}/ccneg_preprocessed.pt"                                  
#configs.negative_image_ft_mapping_path = f"{configs.ccneg_root_folder}/distractor_image_mapping.pt"
#configs.num_ccneg_eval_samples = 40000  ## the last 40,000 indices consist of the decided evaluation split for CC-Neg
## MS-COCO (for finetuning)                                                                                             
#configs.coco_root_folder = "D:/download/coco"           ## change this accordingly
#configs.negative_image_dataset_root = f"{configs.coco_root_folder}/train2017"    ## path to coco train2017 images                                                                                                                
#configs.negative_image_dataset_annotations_path = f"{configs.coco_root_folder}/annotations/captions_train2017.json"    ## path to coco train2017 annotations (not used but required by torchvision dataset)

## CC-Neg (for fine-tuning and evaluation)
configs.ccneg_root_folder = "/project2/pabeerel_971/LucaXu/CoN-CLIP/ccneg_dataset"         ## CARC path                                    
configs.finetuning_dataset_path = f"{configs.ccneg_root_folder}/ccneg_preprocessed.pt"                                  
configs.negative_image_ft_mapping_path = f"{configs.ccneg_root_folder}/distractor_image_mapping.pt"
configs.num_ccneg_eval_samples = 40000  ## the last 40,000 indices consist of the decided evaluation split for CC-Neg
## MS-COCO (for finetuning) - NOT USED when negative_images="off"                                                                                            
configs.coco_root_folder = "/project2/pabeerel_971/LucaXu/coco"           ## not used
configs.negative_image_dataset_root = f"{configs.coco_root_folder}/train2017"    ## not used                                                                                                                
configs.negative_image_dataset_annotations_path = f"{configs.coco_root_folder}/annotations/captions_train2017.json"    ## not used