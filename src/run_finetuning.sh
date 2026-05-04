#python3 conclip_fine_tuning.py \
	#--clip-model-name=ViT-B/32 \
	#--experiment-name=conclip_b32 \
	#--negative-images=off \
	#--lock-image-encoder=on \
	#--batch-size=200 \
	#--num-workers=4
	
# 使用 OriginalLoss（没有 L3）
#python3 conclip_fine_tuning.py --negative-images=original --experiment-name=original_loss_exp --clip-model-name=ViT-B/32 --lock-image-encoder=on --batch-size=200 --num-workers=4

#finetune CLIP
#python3 conclip_fine_tuning.py \
    #--clip-model-name=ViT-B/32 \
    #--experiment-name=finetuneclip_exp \
    #--negative-images=finetuneclip \
    #--lock-image-encoder=on \
    #--batch-size=200 \
    #--num-workers=4
    
# negclip(L123)
python3 conclip_fine_tuning.py \
    --clip-model-name=ViT-B/32 \
    --experiment-name=conclip_b32_stage1_30epoch \
    --negative-images=off \
    --lock-image-encoder=on \
    --batch-size=200 \
    --num-workers=4
