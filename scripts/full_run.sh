# Train models
# python ./tools/train.py --dataset_dir ./datasets/RAVDESS --label_index 7 --device 0 --out_dir './results/en/' --overwrite --epoch 500
# python ./tools/train.py --dataset_dir ./datasets/Xper_DES --label_index 0 --device 0 --out_dir './results/jp/' --overwrite --epoch 500
# python ./tools/train.py --dataset_dir ./datasets/Xper_DES --label_index 0 --device 0 --out_dir './results/en2jp/' --init_weights ./results/en/model_epoch-500 --overwrite --epoch 500

# Evaluate models
python ./tools/evaluate.py --dataset_dir ./datasets/Xper_DES_test --label_index 0 --device 0 --out_dir ./results/en/ --init_weights  ./results/en/model_epoch-500 --title 'RAVDESS Model'
python ./tools/evaluate.py --dataset_dir ./datasets/Xper_DES_test --label_index 0 --device 0 --out_dir ./results/jp/ --init_weights  ./results/jp/model_epoch-500 --title 'XperDES Model'
python ./tools/evaluate.py --dataset_dir ./datasets/Xper_DES_test --label_index 0 --device 0 --out_dir ./results/en2jp/ --init_weights  ./results/en2jp/model_epoch-500 --title 'Finetune XperDES Model'
