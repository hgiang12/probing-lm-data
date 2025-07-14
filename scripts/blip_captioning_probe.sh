export CUDA_VISIBLE_DEVICES=0

for model_path in "Salesforce/blip-image-captioning-base" "Salesforce/blip-image-captioning-large"; do

    echo -e "\n==============================="
    echo -e "Running for model: ${model_path}"
    echo -e "===============================\n"

    python src/generate_acts_blip.py \
        --model_path $model_path \
        --dataset stock_dev \
        --dataset_path ./data/stock_dev.jsonl \
        --images_dir ./data/images \
        --output_dir ./acts/

    python src/generate_acts_blip.py \
        --model_path $model_path \
        --dataset stock_test \
        --dataset_path ./data/stock_test.jsonl \
        --images_dir ./data/images \
        --output_dir ./acts/

    for lr in 2e-5 3e-5 4e-5; do

        echo -e "***********************"
        echo -e "learning rate: ${lr}"
        echo -e "***********************" 

        python src/ft_proxy_model_ds_blip.py \
            --model_path $model_path \
            --data_path ./data/stock_train.jsonl \
            --images_dir ./data/images \
            --epochs 2 \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --lr $lr \
            --save_dir ./saved_models

        python src/generate_acts_blip.py \
            --model_path ./saved_models/$(basename $model_path) \
            --dataset stock_train \
            --dataset_path ./data/stock_train.jsonl \
            --images_dir ./data/images \
            --output_dir ./acts/

        python src/run_probe_blip.py \
            --seed 42 \
            --target_model $(basename $model_path) \
            --train_set stock_train \
            --train_set_path ./data/stock_train.jsonl \
            --dev_set stock_dev \
            --dev_set_path ./data/stock_dev.jsonl \
            --test_set stock_test \
            --test_set_path ./data/stock_test.jsonl

    done

done