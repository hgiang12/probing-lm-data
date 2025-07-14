export CUDA_VISIBLE_DEVICES=0

model_path="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


python src/generate_acts.py \
    --model_path $model_path \
    --dataset arxiv_mia_dev \
    --dataset_path ./data/arxiv_mia_dev.jsonl

python src/generate_acts.py \
    --model_path $model_path \
    --dataset arxiv_mia_test \
    --dataset_path ./data/arxiv_mia_test.jsonl


r=8
alpha=16
bias="lora_only"


for lr in 8e-4 9e-4 1e-3; do

    echo -e "***********************"
    echo -e "learning rate: ${lr}"
    echo -e "***********************"    

    echo -e "\nProbing with baseline (LogReg) and scikit-learn classifiers (SCK)"
    echo -e "-----------------------"

    python src/ft_proxy_model_ds.py \
        --model_path $model_path \
        --seed 42 \
        --data_path ./data/arxiv_mia_train_real.jsonl \
        --epochs 2 \
        --per_device_train_batch_size 25 \
        --gradient_accumulation_steps 4 \
        --lr $lr

    python src/generate_acts.py \
        --dataset arxiv_mia_train_real \
        --dataset_path ./data/arxiv_mia_train_real.jsonl \
        --model_path ./saved_models/$(basename $model_path)

    python src/run_probe.py \
        --seed 42 \
        --target_model $(basename $model_path) \
        --train_set arxiv_mia_train_real \
        --train_set_path ./data/arxiv_mia_train_real.jsonl \
        --dev_set arxiv_mia_dev \
        --dev_set_path ./data/arxiv_mia_dev.jsonl \
        --test_set arxiv_mia_test \
        --test_set_path ./data/arxiv_mia_test.jsonl

    python src/run_probe_sck.py \
        --seed 42 \
        --target_model $(basename $model_path) \
        --train_set arxiv_mia_train_real \
        --train_set_path ./data/arxiv_mia_train_real.jsonl \
        --dev_set arxiv_mia_dev \
        --dev_set_path ./data/arxiv_mia_dev.jsonl \
        --test_set arxiv_mia_test \
        --test_set_path ./data/arxiv_mia_test.jsonl

    echo -e "\n[LoRA] Finetuning and probing"
    echo -e "-----------------------"

    python src/ft_proxy_model_ds_lora.py \
        --model_path $model_path \
        --seed 42 \
        --data_path ./data/arxiv_mia_train_real.jsonl \
        --epochs 2 \
        --per_device_train_batch_size 25 \
        --gradient_accumulation_steps 4 \
        --lr $lr \
        --r $r \
        --alpha $alpha \
        --bias $bias

    python src/generate_acts_lora.py \
        --dataset arxiv_mia_train_real \
        --dataset_path ./data/arxiv_mia_train_real.jsonl \
        --model_path ./saved_models/$(basename $model_path)

    python src/run_probe.py \
        --seed 42 \
        --target_model $(basename $model_path) \
        --train_set arxiv_mia_train_real \
        --train_set_path ./data/arxiv_mia_train_real.jsonl \
        --dev_set arxiv_mia_dev \
        --dev_set_path ./data/arxiv_mia_dev.jsonl \
        --test_set arxiv_mia_test \
        --test_set_path ./data/arxiv_mia_test.jsonl

    echo -e "\n[QLoRA] Finetuning and probing"
    echo -e "-----------------------"

    python src/ft_proxy_model_ds_qlora.py \
        --model_path $model_path \
        --seed 42 \
        --data_path ./data/arxiv_mia_train_real.jsonl \
        --epochs 2 \
        --per_device_train_batch_size 25 \
        --gradient_accumulation_steps 4 \
        --lr $lr \
        --r $r \
        --alpha $alpha \
        --bias $bias

    python src/generate_acts_lora.py \
        --dataset arxiv_mia_train_real \
        --dataset_path ./data/arxiv_mia_train_real.jsonl \
        --model_path ./saved_models/$(basename $model_path)

    python src/run_probe.py \
        --seed 42 \
        --target_model $(basename $model_path) \
        --train_set arxiv_mia_train_real \
        --train_set_path ./data/arxiv_mia_train_real.jsonl \
        --dev_set arxiv_mia_dev \
        --dev_set_path ./data/arxiv_mia_dev.jsonl \
        --test_set arxiv_mia_test \
        --test_set_path ./data/arxiv_mia_test.jsonl
done
