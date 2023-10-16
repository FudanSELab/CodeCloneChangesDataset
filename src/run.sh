CUDA_VISIBLE_DEVICES=0 python gcn.py \
    --pdg_path=../dataset/FC-PDG \
    --model_save_dir=../saveModels \
    --graph_data_file=./graph.bin \
    --do_embedding \
    --project_info_file=./projectInfo.csv \
    --project_test_dataset_file=./testDataset.scv \
    --num_train_epochs 100 \
    --learning_rate 1e-4  \
    --batch_size 20 \
    --hidden_dim 400 \
    --softmax_threhold 0.5 \
    2>&1 | tee train.log