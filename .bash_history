pip install kaggle
clear
cd dataset/
kaggle datasets download -d olgabelitskaya/art-pictogram
chmod 600 /teamspace/studios/this_studio/.kaggle/kaggle.json
clear
ls
unzip art-pictogram.zip 
clear
cd ..
python3 main.py 
python3 data_preparation_pictograms.py 
python3 data_preparation_pictograms.py 
python3 data_preparation_pictograms.py 
python3 data_preparation_pictograms.py 
clear
python3 main.py 
clear
python3 main.py 
clear
python3 main.py 
clear
python3 utils/path.py 
pwd
clear
pip install tree
tree -L 2
tree 
clear
clear
ls
clear
python3 main.py 
clear
python3 main.py 
clear
python3 main.py 
clear
python3 main.py 
clear
python3 main.py 
clear
python3 main.py 
python3 eval.py 
python3 main.py 
python3 main.py 
clear
python3 main.py 
python3 main.py 
python3 eval.py 
python3 main.py 
python3 eval.py 
python3 eval.py 
python3 eval.py 
python3 eval.py 
clear
python3 eval.py 
python3 train.py --config_path configs/resnet50.json 
clear
python3 train.py --config_path configs/resnet50.json 
clear
python3 train.py --config_path configs/resnet50.json 
python3 train.py --config_path configs/resnet50.json 
python3 train.py --config_path configs/vit.json 
python3 train.py --config_path configs/vit.json 
python3 train.py --config_path configs/vit.json 
clear
python3 train.py --config_path configs/resnet50.json 
clear
python3 train.py --config_path configs/resnet50.json 
pip install wandb
clear
python3 train.py --config_path configs/resnet50.json 
clear
python3 train.py --config_path configs/resnet50.json --max_epochs 3 
clear
python3 train.py --config_path configs/resnet50.json --max_epochs 3 
clear
python3 train.py --config_path configs/resnet50.json --max_epochs 3 
clear
python3 train.py --config_path configs/resnet50.json --max_epochs 3 
clear
python3 train.py --config_path configs/resnet50.json --max_epochs 3 
clear
python3 train.py --config_path configs/resnet50.json --max_epochs 3 
clear
python3 train.py --config_path configs/resnet50.json --max_epochs 3 
clear
python3 train.py --config_path configs/resnet50.json --max_epochs 3 
python3 train.py --config_path configs/vit.json --max_epochs 3 
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_0%_test1 
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_0%_test2 
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_0%_test2 --resume --id "m1xm44qt" --ckpt_path experiments/resnet50/resnet50_0%_test2/last.ckpt
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_0%_test3 
clear
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_0%_test3 
clear
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_0%_test3 
clear
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_0%_test3 
clear
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_0%_test3 
clear
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_0%_test3 
git
clear
git
clear
git --version
git config --global user.name "FlowerCS"
git config --global user.email "carlos.flores.p@utec.edu.pe"
git init
pip list
python3 -m pip freeze > requirements.txt
git remode add origin https://github.com/FlowersCS/visual-agnosia-simulation.git
git remote add origin https://github.com/FlowersCS/visual-agnosia-simulation.git
git add .
git commint -m "Preparation and setup"
git commit -m "Preparation and setup"
git push -u origin master
git rm -r --cached .idea .vscode __pycache__ .bash_history
git status
git commit -m "Removed ignored files and folders"
git push origin main
git push origin main
git push origin master
clear
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_0%_test4
clear
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_0%_test4 --max_epochs 10
python3 eval.py 
ls
mv ILSVRC2012_img_test_v10102019.tar dataset/ILSVRC2012_img_test_v10102019.tar
cd dataset/
tar -xvf ILSVRC2012_img_test_v10102019.tar 
clear
cd ..
python3 eval.py --data_dir dataset/test/
cd dataset/
mv ILSVRC2012_devkit_t12.tar.gz test/ILSVRC2012_devkit_t12.tar.gz
cd test/
tar -xzvf ILSVRC2012_devkit_t12.tar.gz 
cd ~
python3 eval.py --data_dir dataset/test/
mv dataset/test/ILSVRC2012_devkit_t12.tar.gz dataset/test/ILSVRC2012_devkit_t12.tar.gz
cd dataset/test/
mv ILSVRC2012_devkit_t12.tar.gz ../test/
mv ILSVRC2012_devkit_t12.tar.gz ../test/ILSVRC2012_devkit_t12.tar.gz
mv ILSVRC2012_devkit_t12.tar.gz ../ILSVRC2012_devkit_t12.tar.gz
cd ..
ls
cd ..
ls
cd dataset/
rm -rf pictograms/
rm -rf test/
rm art-pictogram.zip 
clear
cd ..
python3 eval.py --data_dir dataset/test/
python3 eval.py --data_dir dataset/
python3 eval.py --data_dir dataset/
cd dataset/
tar -xvf ILSVRC2012_img_test_v10102019.tar 
cd ~
python3 eval.py --data_dir dataset/
tree -L 2
. tree -L 2
tree .
pip install tree
tree .
cd dataset/
tar -xzvf ILSVRC2012_devkit_t12.tar.gz
python3 script.py 
python3 script.py 
cd test/
ls
clear
cd ..
python3 script.py 
cd ..
python3 script.py 
python3 dataset/script.py 
clear
python3 dataset/script.py 
python3 eval.py --data_dir dataset/
clear
python3 eval.py --data_dir dataset/
python3 eval.py --data_dir dataset/
python3 eval.py --data_dir dataset/ --weight_diminution 0 --prune_type initial
cd dataset/
tar -xvf ILSVRC2012_img_val.tar 
rm ILSVRC2012_val_000*.JPEG
mkdir val
mv ILSVRC2012_img_val.tar val/
cd val/
ls
mv ILSVRC2012_img_val.tar ../
ls
cd ..
tar -xvf ILSVRC2012_img_val.tar -C val/
clear
python3 script.py 
cd ~
python3 dataset/script.py 
cd dataset/
ls val_organized/
clear
cd ..
python3 eval.py --data_dir dataset/
cd dataset/
#!/bin/bash
kaggle datasets download marquis03/plants-classification
cd dataset/
rm -rf ILSVRC2012_devkit_t12
rm -rf test/
rm -rf test_organized/
rm -rf val_organized/
clear
unzip plants-classification.zip
clear
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50test1_plants
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50test1_plants --max_epochs 50
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50test1_plants --max_epochs 50
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50test2_plants --max_epochs 50
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50test3_plants --max_epochs 50
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50test4_plants --max_epochs 50
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50test5_plants --max_epochs 50
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_20%_initial --max_epochs 50
clear
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_20%_initial --max_epochs 50
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_20%_initial --max_epochs 50
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_20%_initial --max_epochs 50
clear
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_20%_initial --max_epochs 50
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_70%_initial --max_epochs 50
clear
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_70%_initial --max_epochs 50
clear
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_70%_initial --max_epochs 50
clear
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_70%_initial --max_epochs 50
clear
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_70%_initial --max_epochs 50
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_70%_v2_initial --max_epochs 50
clear
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_70%_v3_initial --max_epochs 5
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_70%_v3_final --max_epochs 5
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_20%_v2_initial --max_epochs 5
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_20%_v3_initial --max_epochs 5
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_50%_initial --max_epochs 5
python3 train.py --config_path configs/resnet50.json --experiment_name resnet50_50%_final --max_epochs 5
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.2 --layers_to_prune initial
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.2 --layers_to_prune initial --experiment_name testresnet20%
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.2 --layers_to_prune initial --experiment_name testresnet20%
clear
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.2 --layers_to_prune initial --experiment_name testresnet20%
clear
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.2 --layers_to_prune initial --experiment_name testresnet20%
python3 train.py --config_path configs/resnet50.json --max_epochs 10 --experiment_name resnet50_0% 
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.2 --layers_to_prune initial --experiment_name resnet50_20%_initial --ckpt_path experiments/resnet50/resnet50_0%/last.ckpt 
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.3 --layers_to_prune initial --experiment_name resnet50_30%_initial --ckpt_path experiments/resnet50/resnet50_0%/last.ckpt 
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.5 --layers_to_prune initial --experiment_name resnet50_50%_initial --ckpt_path experiments/resnet50/resnet50_0%/last.ckpt 
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.7 --layers_to_prune initial --experiment_name resnet50_70%_initial --ckpt_path experiments/resnet50/resnet50_0%/last.ckpt 
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.2 --layers_to_prune final --experiment_name resnet50_20%_final --ckpt_path experiments/resnet50/resnet50_0%/last.ckpt 
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.2 --layers_to_prune final --experiment_name resnet50_20%_final --ckpt_path experiments/resnet50/resnet50_0%/last.ckpt 
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.3 --layers_to_prune final --experiment_name resnet50_30%_final --ckpt_path experiments/resnet50/resnet50_0%/last.ckpt 
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.5 --layers_to_prune final --experiment_name resnet50_50%_final --ckpt_path experiments/resnet50/resnet50_0%/last.ckpt 
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.7 --layers_to_prune final --experiment_name resnet50_70%_final --ckpt_path experiments/resnet50/resnet50_0%/last.ckpt 
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.0 --layers_to_prune final --experiment_name resnet50_0%_test --ckpt_path experiments/resnet50/resnet50_0%/last.ckpt 
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.0 --layers_to_prune initial --experiment_name resnet50_0%_test_2 --ckpt_path experiments/resnet50/resnet50_0%/last.ckpt 
clear
chmod +x run_test.sh 
clear
python3 train.py --config_path configs/resnet50.json --max_epochs 10 --experiment_name resnet50_0%_v2
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.0 --layers_to_prune initial --experiment_name resnet50_0%_test_2 --ckpt_path experiments/resnet50/resnet50_0%_v2/last.ckpt
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.0 --layers_to_prune final --experiment_name resnet50_0%_test_2 --ckpt_path experiments/resnet50/resnet50_0%_v2/last.ckpt
python3 train.py --config_path configs/resnet50.json --max_epochs 10 --experiment_name resnet50_0%_v3
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.0 --layers_to_prune final --experiment_name resnet50_0%_test_3 --ckpt_path experiments/resnet50/resnet50_0%_v3/last.ckpt
python3 train.py --config_path configs/resnet50.json --max_epochs 10 --experiment_name resnet50_0%_v4
clear
python3 train.py --config_path configs/resnet50.json --max_epochs 10 --experiment_name resnet50_0%_v5
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.0 --layers_to_prune final --experiment_name resnet50_0%_test_5 --ckpt_path experiments/resnet50/resnet50_0%_v5/last.ckpt
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.0 --layers_to_prune final --experiment_name resnet50_0%_test_5 --ckpt_path experiments/resnet50/resnet50_0%_v5/last.ckpt
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.0 --layers_to_prune final --experiment_name resnet50_0%_test_3 --ckpt_path experiments/resnet50/resnet50_0%_v3/last.ckpt
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.0 --layers_to_prune final --experiment_name resnet50_0%_test_5 --ckpt_path experiments/resnet50/resnet50_0%_v5/last.ckpt
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.0 --layers_to_prune final --experiment_name resnet50_0%_test_3 --ckpt_path experiments/resnet50/resnet50_0%_v3/last.ckpt
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.0 --layers_to_prune final --experiment_name resnet50_0%_test_3 --ckpt_path experiments/resnet50/resnet50_0%_v3/last.ckpt
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.0 --layers_to_prune final --experiment_name resnet50_0%_test_4 --ckpt_path experiments/resnet50/resnet50_0%_v4/last.ckpt
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.0 --layers_to_prune final --experiment_name resnet50_0%_test_2 --ckpt_path experiments/resnet50/resnet50_0%_v2/last.ckpt
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.0 --layers_to_prune final --experiment_name resnet50_0%_test_1 --ckpt_path experiments/resnet50/resnet50_0%/last.ckpt
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.0 --layers_to_prune final --experiment_name resnet50_0%_test_5 --ckpt_path experiments/resnet50/resnet50_0%_v5/last.ckpt
clear
./run_test.sh 
clear
./run_test.sh 
clear
./run_test.sh 
sudo apt-get install bc
clear
./run_test.sh 
clear
./run_test.sh 
clear
./run_test.sh 
./run_test.sh 
clear
./run_test.sh 
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.0 --layers_to_prune final --experiment_name resnet50_0%_initial --ckpt_path experiments/resnet50/resnet50_0%/last.ckpt
python3 train.py --config_path configs/vit.json --max_epochs 10 --experiment_name vittest 
clear
python3 train.py --config_path configs/vit.json --max_epochs 10 --experiment_name vittest 
python3 train.py --config_path configs/vit.json --max_epochs 10 --experiment_name vittest 
python3 test_pruning.py --config_path configs/vit.json --pruning_amount 0.0 --layers_to_prune final --experiment_name vit_0% --ckpt_path experiments/vit/vittest/last.ckpt
clear
./run_test.sh 
python3 test_pruning.py --config_path configs/vit.json --pruning_amount 0.2 --layers_to_prune initial --experiment_name vit_20%_initial --ckpt_path experiments/vit/vittest/last.ckpt
clear
./run_test.sh 
python3 test_pruning.py --config_path configs/vit.json --pruning_amount 0.0 --layers_to_prune initial --experiment_name vit_0%_initial --ckpt_path experiments/vit/vittest/last.ckpt
python3 train.py --config_path configs/basic_cnn.json --experiment_name cnn_test1
clear
python3 train.py --config_path configs/basic_cnn.json --experiment_name cnn_test1
clear
python3 train.py --config_path configs/basic_cnn.json --experiment_name cnn_test1
clear
python3 train.py --config_path configs/basic_cnn.json --experiment_name cnn_test1
python3 train.py --config_path configs/basic_cnn.json --experiment_name cnn_test1
python3 train.py --config_path configs/basic_cnn.json --experiment_name cnn_test2
clear
python3 train.py --config_path configs/basic_cnn.json --experiment_name cnn_test2 --max_epochs 10
python3 train.py --config_path configs/basic_cnn.json --experiment_name cnn_test3 --max_epochs 10
python3 train.py --config_path configs/basic_cnn.json --experiment_name cnn_test4 --max_epochs 10
python3 test_pruning.py --config_path configs/basic_cnn.json --pruning_amount 0.0 --layers_to_prune initial --experiment_name cnn_0%_initial --ckpt_path experiments/BasicCNN/cnn_test1/last.ckpt 
./run_test.sh 
clear
clear
./run_test.sh 
clear
clear
./run_test.sh 
clear
clear
./run_test.sh 
clear
python3 train.py --config_path configs/basic_cnn.json --experiment_name cnn_test5 --max_epochs 30
clear
python3 train.py --config_path configs/basic_cnn.json --experiment_name cnn_test5 --max_epochs 30
python3 train.py --config_path configs/basic_cnn.json --max_epochs 60 --ckpt_path experiments/BasicCNN/cnn_test5/last.ckpt --experiment_name cnn_test5 --resume --id "0zhi54lj"
python3 train.py --config_path configs/basic_cnn.json --max_epochs 90 --ckpt_path experiments/BasicCNN/cnn_test5/last.ckpt --experiment_name cnn_test5 --resume --id "0zhi54lj"
./run_test.sh 
clear
clear
./run_test.sh 
python3 test_pruning.py --config_path configs/basic_cnn.json --pruning_amount 0.0 --layers_to_prune "initial" --experiment_name cnntest5wa --ckpt_path experiments/BasicCNN/cnn_test5/last.ckpt --project Exposicion_1_proyecto
python3 test_pruning.py --config_path configs/basic_cnn.json --pruning_amount 0.2 --layers_to_prune "final" --experiment_name cnntest5wa --ckpt_path experiments/BasicCNN/cnn_test5/last.ckpt --project Exposicion_1_proyecto
python3 test_pruning.py --config_path configs/basic_cnn.json --pruning_amount 0.3 --layers_to_prune "final" --experiment_name cnntest5wa --ckpt_path experiments/BasicCNN/cnn_test5/last.ckpt --project Exposicion_1_proyecto
python3 test_pruning.py --config_path configs/basic_cnn.json --pruning_amount 0.9 --layers_to_prune "final" --experiment_name cnntest5wa --ckpt_path experiments/BasicCNN/cnn_test5/last.ckpt --project Exposicion_1_proyecto
python3 test_pruning.py --config_path configs/basic_cnn.json --pruning_amount 0.2 --layers_to_prune "final" --experiment_name cnntest5wa --ckpt_path experiments/BasicCNN/cnn_test5/last.ckpt --project Exposicion_1_proyecto
python3 test_pruning.py --config_path configs/basic_cnn.json --pruning_amount 0.2 --layers_to_prune "final" --experiment_name cnntest5wa --ckpt_path experiments/BasicCNN/cnn_test5/last.ckpt --project Exposicion_1_proyecto
python3 train.py --config_path configs/vit.json --max_epochs 10 --experiment_name vit_test2
clear
python3 train.py --config_path configs/vit.json --max_epochs 10 --experiment_name vit_test2 
clear
python3 train.py --config_path configs/vit.json --max_epochs 10 --experiment_name vit_test2 
clear
python3 train.py --config_path configs/vit.json --max_epochs 10 --experiment_name vit_test2 
clear
python3 train.py --config_path configs/vit.json --max_epochs 10 --experiment_name vit_test2 
python3 train.py --config_path configs/vit.json --max_epochs 10 --experiment_name vit_test2 
python3 train.py --config_path configs/vit.json --max_epochs 10 --experiment_name vit_test3
./run_test.sh 
clear
./run_test.sh 
clear
./run_test.sh 
python3 test_pruning.py --config_path configs/vit.json --pruning_amount 0.0 --layers_to_prune "final" --experiment_name vittest5wa --ckpt_path experiments/vit/vit_test3/last.ckpt --project Exposicion_1_proyecto
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.0 --layers_to_prune "initial" --ckpt_path experiments/resnet50/resnet50_0%/last.ckpt --experiment_name resnet50_poda_aleatoria
python3 test_pruning.py --config_path configs/resnet50.json --pruning_amount 0.0 --layers_to_prune "initial" --ckpt_path experiments/resnet50/resnet50_0%/last.ckpt --experiment_name resnet50_poda_aleatoria
./run_test.sh 
clear
./run_test.sh 
