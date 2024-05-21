## To run dataset coliee2023, please change '--data 2022' to '--data 2023'

python PromptCase/preprocessing/process.py --data 2022 --dataset train

python PromptCase/preprocessing/reference.py --data 2022 --dataset train

python PromptCase/preprocessing/openaiAPI.py --data 2022 --dataset train

python PromptCase/preprocessing/process.py --data 2022 --dataset test 

python PromptCase/preprocessing/reference.py --data 2022 --dataset test

python PromptCase/preprocessing/openaiAPI.py --data 2022 --dataset test