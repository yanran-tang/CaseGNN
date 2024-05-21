## To run dataset coliee2023, please change '--data 2022' to '--data 2023'

python Information_extraction/knowledge_graph.py --data 2022 --dataset train --feature fact

python Information_extraction/relation_extractor.py --data 2022 --dataset train --feature fact

python Information_extraction/create_structured_csv.py --data 2022 --dataset train --feature fact

python Information_extraction/knowledge_graph.py --data 2022 --dataset train --feature issue

python Information_extraction/relation_extractor.py --data 2022 --dataset train --feature issue

python Information_extraction/create_structured_csv.py --data 2022 --dataset train --feature issue

python Information_extraction/knowledge_graph.py --data 2022 --dataset test --feature fact

python Information_extraction/relation_extractor.py --data 2022 --dataset test --feature fact

python Information_extraction/create_structured_csv.py --data 2022 --dataset test --feature fact

python Information_extraction/knowledge_graph.py --data 2022 --dataset test --feature issue

python Information_extraction/relation_extractor.py --data 2022 --dataset test --feature issue

python Information_extraction/create_structured_csv.py --data 2022 --dataset test --feature issue