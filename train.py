import torch
import torch.nn as nn
from tqdm import tqdm
import random
import dgl
import os
import json
from torch_metrics import t_metrics, metric, yf_metric, rank


def forward(data, model, device, writer, dataloader, sumfact_pool_dataset, referissue_pool_dataset, label_dict, yf_path, epoch, temp, bm25_hard_neg_dict, hard_neg, hard_neg_num, train_flag, embedding_saving, optimizer=None):
    if train_flag:
        ## Training
        loss_model = nn.CrossEntropyLoss()
        model.train()
        optimizer.zero_grad()

        for batched_graph, labels in tqdm(dataloader):
            batched_case_list = []
            for i in range(len(labels)):
                batched_case_list.append(str(int(labels[i])).zfill(6))

            query_sumfact_graph = []
            query_referissue_graph = []

            positive_sumfact_graph = []
            positive_referissue_graph = []

            ran_neg_sumfact_graph = []
            ran_neg_referissue_graph = []

            bm25_neg_sumfact_graph = []
            bm25_neg_referissue_graph = []
            for x in range(len(batched_case_list)):
                query_name = batched_case_list[x] + '.txt'
                query_sumfact_graph.append(sumfact_pool_dataset.graphs[batched_case_list[x] ])
                query_referissue_graph.append(referissue_pool_dataset.graphs[batched_case_list[x]])

                pos_case = random.choice(label_dict[query_name]).split('.')[0]
                positive_sumfact_graph.append(sumfact_pool_dataset.graphs[pos_case])  
                positive_referissue_graph.append(referissue_pool_dataset.graphs[pos_case]) 
                i = 0
                while i<4400: 
                    ran_neg_case = random.choice(list(sumfact_pool_dataset.labels.keys()))
                    if ran_neg_case+'.txt' not in label_dict[query_name]:
                        break
                    break

                ran_neg_sumfact_graph.append(sumfact_pool_dataset.graphs[ran_neg_case])
                ran_neg_referissue_graph.append(referissue_pool_dataset.graphs[ran_neg_case])
                
                for i in range(hard_neg_num):
                    bm25_neg_case = random.choice(bm25_hard_neg_dict[query_name]).split('.')[0]
                    bm25_neg_sumfact_graph.append(sumfact_pool_dataset.graphs[bm25_neg_case])  
                    bm25_neg_referissue_graph.append(referissue_pool_dataset.graphs[bm25_neg_case]) 
                
            que_sumfact_batch = dgl.batch(query_sumfact_graph)
            que_referissue_batch = dgl.batch(query_referissue_graph)
            pos_sumfact_batch = dgl.batch(positive_sumfact_graph)
            pos_referissue_batch = dgl.batch(positive_referissue_graph)
            ran_sumfact_batch = dgl.batch(ran_neg_sumfact_graph)
            ran_referissue_batch = dgl.batch(ran_neg_referissue_graph)
            bm25_sumfact_batch = dgl.batch(bm25_neg_sumfact_graph)
            bm25_referissue_batch = dgl.batch(bm25_neg_referissue_graph)

            # query: input_1_fact, input_1_issue
            feat_1_node_fact = que_sumfact_batch.ndata['w']
            feat_1_edge_fact = que_sumfact_batch.edata['w']
            output_1_fact = model(que_sumfact_batch.to(device), feat_1_node_fact.to(device), feat_1_edge_fact.to(device))
            output_1_norm_fact = output_1_fact / output_1_fact.norm(dim=1)[:, None]

            feat_1_node_issue = que_referissue_batch.ndata['w']
            feat_1_edge_issue = que_referissue_batch.edata['w']
            output_1_issue = model(que_referissue_batch.to(device), feat_1_node_issue.to(device), feat_1_edge_issue.to(device))
            output_1_norm_issue = output_1_issue / output_1_issue.norm(dim=1)[:, None]

            # pos: input_2_fact, input_2_issue
            feat_2_node_fact = pos_sumfact_batch.ndata['w']
            feat_2_edge_fact = pos_sumfact_batch.edata['w']
            output_2_fact = model(pos_sumfact_batch.to(device), feat_2_node_fact.to(device), feat_2_edge_fact.to(device))
            output_2_norm_fact = output_2_fact / output_2_fact.norm(dim=1)[:, None]

            feat_2_node_issue = pos_referissue_batch.ndata['w']
            feat_2_edge_issue = pos_referissue_batch.edata['w']
            output_2_issue = model(pos_referissue_batch.to(device), feat_2_node_issue.to(device), feat_2_edge_issue.to(device))
            output_2_norm_issue = output_2_issue / output_2_issue.norm(dim=1)[:, None]

            # ran_neg: input_3_fact, input_3_issue
            feat_3_node_fact = ran_sumfact_batch.ndata['w']
            feat_3_edge_fact = ran_sumfact_batch.edata['w']
            output_3_fact = model(ran_sumfact_batch.to(device), feat_3_node_fact.to(device), feat_3_edge_fact.to(device))
            output_3_norm_fact = output_3_fact / output_3_fact.norm(dim=1)[:, None]

            feat_3_node_issue = ran_referissue_batch.ndata['w']
            feat_3_edge_issue = ran_referissue_batch.edata['w']
            output_3_issue = model(ran_referissue_batch.to(device), feat_3_node_issue.to(device), feat_3_edge_issue.to(device))
            output_3_norm_issue = output_3_issue / output_3_issue.norm(dim=1)[:, None]

            # bm25_neg: input_4
            feat_4_node_fact = bm25_sumfact_batch.ndata['w']
            feat_4_edge_fact = bm25_sumfact_batch.edata['w']
            output_4_fact = model(bm25_sumfact_batch.to(device), feat_4_node_fact.to(device), feat_4_edge_fact.to(device))
            output_4_norm_fact = output_4_fact / output_4_fact.norm(dim=1)[:, None]

            feat_4_node_issue = bm25_referissue_batch.ndata['w']
            feat_4_edge_issue = bm25_referissue_batch.edata['w']
            output_4_issue = model(bm25_referissue_batch.to(device), feat_4_node_issue.to(device), feat_4_edge_issue.to(device))
            output_4_norm_issue = output_4_issue / output_4_issue.norm(dim=1)[:, None]


            # positive logits: l_pos[batch_size, batch_size]:output_1 x output_2
            l_pos_fact = torch.mm(output_1_norm_fact, output_2_norm_fact.transpose(0,1))
            l_pos_issue = torch.mm(output_1_norm_issue, output_2_norm_issue.transpose(0,1))
            l_pos = l_pos_fact+l_pos_issue  
            
            # negative logits: l_neg[batch_size, batch_size]: AxA 
            ## diagonal is the dot product of query and itself
            l_neg_fact = torch.mm(output_1_norm_fact, output_1_norm_fact.transpose(0,1))
            l_neg_issue = torch.mm(output_1_norm_issue, output_1_norm_issue.transpose(0,1))
            l_neg = l_neg_fact+l_neg_issue
            l_neg.fill_diagonal_(float('-inf'))

            ## random negative logits: l_ran_neg[batch_size, batch_size]:
            l_ran_neg_fact = torch.mm(output_1_norm_fact, output_3_norm_fact.transpose(0,1)) 
            l_ran_neg_issue = torch.mm(output_1_norm_issue, output_3_norm_issue.transpose(0,1))
            l_ran_neg = l_ran_neg_fact+l_ran_neg_issue           

            l_bm25_neg_fact = torch.mm(output_1_norm_fact, output_4_norm_fact.transpose(0,1))
            l_bm25_neg_issue = torch.mm(output_1_norm_issue, output_4_norm_issue.transpose(0,1))
            l_bm25_neg = l_bm25_neg_fact+l_bm25_neg_issue         

            if hard_neg == True:
                logits = torch.cat([l_pos, l_neg, l_ran_neg, l_bm25_neg], dim=1).to(device)
            else:
                logits = torch.cat([l_pos, l_neg, l_ran_neg], dim=1).to(device)

            logits_label = torch.arange(0, len(labels)).type(torch.LongTensor).to(device)

            loss = loss_model(logits/temp, logits_label)

            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/Train', loss.item(), epoch)
            print('Loss/Train:', loss.item())
                
    else:
        ## Test
        model.eval()
        with torch.no_grad():
            test_label_list = []
            num = 0
            for batched_graph, labels in tqdm(dataloader):
                num += 1
                if num == 1:
                    sumfact_graph_batch = batched_graph
                    sumfact_graph_rep_matrix = model(sumfact_graph_batch.to(device), sumfact_graph_batch.ndata['w'].to(device), sumfact_graph_batch.edata['w'].to(device))
                    referissue_graph_list = []
                    for i in labels:
                        case_name = str(int(i)).zfill(6)
                        test_label_list.append(case_name)
                        referissue_graph = referissue_pool_dataset.graphs[case_name]
                        referissue_graph_list.append(referissue_graph)
                    referissue_graph_batch = dgl.batch(referissue_graph_list)
                    referissue_graph_rep_matrix = model(referissue_graph_batch.to(device), referissue_graph_batch.ndata['w'].to(device), referissue_graph_batch.edata['w'].to(device))
                else:
                    sumfact_graph_batch = batched_graph
                    sumfact_graph_rep = model(sumfact_graph_batch.to(device), sumfact_graph_batch.ndata['w'].to(device), sumfact_graph_batch.edata['w'].to(device))
                    sumfact_graph_rep_matrix = torch.cat((sumfact_graph_rep_matrix, sumfact_graph_rep), 0)          
                    referissue_graph_list = []
                    for i in labels:
                        case_name = str(int(i)).zfill(6)
                        test_label_list.append(case_name)
                        referissue_graph = referissue_pool_dataset.graphs[case_name]
                        referissue_graph_list.append(referissue_graph)
                    referissue_graph_batch = dgl.batch(referissue_graph_list)
                    referissue_graph_rep = model(referissue_graph_batch.to(device), referissue_graph_batch.ndata['w'].to(device), referissue_graph_batch.edata['w'].to(device))  
                    referissue_graph_rep_matrix = torch.cat((referissue_graph_rep_matrix, referissue_graph_rep), 0)  
            
            test_sumfact_graph_rep = sumfact_graph_rep_matrix
            test_referissue_graph_rep = referissue_graph_rep_matrix

            test_sumfact_graph_rep_norm = test_sumfact_graph_rep / test_sumfact_graph_rep.norm(dim=1)[:, None]
            test_referissue_graph_rep_norm = test_referissue_graph_rep / test_referissue_graph_rep.norm(dim=1)[:, None]

            test_sumfact_score = torch.mm(test_sumfact_graph_rep_norm, test_sumfact_graph_rep_norm.T)
            test_referissue_score = torch.mm(test_referissue_graph_rep_norm, test_referissue_graph_rep_norm.T)

            test_sim_score = test_sumfact_score+test_referissue_score
            test_sim_score.fill_diagonal_(float('-inf'))

            sim_score = []

            test_query_list = []
            for key, value in label_dict.items():
                test_query_list.append(key)
                query_index = test_label_list.index(key.split('.')[0])
                score = test_sim_score[query_index, :]
                sim_score.append(score)
            sim_score = torch.stack(sim_score)                        
            
            final_pre_dict = rank(sim_score, len(test_label_list), test_query_list, test_label_list)

            ##1stage
            correct_pred, retri_cases, relevant_cases, Micro_pre, Micro_recall, Micro_F, macro_pre, macro_recall, macro_F = metric(5, final_pre_dict, label_dict)
            yf_dict, correct_pred_yf, retri_cases_yf, relevant_cases_yf, Micro_pre_yf, Micro_recall_yf, Micro_F_yf, macro_pre_yf, macro_recall_yf, macro_F_yf = yf_metric(5, yf_path, final_pre_dict, label_dict)

            ndcg_score, mrr_score, map_score, p_score = t_metrics(label_dict, final_pre_dict, 5)
            ndcg_score_yf, mrr_score_yf, map_score_yf, p_score_yf = t_metrics(label_dict, yf_dict, 5)


            print("Correct Predictions: ", correct_pred)
            print("Retrived Cases: ", retri_cases)
            print("Relevant Cases: ", relevant_cases)

            print("Micro Precision: ", Micro_pre)
            print("Micro Recall: ", Micro_recall)
            print("Micro F1: ", Micro_F)

            print("Macro Precision: ", macro_pre)
            print("Macro Recall: ", macro_recall)
            print("Macro F1: ", macro_F)

            print("NDCG@5: ", ndcg_score)
            print("MRR@5: ", mrr_score)
            print("MAP: ", map_score)

            print("Correct Predictions yf: ", correct_pred_yf)
            print("Retrived Cases yf: ", retri_cases_yf)
            print("Relevant Cases yf: ", relevant_cases_yf)

            print("Micro Precision yf: ", Micro_pre_yf)
            print("Micro Recall yf: ", Micro_recall_yf)
            print("Micro F1 yf: ", Micro_F_yf)

            print("Macro Precision yf: ", macro_pre_yf)
            print("Macro Recall yf: ", macro_recall_yf)
            print("Macro F1 yf: ", macro_F_yf)

            print("NDCG@5 yf: ", ndcg_score_yf)
            print("MRR@5 yf: ", mrr_score_yf)
            print("MAP yf: ", map_score_yf)

    if embedding_saving:
        model.eval()
        with torch.no_grad():
            label_list = []
            num =0
            sumfact_graph_list = []
            for batched_graph, labels in tqdm(dataloader):
                num += 1
                sumfact_graph_batch = batched_graph
                sumfact_graph_list.append(sumfact_graph_batch)
                sumfact_graph = model(sumfact_graph_batch.to(device), sumfact_graph_batch.ndata['w'].to(device), sumfact_graph_batch.edata['w'].to(device))
                labels = [str(int(x)).zfill(6) for x in labels]
                label_list.append(labels)
                if num == 1:
                    sumfact_graph_rep = sumfact_graph
                else:
                    sumfact_graph_rep = torch.cat((sumfact_graph_rep, sumfact_graph), dim=0)
            
                referissue_graph_list = []
                for i in labels:
                    referissue_graph = referissue_pool_dataset.graphs[i]
                    referissue_graph_list.append(referissue_graph)
                referissue_graph_batch = dgl.batch(referissue_graph_list)
                referissue_graph = model(referissue_graph_batch.to(device), referissue_graph_batch.ndata['w'].to(device), referissue_graph_batch.edata['w'].to(device))   
                if num == 1:
                    referissue_graph_rep = referissue_graph
                else:
                    referissue_graph_rep = torch.cat((referissue_graph_rep, referissue_graph), dim=0)
            label_list = [y for x in label_list for y in x]
        
        if train_flag:
            dataset = 'train'
        else:
            dataset = 'test'
        
        case_embedding_matrix = torch.cat((sumfact_graph_rep, referissue_graph_rep), 1) 
        torch.save(case_embedding_matrix, os.getcwd()+'/Graph_generation/coliee'+data+'_'+dataset+'_casegnn_embedding.pt')

        with open(os.getcwd()+'/Graph_generation/coliee'+data+'_'+dataset+'_casegnn_embedding_case_name_list.json' , "w") as fOut:
            json.dump(label_list, fOut)
            fOut.close() 

    if train_flag == False:
        return ndcg_score_yf