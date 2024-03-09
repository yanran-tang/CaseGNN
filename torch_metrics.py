import torch
from torchmetrics import RetrievalNormalizedDCG, RetrievalMRR, RetrievalMAP, RetrievalPrecision
import json


def rank(test_sim_score_matrix, ranking_num, test_query_list, test_label_list):
    test_sim_score = test_sim_score_matrix
    
    b = torch.topk(test_sim_score, ranking_num, dim=1)
    c = b[1].tolist()
    c_list = c
            
    final_pre_dict = {}
    for i in range(len(c_list)):
        pre_list = []
        que_name = test_query_list[i]
        for j in range(len(c_list[i])):
            predict_ind = c_list[i][j]
            case_name = test_label_list[predict_ind]+'.txt'
            pre_list.append(case_name)
        final_pre_dict.update({que_name: pre_list})
    
    return final_pre_dict

def t_metrics(label_dict, predict_dict, topk):
    topk = topk
    
    ## prediction preprocess
    pre_dic = predict_dict
    label_dict = label_dict

    index = -1
    index_list = []
    preds_list = []
    traget_list = []
    for key,value in pre_dic.items():
        index += 1
        rank = 1.0
        for v in value[:topk]:
            index_list.append(index)
            preds_list.append(rank)
            if v in label_dict[key]:
                traget_list.append(True)
            else:
                traget_list.append(False)
            rank -= 0.05

    ## mrr@5
    mrr_index_list = []
    mrr_preds_list = []
    mrr_traget_list = []
    for key,value in pre_dic.items():
        index += 1
        rank = 1.0
        for v in value[:topk]:
            mrr_index_list.append(index)
            mrr_preds_list.append(rank)
            if v in label_dict[key]:
                mrr_traget_list.append(True)
            else:
                mrr_traget_list.append(False)
            rank -= 0.05

    ndcg = RetrievalNormalizedDCG(top_k=topk)

    mrr = RetrievalMRR()
    map = RetrievalMAP()
    p = RetrievalPrecision(k=topk)
    ndcg_score = ndcg(torch.tensor(preds_list), torch.tensor(traget_list), indexes=torch.tensor(index_list))
    mrr_score = mrr(torch.tensor(mrr_preds_list), torch.tensor(mrr_traget_list), indexes=torch.tensor(mrr_index_list)) ##mrr@5
    map_score = map(torch.tensor(preds_list), torch.tensor(traget_list), indexes=torch.tensor(index_list)) 
    p_score = p(torch.tensor(preds_list), torch.tensor(traget_list), indexes=torch.tensor(index_list))

    return ndcg_score, mrr_score, map_score, p_score

#Micro Precision Function
def micro_prec(true_list,pred_list,k):
    #define list of top k predictions
    cor_pred = 0
    top_k_pred = pred_list[0:k].copy()
    #iterate throught the top k predictions
    for doc in top_k_pred:
        #if document in true list, then increment count of relevant predictions
        if doc in true_list:
            cor_pred += 1
    #return total_relevant_predictions_in_top_k/k
    return cor_pred, k 

def micro_prec_datefilter(can_list, pred_list):
    yearfilter_can_list = []
    for i in pred_list:
        if i in can_list:
            yearfilter_can_list.append(i)
    return yearfilter_can_list

def metric(topk, final_pre_dict, label_dict):   
    correct_pred = 0
    retri_cases = 0
    relevant_cases = 0
    cls_pre = 0
    cls_recall = 0

    for i in final_pre_dict.keys():
        query_case = i
        true_list = label_dict[i]
        r = topk
        pred_list = final_pre_dict[i]

        ##w/o year filter
        c_p, r_c = micro_prec(true_list, pred_list, r)
        correct_pred += c_p
        retri_cases += r_c
        relevant_cases += len(true_list)
        ## macro precision
        if c_p > 0:
            cls_pre += c_p/topk
            cls_recall += c_p/len(true_list)
        else:
            cls_pre += 0
            cls_recall += 0

    Micro_pre = correct_pred/retri_cases
    Micro_recall = correct_pred/relevant_cases
    Micro_F = 2*Micro_pre*Micro_recall/ (Micro_pre + Micro_recall)

    macro_pre = cls_pre/len(final_pre_dict.keys())
    macro_recall = cls_recall/len(final_pre_dict.keys())
    macro_F = 2*macro_pre*macro_recall/ (macro_pre + macro_recall)

    return correct_pred, retri_cases, relevant_cases, Micro_pre, Micro_recall, Micro_F, macro_pre, macro_recall, macro_F

def yf_metric(topk, yf_path, final_pre_dict, label_dict ):   
    with open(yf_path, 'r') as f:
        yearfilter_can_list = json.load(f)
        f.close()

    correct_pred_yf = 0
    retri_cases_yf = 0
    relevant_cases_yf = 0
    cls_pre_yf = 0
    cls_recall_yf = 0
    yf_dict = {}
    for i in final_pre_dict.keys():
        query_case = i
        true_list = label_dict[i]
        r = topk
        pred_list = final_pre_dict[i]

        ##w year filter    
        yearfilter_pred_list = micro_prec_datefilter(yearfilter_can_list[query_case], pred_list)
        c_p_yf, r_c_yf = micro_prec(true_list, yearfilter_pred_list, r)
        correct_pred_yf += c_p_yf
        retri_cases_yf += r_c_yf
        relevant_cases_yf += len(true_list)
        ## macro precision
        if c_p_yf > 0:
            cls_pre_yf += c_p_yf/topk
            cls_recall_yf += c_p_yf/len(true_list)
        else:
            cls_pre_yf += 0
            cls_recall_yf += 0
        yf_dict.update({query_case:yearfilter_pred_list})

    ## w yearfilter
    Micro_pre_yf = correct_pred_yf/retri_cases_yf
    Micro_recall_yf = correct_pred_yf/relevant_cases_yf
    Micro_F_yf = 2*Micro_pre_yf*Micro_recall_yf/ (Micro_pre_yf + Micro_recall_yf)

    macro_pre_yf = cls_pre_yf/len(final_pre_dict.keys())
    macro_recall_yf = cls_recall_yf/len(final_pre_dict.keys())
    macro_F_yf = 2*macro_pre_yf*macro_recall_yf/ (macro_pre_yf + macro_recall_yf)

    return yf_dict, correct_pred_yf, retri_cases_yf, relevant_cases_yf, Micro_pre_yf, Micro_recall_yf, Micro_F_yf, macro_pre_yf, macro_recall_yf, macro_F_yf