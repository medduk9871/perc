import json
import multiprocessing
import os
from math import comb
from pathlib import Path
from typing import List

from evaluate import load
from pydantic import BaseModel
from tqdm import tqdm

from datasets import load_dataset
import json

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


doc_names = []
doc_prompts = []
doc_source = []
doc_cpp = []
doc_java = []
doc_plan = []
documents = []

# with open('./data/rag_pseudo/codecontests-train-cpp-java.json', 'r') as f_cpp_java, open('./data/rag_pseudo/codecontests-train-py.json', 'r') as f_py:
#     data_py = json.load(f_py)
#     data_cpp_java = json.load(f_cpp_java)
    
#     py_probs = set()
#     cpp_probs = set()
    
#     for item in data_py:
#         if not item['code'] or item['code'][0] == "":
#             continue
        
#         py_probs.add(item['id'])
        
#     for item in data_cpp_java:
#         if not item['code']:
#             continue
        
#         for idx, code in enumerate(item['code']):
#             if 'java.' in code or 'public class' in code:
#                 continue
#             if '#include' in code:
#                 cpp_probs.add(item['id'])

#     probs = list(py_probs & cpp_probs)

    # for item in data_py:
    #     if not item['code'] or item['code'][0] == "":
    #         continue
    #     if item['id'] not in probs:
    #         continue
        
    #     for idx, code in enumerate(item['code']):
    #         doc_names.append(item['id'])
    #         doc_prompts.append(item['prompt'])
    #         doc_cpp.append(item['code'][idx])
    #         doc_source.append(item['code'][idx])
    #         # doc_java.append(item['requirements'])
    #         doc_plan.append(item['draft_plan'][idx])
    #         documents.append(item['prompt'] + "\nLet's think step by step.\n" + item['draft_plan'][idx])
    #         # documents.append(item['code'][idx])
    #         # documents.append(item['prompt'])
    #         # documents.append(item['prompt'] + item['code'][idx])


    # for item in data_cpp_java:
    #     if not item['code']:
    #         continue
        
    #     if item['id'] not in probs:
    #         continue

    #     for idx, code in enumerate(item['code']):
    #         if 'java.' in code or 'public class' in code:
    #             continue
    #         # if '#include' in code:
    #         #     continue
    #         doc_names.append(item['id'])
    #         doc_prompts.append(item['prompt'])
    #         doc_cpp.append(item['final_plan'][idx])
    #         doc_source.append(item['code'][idx])
    #         # doc_java.append(item['requirements'])
    #         doc_plan.append(item['draft_plan'][idx])
    #         documents.append(item['prompt'] + "\nLet's think step by step.\n" + item['draft_plan'][idx])
    #         # documents.append(item['prompt'] + item['code'][idx])
            


# with open('./data/rag_pseudo/mbpp_pseudo_rb_new_plan.json', 'r') as f:
#     data = json.load(f)

#     for item in data:
#         if not item['code']:
#             continue
        
#         for idx, code in enumerate(item['code']):
#             doc_names.append(item['id'])
#             doc_prompts.append(item['prompt'])
#             # doc_cpp.append(item['final_plan'][idx])
#             doc_source.append(item['final_plan'][idx])
#             # doc_java.append(item['requirements'])
#             doc_plan.append(item['draft_plan'][idx])
#             # documents.append(item['prompt'] + "\nLet's think step by step.\n" + item['draft_plan'][idx])
#             documents.append(item['prompt'] + item['code'][idx])
#             # documents.append(item['prompt'])

with open('./data/rag_pseudo/mbpp_pseudo_lua.json', 'r') as f_py:
    data_py = json.load(f_py)

    for item in data_py:
        if not item['code']:
            continue
        
        for idx, code in enumerate(item['code']):
            doc_names.append(item['id'])
            doc_prompts.append(item['prompt'])
            # doc_cpp.append(item['final_plan'][idx])
            doc_cpp.append(item['code'][idx])
            doc_source.append(item['final_plan'][idx])
            # doc_java.append(item['requirements'])
            doc_plan.append(item['draft_plan'][idx])
            documents.append(item['prompt'] + "\nLet's think step by step.\n" + item['draft_plan'][idx])
            # documents.append(item['prompt'] + item['code'][idx])
            # documents.append(item['code'][idx])
            # documents.append(item['prompt'])


# 각 문서를 벡터로 변환
document_embeddings = model.encode(documents, convert_to_tensor=True)
source_embeddings = model.encode(doc_source, convert_to_tensor=True)
# cpp_java_embeddings = model.encode(doc_cpp, convert_to_tensor=True)

print(len(doc_cpp), len(doc_source), len(documents))
def dense_retrieval(query_embedding, top_k=3, descending=True, document_embeddings=document_embeddings):
    # 코사인 유사도 계산
    cosine_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]

    # 상위 K개의 문서 인덱스 추출
    top_results = cosine_scores.argsort(descending=descending)[:top_k]

    top_3_examples = []

    # 상위 K개의 문서 및 점수 출력
    for idx in top_results:
        # print(f"Document: {documents[idx]}\nSimilarity Score: {cosine_scores[idx]:.4f}\n")
        # print(f"Doc Name: {doc_names[idx]}\nSource: {doc_source[idx]}\n")

        top_3_examples.append({
            "id": doc_names[idx],
            "prompt": doc_prompts[idx],
            "draft_plan": doc_plan[idx],
            "requirements": "",
            "final_plan": doc_cpp[idx],
            "code": doc_source[idx],
            # "code": doc_source[idx],
            "gen_tc": '',
            "sim_score": f'{cosine_scores[idx]:.4f}'
        })
    return top_3_examples


from rank_bm25 import BM25Okapi

# BM25 모델 초기화
bm25 = BM25Okapi([doc.split() for doc in documents])

def bm25_keyword_search(query, top_k=3):
    bm25_scores = bm25.get_scores(query.split())
    # 상위 K개의 문서 인덱스 추출
    top_results = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]

    top_3_examples = []

    # 상위 K개의 문서 및 점수 출력
    for idx in top_results:
        # print(f"Document: {documents[idx]}\nSimilarity Score: {cosine_scores[idx]:.4f}\n")
        # print(f"Doc Name: {doc_names[idx]}\nSource: {doc_source[idx]}\n")

        top_3_examples.append({
            "id": doc_names[idx],
            "prompt": doc_prompts[idx],
            "draft_plan": doc_plan[idx],
            "requirements": '',
            "final_plan": doc_cpp[idx],
            "code": doc_source[idx],
            "gen_tc": '',
            "sim_score": f'{bm25_scores[idx]:.4f}'
        })

    return top_3_examples


class PseudoRetriever(BaseModel):
    def __init__(
        self,
        **data,
    ):
        super().__init__(**data)

    def test(self):
        print('hello')

    def run(self):
        retrieved_data = {}
        
        with open('data/rag_pseudo/humaneval-test-pseudo.json', 'r') as f:
            
            data = json.load(f)

            for prob in tqdm(data):
                query = prob['prompt'] + "\nLet's think step by step.\n" + prob['draft_plan'][0]

                # 쿼리를 벡터로 변환
                query_embedding = model.encode(query, convert_to_tensor=True)
                # Dense Retrieval 수행
                retrieved_data[prob['id']] = dense_retrieval(query_embedding, top_k=10, descending=True)


        with open('he_retrieved_examples_mpnet_10_mbpp_py_by_ours_for_lua.json', 'w') as f:
            json.dump(retrieved_data, f, indent=4)

        # BM25 키워드 검색 (top 3) 수행
#       bm25_keyword_search(documents, query, top_k=3)

    def run_bm(self):
        retrieved_data = {}
        
        with open('data/rag_pseudo/humaneval-test-pseudo.json', 'r') as f:
            
            data = json.load(f)

            for prob in tqdm(data):
                query = prob['prompt']# + "\nLet's think step by step.\n" + prob['draft_plan'][0]

                # Dense Retrieval 수행
                retrieved_data[prob['id']] = bm25_keyword_search(query, top_k=10)


        with open('he_retrieved_examples_bm25_10_mbpp_py_for_rb_prob.json', 'w') as f:
            json.dump(retrieved_data, f, indent=4)

    def run_bm_repocoder(self):
        retrieved_data = {}
        
        with open('data/rag_pseudo/humaneval-rb-test-code.json', 'r') as f:
            
            data = json.load(f)

            for prob in tqdm(data):
                # query = prob['prompt'] + prob['code'][0]
                query = prob['code'][0]

                # Dense Retrieval 수행
                retrieved_data[prob['id']] = bm25_keyword_search(query, top_k=10)


        with open('he_retrieved_examples_bm25_10_mbpp_py_for_rb_code.json', 'w') as f:
            json.dump(retrieved_data, f, indent=4)

    def run_cc(self):
        retrieved_data = {}
        
        with open('data/rag_pseudo/codecontests-test-pseudo.json', 'r') as f:
            
            data = json.load(f)

            for prob in tqdm(data):
                query = prob['prompt'] + "\nLet's think step by step.\n" + prob['draft_plan'][0]
                # query = prob['prompt']
                # 쿼리를 벡터로 변환
                query_embedding = model.encode(query, convert_to_tensor=True)
                # Dense Retrieval 수행
                retrieved_data[prob['id']] = dense_retrieval(query_embedding, top_k=10, descending=True)


        with open('cc_retrieved_examples_mpnet_10_cpp_by_ours_for_rebuttal.json', 'w') as f:
            json.dump(retrieved_data, f, indent=4)

    def run_cc_with_code(self):
        retrieved_data = {}
        
        with open('data/rag_pseudo/codecontests-test-code.json', 'r') as f:
            
            data = json.load(f)

            for prob in tqdm(data): 
                # query = prob['prompt'] + prob['code'][0]
                # query = prob['code'][0]
                query = prob['prompt']

                # 쿼리를 벡터로 변환
                query_embedding = model.encode(query, convert_to_tensor=True)
                # Dense Retrieval 수행
                retrieved_data[prob['id']] = dense_retrieval(query_embedding, top_k=10, descending=True)


        with open('cc_retrieved_examples_mpnet_10_py_by_prob.json', 'w') as f:
            json.dump(retrieved_data, f, indent=4)

    def run_with_code(self):
        retrieved_data = {}
        
        with open('data/rag_pseudo/humaneval-rs-test-code.json', 'r') as f:
            
            data = json.load(f)

            for prob in tqdm(data):
                query = prob['code'][0]

                # 쿼리를 벡터로 변환
                query_embedding = model.encode(query, convert_to_tensor=True)
                # Dense Retrieval 수행
                retrieved_data[prob['id']] = dense_retrieval(query_embedding, top_k=10, descending=True)


        with open('he_retrieved_examples_mpnet_10_mbpp_py_by_code_rs.json', 'w') as f:
            json.dump(retrieved_data, f, indent=4)

    def run2(self):
        retrieved_data = {}
        
        with open('cc_retrieved_examples_mpnet_10.json', 'r') as f:
            data = json.load(f)
            
            for name in tqdm(data):
                cur_data = data[name]
                cur_top1 = cur_data[0]
                cur_top1['score_diff_with_prev'] = '0'
                
                new_list = [cur_top1]
                
                for item in cur_data:
                    if float(cur_top1['sim_score']) - float(item['sim_score']) > 0.2:
                        continue
                    
                    topK_prompt_embedding = model.encode(new_list[-1]['prompt'], convert_to_tensor=True)
                    cur_prompt_embedding = model.encode(item['prompt'], convert_to_tensor=True)
                    cosine_score = util.pytorch_cos_sim(topK_prompt_embedding, cur_prompt_embedding)
                    cur_sim_score = float(f'{cosine_score[0][0]:.4f}')

                    if cur_sim_score >= 0.7:
                        continue
                    
                    item['sim_score_with_prev'] = str(cur_sim_score)
                    new_list.append(item)
                
                data[name] = new_list

        with open('cc_retrieved_examples_mpnet_new2_10.json', 'w') as f:
            json.dump(data, f, indent=4)
        
    def run3(self):
        retrieved_data = {}
        
        with open('cc_retrieved_examples_bm25_10.json', 'r') as f:
            data = json.load(f)
            
            for name in tqdm(data):
                cur_data = data[name]
                cur_top1 = cur_data[0]
                cur_top1['score_diff_with_prev'] = '0'
                
                new_list = [cur_top1]
                
                for item in cur_data:
                    if float(cur_top1['sim_score']) - float(item['sim_score']) > 50:
                        continue
                    
                    
                    token1 = new_list[-1]['prompt'].split()
                    
                    token2 = item['prompt'].split()

                    bm25 = BM25Okapi([token1])

                    bm25_scores = bm25.get_scores(token2)
                    print(f'{bm25_scores[0]:.4f}')
                    # cur_sim_score = float(f'{bm25_scores[0][0]:.4f}')
                    
                    # score_diff = float(cur_top1['sim_score']) - cur_sim_score

                    # if score_diff < 0.1:
                    #     continue
                    
                    # item['score_diff_with_prev'] = str(score_diff)
                    # new_list.append(item)
                
                exit()
                data[name] = new_list

        with open('cc_retrieved_examples_mpnet_new_10.json', 'w') as f:
            json.dump(data, f, indent=4)

        # for prob in tqdm(dataset_sanitized['test']):
        #     query = prob['description']

        #     # 쿼리를 벡터로 변환
        #     query_embedding = model.encode(query, convert_to_tensor=True)
        #     # Dense Retrieval 수행
        #     retrieved_data[prob['name']] = dense_retrieval(query_embedding, top_k=3)

        #     # 쿼리에 대한 BM25 점수 계산
        #     #retrieved_data[prob['name']] = bm25_keyword_search(query, top_k=10)
        #     break


        # with open('cc_retrieved_examples_mpnet_10.json', 'w') as f:
        #     json.dump(retrieved_data, f, indent=4)