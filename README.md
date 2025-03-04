# perc

다음과 같은 3단계로 구성됩니다.

1. Retrieve: MPNET 기반으로 문제별 top-10 Example을 선택한 json 파일 생성
2. Generate: ArchCode와 99%동일함. 단, chain에서 code, plan 생성 시 Retrieve 단계에서 선택한 문제별 Example을 3shot 사용함.
3. Evaluate: CodeContests, HumanEval은 ArchCode와 동일(CodeEval 사용), MultiPL-E는 bigcode eval 사용. docker 환경 구성 필요.)

아쉽게도, 위 전체 단계를 모두 한번에 실행하도록 자동화 하지는 않았고, 각 단계별 수동으로 실행하여 실험하였음.

## 단계별 실행 방법
1. Retrieve
  - Test 대상 문제들에 대해 perc 또는 baseline 방법들로 대상 pool에서 top-k example을 미리 선택해둔 파일을 생성.
  - 파일명, 검색 방법 변경등 전부 하드코딩으로 수정 및 변경해서 실험했음. 커멘드 옵션화 하지 않아서 이부분은 감안해주세요.
  ```
  $ python run.py retrieve run
  # src/pseudo_retriever.py의 run 함수에 multipl-e lua에 perc 방식 실행되도록 코딩되어있음.
  # 예제: ./data/he_retrieved_examples_mpnet_10_mbpp_py_by_ours_for_lua.json  파일 생성됨
  ```
  

2. Generate
  - ArchCode와 동일. 단, Retrieve 단계에서 생성한 파일을 config example에 설정
  - config/humaneval-lua-gpt35-rag.yaml 참고
  ```
  example:
  desc: "Example to use"
  value:
    code:
      dataset_type: "nuprl/MultiPL-E,mbpp-lua"
      path: "data/he_retrieved_examples_mpnet_10_mbpp_py_by_ours_for_lua.json"
    final_plan:
      dataset_type: "nuprl/MultiPL-E,mbpp-lua"
      path: "data/he_retrieved_examples_mpnet_10_mbpp_py_by_ours_for_lua.json"
    requirements:
      dataset_type: "nuprl/MultiPL-E,mbpp-lua"
      path: "data/he_retrieved_examples_mpnet_10_mbpp_py_by_ours_for_lua.json"
    draft_plan:
      dataset_type: "nuprl/MultiPL-E,mbpp-lua"
      path: "data/he_retrieved_examples_mpnet_10_mbpp_py_by_ours_for_lua.json"

  ```

  아래와 같이 실행
  ```
  $ python run.py main run --configs ./configs/humaneval-lua-gpt35-rag.yaml --run_name humaneval-lua-gpt35-rag-3shot-ours
  ```

3. Evaluate
  - CodeContests, HumanEval: ArchCode와 동일
    ```
    $ python run.py eval run --run_name <result-dir> --dataset_type openai_humaneval --timeout 6
    ```
  - MultiPL-E: bigcode-eval 세팅 필요 (https://github.com/bigcode-project/bigcode-evaluation-harness)
    
    - Generate 단계 결과를 bigcode-eval 입력 포멧으로 변경 필요
      ```
      $ python gen_generations.py -l <lang> -d <result-dir>
      # python gen_generations.py -l lua -d ./results/humaneval-lua-gpt35-rag-3shot-ours
      ```

    - docker로 실행
      ```
      $ docker run -v $(pwd)/logs:/app/logs -v $(pwd)/logs/tmp:/tmp -v $(pwd)/generations.json:/app/generations.json:ro -it evaluation-harness-multiple python3 main.py --tasks multiple-lua --n_samples 10 --load_generations_path /app/generations.json --allow_code_execution --temperature 0.8 --save_references --metric_output_path "/app/logs/metric.json”
      ```

