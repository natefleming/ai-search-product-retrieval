#!/bin/zsh
# usage: run_notebook.sh <notebook_stem> [base_parameters_json]  -> submits on serverless (env 5 via notebook metadata), waits, prints result/error
NB=$1; PARAMS=$2; [[ -z "$PARAMS" ]] && PARAMS="{}"; RN=${NB//\//-}
P=${RETRIEVAL_PROFILE:-DEFAULT}  # Databricks CLI profile of the target workspace
field() { python3 -c "import sys, json; d = json.loads(sys.stdin.read(), strict=False); print(eval(sys.argv[1], {'d': d, 'json': json}))" "$1"; }
U=/Users/$(databricks -p $P current-user me -o json | field "d['userName']")
R=$(databricks -p $P jobs submit --no-wait --json "{\"run_name\":\"retrieval-$RN\",\"timeout_seconds\":14400,\"tasks\":[{\"task_key\":\"t\",\"notebook_task\":{\"notebook_path\":\"$U/ai-search-product-retrieval/$NB\",\"base_parameters\":$PARAMS}}]}" | field "d['run_id']")
echo "run_id=$R"
until s=$(databricks -p $P jobs get-run $R -o json | field "d['state']['life_cycle_state']"); [[ "$s" == TERMINATED || "$s" == INTERNAL_ERROR || "$s" == SKIPPED ]]; do sleep 20; done
until J=$(databricks -p $P jobs get-run $R -o json 2>/dev/null) && [[ -n "$J" ]]; do sleep 30; done  # survive network blips
printf "%s" "$J" | field "json.dumps({'result': d['state'].get('result_state'), 'ms': d.get('execution_duration'), 'url': d.get('run_page_url')})"
T=$(printf "%s" "$J" | field "d['tasks'][0]['run_id']")
databricks -p $P jobs get-run-output $T -o json | field "(d.get('notebook_output') or {}).get('result') or (d.get('error', '') + chr(10) + (d.get('error_trace') or '')[-3000:])"
