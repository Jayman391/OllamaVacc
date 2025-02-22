#!/bin/bash

echo "Starting ollama server"
ollama serve &
sleep 2

echo "Pulling llama3.2 model"
ollama pull llama3.2

# input and output files
labelfile=labels/$SLURM_JOBID.json
reqfile=requests_newprompt.json

echo "labelfile: $labelfile"
echo "reqfile:   $reqfile"

# Initialize $i and make sure $num_requests is defined
i=1
num_requests=$(wc -l < $reqfile)

echo "Number of requests to process: $num_requests"


while [ $i -le $num_requests ] ; do
    req="$(head -$i $reqfile | tail -1)"
    curl http://localhost:11434/api/chat -d "${req}" | jq '.message | .content ' >> $labelfile
    i=$((i+1))
done

kill %1
