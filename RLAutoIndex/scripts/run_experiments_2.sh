#!/bin/bash

# run within something like a tmux session in background

# run from rlautoindex/root
if [ "$(basename $(pwd))" != "RLAutoIndex" ]
then 
    echo "start script from RLAutoIndex root" ;
    exit ;
fi

# base directory for workload, workload results with timestamp

RESULT_DIR=../res/ 
mkdir ${RESULT_DIR}
RESULT_DIR=${RESULT_DIR}$(date '+%m-%d-%y_%H:%M')
mkdir ${RESULT_DIR}


SEED=17;

# config
mkdir ${RESULT_DIR}/conf
cp conf/{dqn2,spg2,experiment2}.json ${RESULT_DIR}/conf

#
# build workload, run default
# 
echo "#### RUNNING DEFAULT ON WORKLOAD ####"
time python3 src/common/postgres_controller.py \
--dqn=True \
--config=conf/dqn2.json \
--experiment_config=conf/experiment2.json \
--result_dir=${RESULT_DIR} \
--generate_workload=True \
--with_agent=False \
--default_baseline=True \
--seed=${SEED};
#
# run full
#
echo "#### RUNNING FULL ON WORKLOAD ####" ;
time python3 src/common/postgres_controller.py --dqn=True --config=conf/dqn2.json --experiment_config=conf/experiment2.json --result_dir=${RESULT_DIR} --generate_workload=False --with_agent=False --default_baseline=False --seed=${SEED};
#
# run dqn
#
echo "#### RUNNING DQN ON WORKLOAD ####" ;
time python3 src/common/postgres_controller.py --dqn=True --config=conf/dqn2.json --experiment_config=conf/experiment2.json --result_dir=${RESULT_DIR} --generate_workload=False --seed=${SEED} &> ${RESULT_DIR}/dqn.log ;
mv ${RESULT_DIR}/dqn.log ${RESULT_DIR}/dqn ;
#
# run spg
#
echo "#### RUNNING SPG ON WORKLOAD ####" ;
time python3 src/common/postgres_controller.py --dqn=False --config=conf/spg2.json --experiment_config=conf/experiment2.json --result_dir=${RESULT_DIR} --generate_workload=False --seed=${SEED} &> ${RESULT_DIR}/spg.log ;
mv ${RESULT_DIR}/spg.log ${RESULT_DIR}/spg ;
#
# run tuner
#
echo "#### RUNNING TUNER ON WORKLOAD ####" ;
# n.b. python2 not python3
time python src/baseline/postgres_tuner2.py --experiment_config=conf/experiment2.json --data_dir=${RESULT_DIR} &> ${RESULT_DIR}/tuner.log ;
mv ${RESULT_DIR}/tuner.log ${RESULT_DIR}/tuner ;