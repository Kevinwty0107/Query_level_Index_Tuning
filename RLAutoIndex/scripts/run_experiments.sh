#!/bin/bash

# run within something like a tmux session in background

# run from rlautoindex/root
if [ "$(basename $(pwd))" != "RLAutoIndex" ]
then 
    echo "start script from RLAutoIndex root" ;
    exit ;
fi

# base directory for workload, workload results with timestamp

if [ ! -n $1 ]
then
    echo "#### You have to specify the benchmark you want to use #### ";
    exit;
fi




if [ $1 = "tpch" ]
then 
    RESULT_DIR=../res/ 
    mkdir ${RESULT_DIR}
    prefix='tpch'
    RESULT_DIR=${RESULT_DIR}${prefix}_$(date '+%m-%d-%y_%H:%M')
    mkdir ${RESULT_DIR}


    SEED=17;

    # config
    mkdir ${RESULT_DIR}/conf
    cp conf/{dqn,spg,experiment}.json ${RESULT_DIR}/conf

    #
    # build workload, run default
    #    
    echo "#### RUNNING DEFAULT ON WORKLOAD ####"
    time python3 src/common/postgres_controller.py \
    --dqn=True \
    --config=conf/dqn.json \
    --experiment_config=conf/experiment.json \
    --result_dir=${RESULT_DIR} \
    --generate_workload=True \
    --with_agent=False \
    --default_baseline=True \
    --seed=${SEED};
    #
    # run full
    #
    echo "#### RUNNING FULL ON WORKLOAD ####" ;
    time python3 src/common/postgres_controller.py --dqn=True --config=conf/dqn.json --experiment_config=conf/experiment.json --result_dir=${RESULT_DIR} --generate_workload=False --with_agent=False --default_baseline=False --seed=${SEED};
    #
    # run dqn
    #
    echo "#### RUNNING DQN ON WORKLOAD ####" ;
    time python3 src/common/postgres_controller.py --dqn=True --config=conf/dqn.json --experiment_config=conf/experiment1.json --result_dir=${RESULT_DIR} --generate_workload=False --seed=${SEED} &> ${RESULT_DIR}/dqn.log ;
    mv ${RESULT_DIR}/dqn.log ${RESULT_DIR}/dqn ;
    #
    # run spg
    #
    echo "#### RUNNING SPG ON WORKLOAD ####" ;
    time python3 src/common/postgres_controller.py --dqn=False --config=conf/spg.json --experiment_config=conf/experiment1.json --result_dir=${RESULT_DIR} --generate_workload=False --seed=${SEED} &> ${RESULT_DIR}/spg.log ;
    mv ${RESULT_DIR}/spg.log ${RESULT_DIR}/spg ;
    #
    # run tuner
    #
    echo "#### RUNNING TUNER ON WORKLOAD ####" ;
    time python src/baseline/postgres_tuner.py --experiment_config=conf/experiment1.json --data_dir=${RESULT_DIR} &> ${RESULT_DIR}/tuner.log ;
    mv ${RESULT_DIR}/tuner.log ${RESULT_DIR}/tuner ;

elif [ $1 = "imdb" ]
then 
    RESULT_DIR=../res/ 
    mkdir ${RESULT_DIR}
    prefix='imdb'
    RESULT_DIR=${RESULT_DIR}${prefix}_$(date '+%m-%d-%y_%H:%M')
    mkdir ${RESULT_DIR}


    SEED=17;

    # config
    mkdir ${RESULT_DIR}/conf
    cp conf/{dqn,spg,experiment}.json ${RESULT_DIR}/conf

    #
    # build workload, run default
    #    
    echo "#### RUNNING DEFAULT ON WORKLOAD ####"
    time python3 src/imdb_common/postgres_controller.py \
    --dqn=True \
    --config=conf/dqn.json \
    --experiment_config=conf/experiment.json \
    --result_dir=${RESULT_DIR} \
    --generate_workload=True \
    --with_agent=False \
    --default_baseline=True \
    --seed=${SEED};
    #
    # run full
    #
    echo "#### RUNNING FULL ON WORKLOAD ####" ;
    time python3 src/imdb_common/postgres_controller.py --dqn=True --config=conf/dqn.json --experiment_config=conf/experiment.json --result_dir=${RESULT_DIR} --generate_workload=False --with_agent=False --default_baseline=False --seed=${SEED};
    #
    # run dqn
    #
    echo "#### RUNNING DQN ON WORKLOAD ####" ;
    time python3 src/imdb_common/postgres_controller.py --dqn=True --config=conf/dqn.json --experiment_config=conf/experiment.json --result_dir=${RESULT_DIR} --generate_workload=False --seed=${SEED} &> ${RESULT_DIR}/dqn.log ;
    mv ${RESULT_DIR}/dqn.log ${RESULT_DIR}/dqn ;
    #
    # run spg
    #
    echo "#### RUNNING SPG ON WORKLOAD ####" ;
    time python3 src/imdb_common/postgres_controller.py --dqn=False --config=conf/spg.json --experiment_config=conf/experiment.json --result_dir=${RESULT_DIR} --generate_workload=False --seed=${SEED} &> ${RESULT_DIR}/spg.log ;
    mv ${RESULT_DIR}/spg.log ${RESULT_DIR}/spg ;
    #
    # run tuner
    #
    echo "#### RUNNING TUNER ON WORKLOAD ####" ;
    time python src/baseline/postgres_imdb_tuner.py --experiment_config=conf/experiment.json --data_dir=${RESULT_DIR} &> ${RESULT_DIR}/tuner.log ;
    mv ${RESULT_DIR}/tuner.log ${RESULT_DIR}/tuner ;
else

    echo "#### You have to specify the valid benchmark you want to use #### ";
    exit;

fi