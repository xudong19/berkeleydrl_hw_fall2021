python cs285/scripts/run_hw5_expl.py --env_name *Chosen Env 1* --use_rnd \
--unsupervised_exploration --exp_name q1_env1_rnd


python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=10 \
--exp_name q5_awac_easy_supervised_lam10