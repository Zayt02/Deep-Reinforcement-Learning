# DDPG with Batch Normalization and Ornstein Uhlenbeck Action Noise - Keras Implementation (Tensorflow 2.1)
- Work with 1D action space and 1D state space, but can easily modified to adapt with 2-3D space
- Test with Pendulum-v0, whole model save at "save_Pendulumv0/whole_model"
- Model starts converge after 100 episodes. Watch it perform after 400 episodes in "main.py".
- After 500 episodes the score kept continue decreasing, but the model acts a little bit weird. So I stopped training it to avoid (maybe??) diverging.
