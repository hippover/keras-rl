import numpy as np
import sys
import os
sys.path.insert(0, "/Users/Hippolyte/Desktop/Harvard/git_repo/keras-rl")

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from rl.callbacks import ModelPerformanceCheckpoint

from envs import *
import argparse

def train_with_params(sigma_v = 0., sigma_o = 0.,test=False):

    ENV_NAME = 'PongSolo'
    conf_name = '{}_sv_{}_so_{}'.format(ENV_NAME,sigma_v,sigma_o) # sv, so = sigma_v et sigma_orientation

    # Get the environment and extract the number of actions.
    env = EnvPongSolo(sigma_v = sigma_v, sigma_o = sigma_v)
    np.random.seed(123)

    #assert len(env.action_space.shape) == 1
    nb_actions = 1
    leaky_alpha = 0.2

    # Next, we build a very simple model.
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(100))
    actor.add(LeakyReLU(leaky_alpha))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = merge([action_input, flattened_observation], mode='concat')
    x = Dense(200)(x)
    x = LeakyReLU(leaky_alpha)(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(input=[action_input, observation_input], output=x)
    print(critic.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=1)
    n_steps = 5000000
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=1., mu=0., sigma=.3, sigma_min=0.01, n_steps_annealing=n_steps)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=.99, target_model_update=1e-3)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.

    directory_weights = "weights/ddpg/{}".format(conf_name)

    if not os.path.exists(directory_weights):
        os.makedirs(directory_weights)

    if test == False:
        perfCheckPoint = ModelPerformanceCheckpoint('{}/checkpoint_avg{}_steps{}'.format(directory_weights,'{}','{}'), 800)
        agent.fit(env, nb_steps=n_steps, visualize=False, verbose=2, nb_max_episode_steps=200,callbacks=[perfCheckPoint])

        # After training is done, we save the final weights.
        agent.save_weights('{}/final.h5f'.format(directory_weights), overwrite=True)

        # Finally, evaluate our algorithm for 5 episodes.
        agent.test(env, nb_episodes=100, visualize=False, nb_max_episode_steps=200)
    else:
        agent.load_weights('{}/final.h5f'.format(directory_weights))
        agent.test(env, nb_episodes=1000, visualize=False, nb_max_episode_steps=200)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma_v', type=float, default=0., help='percentage of precision on the norm of the speed')
    parser.add_argument('--sigma_o', type=float, default=0., help='percentage of precision on the norm of the angles')
    parser.add_argument('--test', action="store_true", help="If present, load weights and print the scores for 1000 episodes")

    args = parser.parse_args()

    train_with_params(**vars(args))
