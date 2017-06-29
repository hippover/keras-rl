#coding=utf-8

import numpy as np
import sys
sys.path.insert(0, "/Users/Hippolyte/Desktop/Harvard/git_repo/keras-rl")
# don't know how to keep the same folder architecture without having to do this...
import json
#import matplotlib.pyplot as plt
import argparse
import os
import matplotlib.pyplot as plt

import keras
import keras.regularizers as regularizers
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.initializers import RandomUniform

from rl.callbacks import FileLogger, Callback
from rl.core import Processor
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from envs import *
'''from Screen2D_2players import *
from ScreenPong import *'''

def plotStrategy(env):

    actions = np.zeros(300)
    p_left = np.linspace(-0.5,0.5,300)
    state = np.zeros((300,9))
    for i in range(300):
        state[i,0] = p_left[i]
        state[i,1] = 0.
        state[i,6] = -0.2
        state[i,7] = 0.0
        state[i,4] = -0.5
        state[i,5] = 0.
        actions[i] = env.player_left.agent.forward(state[i,:])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(actions)
    plt.show()

def makeEnv(player1,player2, ENV_NAME, wall_reward=0., touch_reward = 0.,ball_speed=1.0):

    env = None
    if ENV_NAME == 'Env2D':
        env = Game2D(agent,wall_reward=wall_reward, touch_reward=touch_reward)
    elif ENV_NAME == 'EnvPong':
        env = Pong(player1,player2,wall_reward=wall_reward, touch_reward=touch_reward, ball_speed=ball_speed)
    return env

def playGames(agent1,agent2,myopie1,myopie2,ENV_NAME,n_games,log_interval = 100, wall_reward = 0., touch_reward = 0.):

    env = makeEnv(ENV_NAME, agent2)
    results = np.zeros((n_games,2))
    for i in range(n_games):
        n_steps = 0
        state = env.reset()
        if i % log_interval == 0:
            print "Game n {}".format(i)
        while (env.over == False):
            state, reward, _, _a = env.step(agent1.forward(state))
            n_steps += 1
        results[i,0] = reward
        results[i,1] = n_steps
    return results

def compareMyopie(env,myopie_max,n_games=500,n_splits=20):

    average = np.zeros((n_splits,2))
    print "Investigating the effects of myopia. \tMax = {}\t{} splits\t{} games per split".format(myopie_max, n_splits, n_games)
    myopies = np.linspace(-myopie_max,myopie_max,n_splits)
    for i in range(n_splits):
        print "starting split {}...".format(i)
        env.player_right.myopie = myopies[i]
        results = np.zeros((n_games,2))
        for j in range(n_games):
            n_steps = 0
            state = env.reset()
            while (env.over == False and n_steps < 10000):
                state, reward, _, _a = env.step(env.player_left.agent.forward(state))
                n_steps += 1
            results[j,0] = reward
            results[j,1] = n_steps
        average[i,0] = np.mean(results[:,0])
        average[i,1] = np.mean(results[:,1])
    return average

def confrontPlayers(env):

    for i in range(10):
        playPong(player_left=env.player_left, player_right=env.player_right)

    max_myopie = 0.03
    average = compareMyopie(env, max_myopie,n_games=500,n_splits=20)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.linspace(-max_myopie,max_myopie,average.shape[0]),average[:,0])
    ax.set_xlabel("Myopie")
    ax.set_ylabel("Score Moyen")
    ax.set_title("Score moyen en fonction de la myopie de l'adversaire")
    plt.show()
    '''results = playGames(agent1,agent2,myopie1,myopie2,ENV_NAME,100)
    print "100 parties jouées avec player 1 à gauche, reward_left moyenne : {}\tn_steps moyens : {}\tstd n_steps : {}".format(np.mean(results[:,0]), np.mean(results[:,1]), np.std(results[:,1]))
    results = playGames(agent2,agent1,myopie1,myopie2,ENV_NAME,100)
    print "100 parties jouées avec player 2 à gauche, reward_left moyenne : {}\tn_steps moyens : {}\tstd n_steps : {}".format(np.mean(results[:,0]), np.mean(results[:,1]), np.std(results[:,1]))
    results = playGames(agent1,agent1,myopie1,myopie2,ENV_NAME,100)
    print "100 parties jouées avec player 1 contre lui-même, reward_left moyenne : {}\tn_steps moyens : {}\tstd n_steps : {}".format(np.mean(results[:,0]), np.mean(results[:,1]), np.std(results[:,1]))
    results = playGames(agent2,agent2,myopie1,myopie2,ENV_NAME,100)
    print "100 parties jouées avec player 2 contre lui-même, reward_left moyenne : {}\tn_steps moyens : {}\tstd n_steps : {}".format(np.mean(results[:,0]), np.mean(results[:,1]), np.std(results[:,1]))
    '''

def main(layers1=[200],layers2=[200], leaky_alpha=0.10,ENV_NAME='EnvPong',show=False,wall_reward=-0.1, touch_reward=0.3,n_steps=80000,n_alternances=10,L_R = 0.0001, only_test=False, opp_aware = [1,1], myopie=[0.00,0.00], ball_speed=1.0,weights1_name='',weights2_name=''):

    ENV_NAME = ENV_NAME

    conf_name = "{}_layers1={}__layers2={}__leaky={}__lr={}__opp={}__myopia={}__speed={}".format(ENV_NAME, layers1,layers2,leaky_alpha,L_R, opp_aware,myopie,ball_speed)
    #gym.undo_logger_setup()
    # Get the environment and extract the number of actions.

    if ENV_NAME =='Env2D':
        env = Game2D(2.)
    elif ENV_NAME =='Env2DSoloSpin':
        env = Game2DSolo(2., spinRacket=True)
    elif ENV_NAME == 'Env3DSolo':
        env = Game3DSolo(2., 9.8,0.5, 7.,3.)
    elif ENV_NAME == 'EnvPong':
        env = Pong(PongPlayer(None,opp_aware=(opp_aware[0] == 1)),PongPlayer(None,opp_aware=(opp_aware[1] == 1)))
    np.random.seed(123)
    #env.seed(123)
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

    # Next, we build a very simple model.
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space_1.shape))
    #actor.add(keras.layers.normalization.BatchNormalization())
    for size in layers1:
        actor.add(Dense(size, kernel_initializer=RandomUniform(minval=-0.005, maxval=0.005, seed=None)))
        #actor.add(keras.layers.core.Dropout(0.2))
        actor.add(LeakyReLU(leaky_alpha))
    #actor.add(keras.layers.normalization.BatchNormalization())
    actor.add(Dense(nb_actions, kernel_initializer=RandomUniform(minval=-0.005, maxval=0.005, seed=None),bias_regularizer=regularizers.l2(0.01)))
    #actor.add(keras.layers.core.Dropout(0.2))
    actor.add(Activation('linear'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space_1.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = merge([action_input, flattened_observation], mode='concat')
    #x = keras.layers.normalization.BatchNormalization()(x)
    for size in layers1:
        x = Dense(size)(x)
        #x = keras.layers.core.Dropout(0.2)(x)
        x = LeakyReLU(alpha=leaky_alpha)(x)
    #x = keras.layers.normalization.BatchNormalization()(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(input=[action_input, observation_input], output=x)
    print(critic.summary())

    actor2 = Sequential()
    actor2.add(Flatten(input_shape=(1,) + env.observation_space_2.shape))
    #actor2.add(keras.layers.normalization.BatchNormalization())
    for size in layers2:
        actor2.add(Dense(size, kernel_initializer=RandomUniform(minval=-0.005, maxval=0.005, seed=None)))
        #actor2.add(keras.layers.core.Dropout(0.2))
        actor2.add(LeakyReLU(alpha=leaky_alpha))
    actor2.add(Dense(nb_actions, kernel_initializer=RandomUniform(minval=-0.005, maxval=0.005, seed=None),bias_regularizer=regularizers.l2(0.01)))
    #actor2.add(keras.layers.core.Dropout(0.2))
    actor2.add(Activation('linear'))
    print(actor2.summary())

    action_input2 = Input(shape=(nb_actions,), name='action_input')
    observation_input2 = Input(shape=(1,) + env.observation_space_2.shape, name='observation_input')
    flattened_observation2 = Flatten()(observation_input2)
    x2 = merge([action_input2, flattened_observation2], mode='concat')
    #x2 = keras.layers.normalization.BatchNormalization()(x2)
    for size in layers2:
        x2 = Dense(size)(x2)
        #x2 = keras.layers.core.Dropout(0.2)(x2)
        x2 = LeakyReLU(alpha=leaky_alpha)(x2)
    x2 = Dense(1)(x2)
    x2 = Activation('linear')(x2)
    critic2 = Model(input=[action_input2, observation_input2], output=x2)
    print(critic2.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory1 = SequentialMemory(limit=50000, window_length=1)
    if opp_aware[0] != opp_aware[1]:
        memory2 = SequentialMemory(limit=50000, window_length=1)
    else:
        memory2 = memory1
    random_process1 = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3, sigma_min=0.,n_steps_annealing=4*n_steps) # Explores less at the end ?
    random_process2 = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3, sigma_min=0.,n_steps_annealing=4*n_steps)
    agent1 = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory1, nb_steps_warmup_critic=5000, nb_steps_warmup_actor=5000,
                      random_process=random_process1, gamma=.99, target_model_update=1e-3, batch_size=200)
    agent2 = DDPGAgent(nb_actions=nb_actions, actor=actor2, critic=critic2, critic_action_input=action_input2,
                      memory=memory2, nb_steps_warmup_critic=5000, nb_steps_warmup_actor=5000,
                      random_process=random_process2, gamma=.99, target_model_update=1e-3, batch_size=200)

    #agent.compile(Adam(lr=L_R, clipnorm=1., clipvalue=0.5), metrics=['mae'])
    agent1.compile(Adam(lr=L_R, clipnorm=1.), metrics=['mae'])
    agent2.compile(Adam(lr=L_R, clipnorm=1.), metrics=['mae'])

    player1 = PongPlayer(agent1, myopie=myopie[0], opp_aware=(opp_aware[0] == 1))
    player2 = PongPlayer(agent2, myopie=myopie[1], opp_aware=(opp_aware[1] == 1))

    # Grid -4
    # Add -1 when lost
    # CEM method

    directory_log = "logs/ddpg/{}".format(conf_name)
    directory_weights = "weights/ddpg/{}".format(conf_name)

    if not os.path.exists(directory_log):
        os.makedirs(directory_log)
    if not os.path.exists(directory_weights):
        os.makedirs(directory_weights)

    if only_test:
        if weights1_name =='':
            weights1_name = "{}/player1_final".format(directory_weights)
        if weights2_name == '':
            weights2_name = "{}/player2_final".format(directory_weights)
        #if os.path.isfile(weights1_name) and os.path.isfile(weights2_name):
        agent1.load_weights(weights1_name)
        agent2.load_weights(weights2_name)

        env = makeEnv(player1,player2,ENV_NAME,ball_speed=ball_speed)
        for i in range(10):
            playPong(env)
        confrontPlayers(env)
        plotStrategy(env)


    else:


        for i in range(n_alternances):

            print "Alternance n {} \n".format(i)
            def learning_rate_schedule(epoch):
                return L_R

            if ENV_NAME == 'Env2D':
                env = Game2D(agent2,wall_reward=wall_reward, touch_reward=touch_reward)
            elif ENV_NAME == 'EnvPong':
                env = Pong(player1,player2,wall_reward=wall_reward, touch_reward=touch_reward,ball_speed=ball_speed)
            agent1.fit(env, nb_steps=n_steps, visualize=False, verbose=1, nb_max_episode_steps=None,callbacks=[FileLogger("{}/player1_{}.h5f".format(directory_log,i)), keras.callbacks.LearningRateScheduler(learning_rate_schedule)])
            agent1.test(env, nb_episodes=100, visualize=False, nb_max_episode_steps=500, verbose=1)
            agent1.save_weights("{}/player1_{}".format(directory_weights,i), overwrite=True)
            agent1.memory = SequentialMemory(limit=500000, window_length=1)
            wall_reward = wall_reward * 0.8
            touch_reward = touch_reward * 0.8
            agent2.load_weights("{}/player1_{}".format(directory_weights,i))


        print "Fin de {}".format(conf_name)
        env = Pong(player1,player2,wall_reward=wall_reward,touch_reward=touch_reward,ball_speed=ball_speed)

        #agent1.fit(env, nb_steps=150000, visualize=False, verbose=2, nb_max_episode_steps=None,callbacks=[FileLogger("logs/ddpg/{}_weights_steps_leaky_reg_bias_drop_lr{}.h5f".format(ENV_NAME,L_R), interval=100)])
        agent1.save_weights("{}/player1_final".format(directory_weights), overwrite=True)
        agent2.save_weights("{}/player2_final".format(directory_weights), overwrite=True)

        agent1.test(env, nb_episodes=15, visualize=False, nb_max_episode_steps=500, verbose=2)

    if show == True:

        if ENV_NAME == 'Env2D':
            for i in range(10):
                play2D(player1=agent1, player2=agent1)
        elif ENV_NAME == 'EnvPong':
            for i in range(10):
                playPong(left=agent1, right=agent2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('ENV_NAME', type=str, default='EnvPong', help='The environment\'s name of the game.')
    parser.add_argument('--L_R', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--layers1', type=int, default=[200], help='Hidden layers of player 1', nargs='+')
    parser.add_argument('--layers2', type=int, default=[200], help='Hidden layers of player 2', nargs='+')
    parser.add_argument('--leaky_alpha', type=float, default=0.03, help='Alpha parameter for the leaky Relu function.')
    parser.add_argument("--show", help="Play animations at the end.", action="store_true")
    parser.add_argument("--opp_aware", help="Are the players aware of their opponent's moves ? 1 if True, other int if False", default=[1,1], type=int, nargs=2)
    parser.add_argument("--myopie", help="Myopias of the two players (during the training)", type = float, default = [0.,0.], nargs=2)
    parser.add_argument("--wall_reward", help="Reward when a player hits the wall (should be < 0).", default=-0.1, type=float)
    parser.add_argument("--touch_reward", help="Reward when a player touches the ball (should be 1 > x >= 0).", default=0.3, type=float)
    parser.add_argument("--n_alternances", help="Number of alternances between player 1 and 2 duting training", default=10, type=int)
    parser.add_argument("--n_steps", help="Nombre de steps par alternance for each player.", default=80000, type=int)
    parser.add_argument("--only_test", help="If set, then the training part is skipped, the weights are loaded and only the testing is done", action="store_true")
    parser.add_argument("--ball_speed", type=float, help="Norm of tyhe ball's velocity", default = 1.0)
    parser.add_argument("--weights1_name", type=str, help="Path to the player 1's weights (for testing mode only, the sizes have to be entered correctly)", default='')
    parser.add_argument("--weights2_name", type=str, help="Path to the player 2's weights (for testing mode only, the sizes have to be entered correctly)", default='')

    args = parser.parse_args()

    main(**vars(args))

#Prioritized exp replay with bins
#convergence pour 2 agents
#autonomous agents and Multi agent systems
#exploration noise depending on Q
#Fourier
