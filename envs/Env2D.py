#coding=utf-8
# Ici, pas de gravité
import numpy as np
from numpy import linalg as LA

class Player2D:

    def __init__(self, theta):
        self.rate = 0.08
        self.theta = theta
        self.dtheta = 0
        self.move = 0 # +1 = dtheta ++ / -1 = dtheta-- / 0 = dtheta constant

    def update_pos(self,delta):
        self.theta += self.dtheta*delta
        self.theta = np.max([np.min([np.pi/2, self.theta]),0.])
        '''if self.theta == 0 or self.theta == np.pi/2:
            self.dtheta = 0.'''

    def update_dteta(self,move):
        self.move = move
        self.dtheta += self.rate*move
        '''if self.theta == 0:
            self.dtheta = max(0,self.dtheta)
        elif self.theta == np.pi/2:
            self.dtheta = min(0,self.dtheta)'''

    def reset(self,theta):
        self.theta = theta
        self.move = 0.
        self.dtheta = 0.

class Game2D:

    def __init__(self, agent, wall_reward = 0.00, touch_reward = 0.00):
        self.agent = agent
        self.wall_reward = wall_reward
        self.touch_reward = touch_reward
        self.action_space = np.zeros((1))
        self.action_space[0] = 1
        self.observation_space = np.zeros(9)
        self.R = 2.
        self.P1 = Player2D(np.pi/4)
        self.P2 = Player2D(np.pi/4)
        self.p_ball = np.array([R/3.,R/3.])
        angle = np.random.rand()*np.pi*2
        self.v_ball = 1.*np.array([np.cos(angle),np.sin(angle)])
        self.s_ball = 0.5
        self.state1 = np.zeros(9)
        self.state2 = np.zeros(9)
        self.colTimes = np.zeros(3)
        self.time = 0.
        self.coeffs = np.zeros(3) # Pour colTimes
        self.over = False
        self.dt = 0.05
        self.touch = 0
        self.positions = []
        self.collisions = 0
        self.toPlay = np.random.randint(1,3)
        self.update_state()
        self.update_col()

    def reset(self):
        self.time = 0.
        self.over = False
        self.touch = 0
        self.P1.reset(np.pi/4)
        self.P2.reset(np.pi/4)
        self.p_ball = np.array([self.R/3,self.R/3])
        angle = np.random.rand()*np.pi*2
        self.v_ball = 1.*np.array([np.cos(angle),np.sin(angle)])
        self.s_ball = np.random.normal(0., 0.8)
        self.collisions = 0
        self.toPlay = np.random.randint(1,3)
        #self.toPlay = 1
        self.update_state()
        self.update_col()
        return self.state1

    def update_col(self):
        self.colTimes[0] = self.time - self.p_ball[0]/self.v_ball[0]
        self.colTimes[1] = self.time - self.p_ball[1]/self.v_ball[1]
        self.coeffs[0] = LA.norm(self.v_ball)**2
        self.coeffs[1] = 2*np.dot(self.v_ball, self.p_ball)
        self.coeffs[2] = LA.norm(self.p_ball)**2 - self.R**2
        roots = np.roots(self.coeffs)
        posroots = roots[roots > 0.000001]
        if len(posroots) > 0:
            self.colTimes[2] = self.time + np.min(posroots)
        elif self.over == False:
            print "caca, pas de solution"
            print "coeffs = ", self.coeffs
            self.colTimes[2] = 10000 + self.time
            self.over = True
        for i in range(2):
            if self.colTimes[i] <= self.time + 0.00001:
                self.colTimes[i] = 10000 + self.time

        #print self.colTimes

    def update_state(self):
        if self.toPlay == 1:
            #self.state1[0] = (self.p_ball[0] - self.R/3.)/self.R # int(x*sqrt(1-x^2), 0,1) = 1/3
            #self.state1[1] = (self.p_ball[1] - self.R/3.)/self.R
            self.state1[0] = self.p_ball[0]/self.R
            self.state1[1] = self.p_ball[1]/self.R
            self.state1[2] = self.v_ball[0]
            self.state1[3] = self.v_ball[1]
            self.state1[4] = self.s_ball
            self.state1[5] = (self.P1.theta - np.pi/4)/(np.pi/2)
            self.state1[6] = self.P1.dtheta
            self.state1[7] = (self.P2.theta - np.pi/4)/(np.pi/2)
            self.state1[8] = self.P2.dtheta

            #self.state2[0] = -(self.p_ball[0] - self.R/3.)/self.R
            #self.state2[1] = -(self.p_ball[1] - self.R/3.)/self.R
            self.state2[0] = -self.p_ball[0]/self.R
            self.state2[1] = -self.p_ball[1]/self.R
            self.state2[2] = -self.v_ball[0]
            self.state2[3] = -self.v_ball[1]
            self.state2[4] = -self.s_ball
            self.state2[5] = (self.P2.theta - np.pi/4)/(np.pi/2)
            self.state2[6] = self.P2.dtheta
            self.state2[7] = (self.P1.theta - np.pi/4)/(np.pi/2)
            self.state2[8] = self.P1.dtheta

        else:
            #self.state1[0] = -(self.p_ball[0] - self.R/3.)/self.R # int(x*sqrt(1-x^2), 0,1) = 1/3
            #self.state1[1] = -(self.p_ball[1] - self.R/3.)/self.R
            self.state1[0] = -self.p_ball[0]/self.R
            self.state1[1] = -self.p_ball[1]/self.R
            self.state1[2] = -self.v_ball[0]
            self.state1[3] = -self.v_ball[1]
            self.state1[4] = -self.s_ball
            self.state1[5] = (self.P1.theta - np.pi/4)/(np.pi/2)
            self.state1[6] = self.P1.dtheta
            self.state1[7] = (self.P2.theta - np.pi/4)/(np.pi/2)
            self.state1[8] = self.P2.dtheta

            #self.state2[0] = (self.p_ball[0] - self.R/3.)/self.R
            #self.state2[1] = (self.p_ball[1] - self.R/3.)/self.R
            self.state2[0] = self.p_ball[0]/self.R
            self.state2[1] = self.p_ball[1]/self.R
            self.state2[2] = self.v_ball[0]
            self.state2[3] = self.v_ball[1]
            self.state2[4] = self.s_ball
            self.state2[5] = (self.P2.theta - np.pi/4)/(np.pi/2)
            self.state2[6] = self.P2.dtheta
            self.state2[7] = (self.P1.theta - np.pi/4)/(np.pi/2)
            self.state2[8] = self.P1.dtheta


    def step(self,move):

        reward = 0
        self.P1.update_dteta(move)
        action2 = self.agent.forward(self.state2)
        self.P2.update_dteta(action2)
        remaining = self.dt

        while(remaining > 0):

            if self.time + remaining < np.min(self.colTimes) or self.over == True:
                self.P1.update_pos(remaining)
                self.P2.update_pos(remaining)

                self.p_ball = self.p_ball + remaining * self.v_ball
                self.time = self.time + remaining
                remaining = 0.

            else:
                self.collisions += 1

                delta = np.min(self.colTimes) - self.time
                #print "delta = ", delta
                self.P1.update_pos(delta)
                self.P2.update_pos(delta)

                self.p_ball = self.p_ball + delta*self.v_ball
                self.time = self.time + delta
                remaining = remaining - delta

                wall = np.argmin(self.colTimes)

                M = np.zeros((2,2))
                theta = 0
                alpha = 0.4
                r_ball = 0.02
                e_x = 1.0
                e_y = 1.0

                if wall == 0:
                    oldVPar = self.v_ball[1]
                    self.v_ball[1] = ((1-alpha*e_x)*self.v_ball[1]+alpha*(1+e_x)*r_ball*self.s_ball)/(1+alpha)
                    self.v_ball[0] = -e_y*self.v_ball[0]
                    self.s_ball = ((1+e_x)*oldVPar + (alpha - e_x)*r_ball*self.s_ball)/(r_ball*(1+alpha))
                elif wall == 1:
                    oldVPar = self.v_ball[0]
                    self.v_ball[0] = ((1-alpha*e_x)*self.v_ball[0]+alpha*(1+e_x)*r_ball*self.s_ball)/(1+alpha)
                    self.v_ball[1] = -e_y*self.v_ball[1]
                    self.s_ball = ((1+e_x)*oldVPar + (alpha - e_x)*r_ball*self.s_ball)/(r_ball*(1+alpha))

                elif wall ==2:

                    if self.toPlay == 1:
                        player = self.P1
                    else:
                        player = self.P2

                    theta = np.arctan(self.p_ball[1]/self.p_ball[0])

                    M[0,0] = np.cos(theta)
                    M[0,1] = - np.sin(theta)
                    M[1,0] = np.sin(theta)
                    M[1,1] = np.cos(theta)

                    pre_energy = LA.norm(self.v_ball)**2 + alpha*(r_ball**2)*self.s_ball**2
                    #theta_impact = theta - np.pi/2

                    oldV = np.dot(LA.inv(M), self.v_ball) - self.R*player.dtheta*np.array([0.,1.])
                    #print "mur : ", wall
                    #print "v_avant impact : ", self.v_ball
                    #print "v_transformée : ", oldV

                    self.v_ball[1] = ((1-alpha*e_x)*oldV[1]+alpha*(1+e_x)*r_ball*self.s_ball)/(1+alpha)
                    self.v_ball[0] = -1.02*e_y*oldV[0]

                    self.s_ball = ((1+e_x)*oldV[1] + (alpha - e_x)*r_ball*self.s_ball)/(r_ball*(1+alpha))
                    self.v_ball = np.dot(M,self.v_ball) + self.R*player.dtheta*np.array([-np.sin(theta), np.cos(theta)])

                    post_energy = LA.norm(self.v_ball)**2 + alpha*(r_ball**2)*self.s_ball**2

                    #print "v_après impact :", self.v_ball
                    #print "KineticRatio = ", pre_energy/post_energy

                if wall == 2 and self.over == False:

                    if np.abs(theta - player.theta) > 0.2:
                        self.over = True
                        if self.toPlay == self.trainingPlayer:
                            reward = -1
                        else:
                            reward = 1 # Change to 1
                    else:
                        self.touch = self.touch + 1
                        if self.toPlay == self.trainingPlayer:
                            reward = 0.1
                    if self.toPlay == 1:
                        self.toPlay = 2
                    else:
                        self.toPlay = 1
                        #reward = 1
                    if self.screen == True:
                        print("Angle : %.2f \t Delta : %.2f \t dtheta : %.2f" % (player.theta, np.abs(theta - player.theta), player.dtheta))

                self.update_col()
        if self.trainingPlayer == 1:
            if np.abs(self.P1.dtheta) > 3 or self.P1.dtheta == np.nan:
                reward = float(reward - 3)
                self.over = True

        elif self.trainingPlayer == 2:
            if np.abs(self.P2.dtheta) > 3 or self.P2.dtheta == np.nan:
                reward = float(reward - 3)
                self.over = True

        self.update_state()

        if self.savePos == True:
            self.positions.append(self.p_ball)
        if self.trainingPlayer == 1:
            return self.state1, reward, self.over, {}
        else:
            return self.state2, reward, self.over, {}
