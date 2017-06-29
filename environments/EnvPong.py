#coding=utf-8
import numpy as np

class PongPlayer():

    def __init__(self, agent, myopie = 0.00, opp_aware = True):
        self.agent = agent
        self.myopie = myopie
        self.opp_aware = opp_aware

class Pong():

    def __init__(self, player_left, player_right, wall_reward=-0.0, touch_reward=0.0, ball_speed = 1.):
        # Myopie : on multiplie la position et la vitesse perçues par np.random.normal(1., myopie*distance) où distance
        self.action_space = np.zeros((1))
        self.action_space[0] = 1
        self.player_left = player_left
        self.player_right = player_right
        if self.player_left.opp_aware:
            self.observation_space_1 = np.zeros(9)
        else:
            self.observation_space_1 = np.zeros(7)
        if self.player_right.opp_aware:
            self.observation_space_2 = np.zeros(9)
        else:
            self.observation_space_2 = np.zeros(7)
        self.colTimes = np.zeros(4)
        self.dt = 0.05
        self.rate = 0.5
        self.r = 0.01 # Rayon de la balle
        self.alpha = 0.4
        self.wall_reward = wall_reward
        self.touch_reward = touch_reward
        self.ball_speed = ball_speed
        self.reset()

    def reset(self):
        self.time = 0.
        self.over = False
        self.touch = 0
        self.p_left = 0.
        self.p_right = 0.
        self.v_left = 0.
        self.v_right = 0.
        angle = np.pi*(3./4.) + np.random.rand()*np.pi/2
        if np.random.rand() > 0.5:
            angle += np.pi
        self.p_ball = np.zeros(2)
        self.v_ball = self.ball_speed*np.array([np.cos(angle), np.sin(angle)])
        self.s_ball = np.random.normal(0.,1.)
        self.compute_coll()
        self.update_states()
        return self.stateLeft

    def update_states(self):
        d1 = np.sqrt((self.p_left - self.p_ball[1])**2 + (self.p_ball[0] + 1)**2)
        d2 = np.sqrt((self.p_right - self.p_ball[1])**2 + (self.p_ball[0] -1)**2)
        if self.player_left.myopie < 0:
            d1 = np.sqrt(5) - d1
        if self.player_right.myopie < 0:
            d2 = np.sqrt(5) - d2
        if self.player_left.opp_aware:
            self.stateLeft = np.zeros(9)
            self.stateLeft[0] = self.p_left + np.random.normal(0.,np.abs(self.player_left.myopie))
            self.stateLeft[1] = self.p_right + np.random.normal(0.,np.abs(self.player_left.myopie)*2)
            self.stateLeft[2] = self.v_left + np.random.normal(0.,np.abs(self.player_left.myopie)*2)
            self.stateLeft[3] = self.v_right + np.random.normal(0.,np.abs(self.player_left.myopie)*2)
            self.stateLeft[4] = self.p_ball[0] + np.random.normal(0.,np.abs(self.player_left.myopie)*d1)
            self.stateLeft[5] = self.p_ball[1] + np.random.normal(0.,np.abs(self.player_left.myopie)*d1)
            self.stateLeft[6] = self.v_ball[0] + np.random.normal(0.,np.abs(self.player_left.myopie)*d1)
            self.stateLeft[7] = self.v_ball[1] + np.random.normal(0.,np.abs(self.player_left.myopie)*d1)
            self.stateLeft[8] = self.s_ball + np.random.normal(0.,np.abs(self.player_left.myopie)*d1)
        else:
            self.stateLeft = np.zeros(7)
            self.stateLeft[0] = self.p_left + np.random.normal(0.,np.abs(self.player_left.myopie))
            self.stateLeft[1] = self.v_left + np.random.normal(0.,np.abs(self.player_left.myopie)*2)
            self.stateLeft[2] = self.p_ball[0] + np.random.normal(0.,np.abs(self.player_left.myopie)*d1)
            self.stateLeft[3] = self.p_ball[1] + np.random.normal(0.,np.abs(self.player_left.myopie)*d1)
            self.stateLeft[4] = self.v_ball[0] + np.random.normal(0.,np.abs(self.player_left.myopie)*d1)
            self.stateLeft[5] = self.v_ball[1] + np.random.normal(0.,np.abs(self.player_left.myopie)*d1)
            self.stateLeft[6] = self.s_ball + np.random.normal(0.,np.abs(self.player_left.myopie)*d1)

        if self.player_right.opp_aware:
            self.stateRight = np.zeros(9)
            self.stateRight[0] = -self.p_right + np.random.normal(0.,np.abs(self.player_right.myopie))
            self.stateRight[1] = -self.p_left + np.random.normal(0.,np.abs(self.player_right.myopie)*2)
            self.stateRight[2] = -self.v_right + np.random.normal(0.,np.abs(self.player_right.myopie)*2)
            self.stateRight[3] = -self.v_left + np.random.normal(0.,np.abs(self.player_right.myopie)*2)
            self.stateRight[4] = -self.p_ball[0] + np.random.normal(0.,np.abs(self.player_right.myopie)*d2)
            self.stateRight[5] = -self.p_ball[1] + np.random.normal(0.,np.abs(self.player_right.myopie)*d2)
            self.stateRight[6] = -self.v_ball[0] + np.random.normal(0.,np.abs(self.player_right.myopie)*d2)
            self.stateRight[7] = -self.v_ball[1] + np.random.normal(0.,np.abs(self.player_right.myopie)*d2)
            self.stateRight[8] = self.s_ball + np.random.normal(0.,np.abs(self.player_right.myopie)*d2)
        else:
            self.stateRight = np.zeros(7)
            self.stateRight[0] = -self.p_right + np.random.normal(0.,np.abs(self.player_right.myopie))
            self.stateRight[1] = -self.v_right + np.random.normal(0.,np.abs(self.player_right.myopie)*2)
            self.stateRight[2] = -self.p_ball[0] + np.random.normal(0.,np.abs(self.player_right.myopie)*d2)
            self.stateRight[3] = -self.p_ball[1] + np.random.normal(0.,np.abs(self.player_right.myopie)*d2)
            self.stateRight[4] = -self.v_ball[0] + np.random.normal(0.,np.abs(self.player_right.myopie)*d2)
            self.stateRight[5] = -self.v_ball[1] + np.random.normal(0.,np.abs(self.player_right.myopie)*d2)
            self.stateRight[6] = self.s_ball + np.random.normal(0.,np.abs(self.player_right.myopie)*d2)

    def compute_coll(self):
        if self.v_ball[0] > 0:
            self.colTimes[0] = self.time + 10000
            self.colTimes[1] = self.time + (1-self.p_ball[0])/self.v_ball[0]
        else:
            self.colTimes[0] = self.time - (self.p_ball[0]+1)/self.v_ball[0]
            self.colTimes[1] = self.time + 10000
        if self.v_ball[1] > 0:
            self.colTimes[2] = self.time + (0.5-self.p_ball[1])/self.v_ball[1]
            self.colTimes[3] = self.time + 10000
        else:
            self.colTimes[2] = self.time + 10000
            self.colTimes[3] = self.time - (self.p_ball[1]+0.5)/self.v_ball[1]

    def step(self,actionLeft):

        actionRight = - self.player_right.agent.forward(self.stateRight)
        self.v_left += self.rate*np.tanh(actionLeft)
        self.v_right += self.rate*np.tanh(actionRight)
        if self.p_left == 0.5:
            self.v_left = min(self.v_left, 0.)
        elif self.p_left == -0.5:
            self.v_left = max(self.v_left, 0.)
        if self.p_right == 0.5:
            self.v_right = min(self.v_right, 0.)
        elif self.p_right == -0.5:
            self.v_right = max(self.v_right, 0.)
        remaining = self.dt
        rewardLeft = 0
        rewardRight = 0

        while(remaining > 0 and self.over == False):

            #print("boucle, t={}, remaining={}".format(self.time, remaining))

            if np.min(self.colTimes) > self.time + remaining:

                delta = remaining
                self.p_ball += delta*self.v_ball
                self.p_right = np.clip(self.p_right + self.v_right*delta, -0.5,0.5)
                self.p_left = np.clip(self.p_left + self.v_left*delta, -0.5,0.5)
                self.time += delta
                remaining -= delta

            else:

                #print("collision n {}".format(self.touch))
                delta = np.min(self.colTimes) - self.time
                wall = np.argmin(self.colTimes)
                self.p_ball += delta*self.v_ball
                self.p_right = np.clip(self.p_right + self.v_right*delta, -0.5,0.5)
                self.p_left = np.clip(self.p_left + self.v_left*delta, -0.5,0.5)
                self.time += delta
                remaining -= delta

                e_x = 1.0

                if wall == 0:
                    if np.abs(self.p_left - self.p_ball[1]) < 0.1:
                        self.touch += 1
                        self.v_ball[0] = -self.v_ball[0]
                        v_par = self.v_ball[1] - self.v_left
                        self.v_ball[1] = ((1-self.alpha)*v_par+self.alpha*(1+e_x )*self.r*self.s_ball)/(1+self.alpha) + self.v_left
                        self.s_ball = ((1+e_x )*v_par + (self.alpha - e_x )*self.r*self.s_ball)/(self.r*(1+self.alpha))
                        rewardLeft += self.touch_reward
                    else:
                        self.over = True
                        # "over because left missed it touch : {}".format(self.touch)
                        rewardLeft -= 1
                        rewardRight += 1
                elif wall == 1:
                    if np.abs(self.p_right - self.p_ball[1]) < 0.1:
                        self.touch += 1
                        self.v_ball[0] = -self.v_ball[0]
                        v_par = self.v_ball[1] - self.v_right
                        self.v_ball[1] = ((1-self.alpha)*v_par-self.alpha*(1+e_x )*self.r*self.s_ball)/(1+self.alpha) + self.v_right
                        self.s_ball = -((1+e_x )*v_par - (self.alpha - e_x )*self.r*self.s_ball)/(self.r*(1+self.alpha))
                    else:
                        #print "over because right missed it touch : {}".format(self.touch)
                        self.over = True
                        rewardLeft += 1
                        rewardRight -= 1

                if wall == 2:
                    self.v_ball[1] = -self.v_ball[1]
                    #v_par = self.v_ball[0]
                    #self.v_ball[0] = ((1-self.alpha)*v_par+self.alpha*(1+e_x )*self.r*self.s_ball)/(1+self.alpha)
                    #self.s_ball = ((1+e_x )*v_par + (self.alpha - e_x )*self.r*self.s_ball)/(self.r*(1+self.alpha))
                elif wall == 3:
                    self.v_ball[1] = -self.v_ball[1]
                    #v_par = self.v_ball[0]
                    #self.v_ball[0] = ((1-self.alpha)*v_par-self.alpha*(1+e_x )*self.r*self.s_ball)/(1+self.alpha)
                    #self.s_ball = -((1+e_x )*v_par - (self.alpha - e_x )*self.r*self.s_ball)/(self.r*(1+self.alpha))

                self.compute_coll()

            self.update_states()

        if np.abs(self.p_left) == 0.5:
            rewardLeft += self.wall_reward
        #print self.stateLeft
        return self.stateLeft, rewardLeft, self.over, {"touches":self.touch}
