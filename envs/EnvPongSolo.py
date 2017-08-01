import numpy as np

class EnvPongSolo():

    def __init__(self,e_x=-0.8,alpha=0.4,v_max = 2.,L = 1., d= 3., sigma_v=0.,sigma_o=0.):
        self.L = L
        self.d = d
        self.v_max = v_max #m/s
        self.delta_t = 1.
        self.e_x = e_x
        self.alpha = alpha
        self.state = np.zeros(6)
        self.observation_space = self.state
        self.action_space = np.zeros(1)
        self.sigma_v = sigma_v
        self.sigma_o = sigma_o
        self.reset()

    def reset(self):
        self.v_ball = np.array([1.,np.tan(np.clip(np.random.normal(0.,np.pi/4),-np.pi/2 + 0.2,np.pi/2 - 0.2))])
        #print "v_ball = ", self.v_ball
        self.p_ball = np.array([0,(np.random.rand()-0.5)*self.L])
        self.p_player = 0.
        self.v_player = 0.
        self.colTimes = np.zeros(3)
        self.time = 0.
        self.compute_colls()
        self.w_ball = 0.
        self.over = False
        self.update_state()
        return np.copy(self.state)

    def compute_colls(self):
        if self.v_ball[1] > 0:
            self.colTimes[0] = self.time + (self.L/2-self.p_ball[1])/self.v_ball[1]
            self.colTimes[1] = self.time + 10000
        else:
            self.colTimes[1] = self.time + (-self.L/2-self.p_ball[1])/self.v_ball[1]
            self.colTimes[0] = self.time + 10000
        self.colTimes[2] = self.time + (self.d - self.p_ball[0])/self.v_ball[0]

    def update_state(self):

        #l'incertitude est sur l'angle de la vitesse et la norme de la vitesse
        angle_speed = np.arctan(self.v_ball[1]/self.v_ball[0])
        angle_perceived = np.clip(angle_speed + np.random.normal(0., self.sigma_o*np.pi),-np.pi/2,np.pi/2)
        norm_speed = np.sqrt(self.v_ball[0]**2 + self.v_ball[1]**2)
        norm_speed_perceived = norm_speed*np.random.normal(1.,self.sigma_v)
        dist_ball = np.sqrt((self.p_ball[0] - self.d)**2 + (self.p_ball[1]-self.p_player)**2)
        if self.p_ball[0] != self.d:
            angle_player_ball = np.arctan((self.p_ball[1]-self.p_player)/(self.d - self.p_ball[0]))
        else:
            angle_player_ball = 0.
        angle_player_ball_perceived = angle_player_ball + np.random.normal(0.,self.sigma_o*np.pi)

        self.state[0] = self.p_player
        self.state[1] = norm_speed_perceived*np.cos(angle_perceived)
        self.state[2] = norm_speed_perceived*np.sin(angle_perceived)
        self.state[3] = self.d/2 - dist_ball*np.cos(angle_player_ball_perceived) #d/2 for 0 average
        self.state[4] = self.p_player + dist_ball*np.sin(angle_player_ball_perceived)
        self.state[5] = self.w_ball

    def step(self, action):

        remaining = self.delta_t
        reward = 0.

        while remaining > 0 and self.over == False:

            if np.min(self.colTimes) > self.time + remaining:
                delta = remaining
                wall = -1
            else:
                delta = delta = np.min(self.colTimes) - self.time
                wall = np.argmin(self.colTimes)

            self.time += delta
            remaining -= delta

            self.p_player += self.v_max*np.tanh(action)*delta
            self.p_player = np.clip(self.p_player, - self.L/2, self.L/2)

            self.p_ball += delta*self.v_ball

            if wall == 0 or wall == 1:

                e_x = self.e_x
                alpha = self.alpha
                R = 0.05

                self.v_ball[1]*= -1
                if wall == 0: #bottom wall
                    self.v_ball[0] = ((1-alpha*e_x)*self.v_ball[0] + alpha*(1+e_x)*R*self.w_ball)/(1+alpha)
                    self.w_ball = ((1+e_x)*self.v_ball[0] + (alpha - e_x)*R*self.w_ball)
                elif wall == 1: #top wall
                    self.v_ball[0] = ((1-alpha*e_x)*self.v_ball[0] - alpha*(1+e_x)*R*self.w_ball)/(1+alpha)
                    self.w_ball = -((1+e_x)*self.v_ball[0] - (alpha - e_x)*R*self.w_ball)

            elif wall == 2:
                self.over = True
                reward = -np.abs(self.p_ball[1] - self.p_player)/self.L

            self.compute_colls()

        self.update_state()
        #print self.state

        return np.copy(self.state), float(reward), self.over, {}
