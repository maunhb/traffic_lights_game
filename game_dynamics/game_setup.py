import numpy as np

class GameVariables():
    def __init__(self, demand_1=1, demand_2=1, num_players_1=1000, num_players_2=1000, num_p=50, num_q=50, sim_time=10000):
        self.demand_1 = demand_1  
        self.demand_2 = demand_2  
        self.num_players_1 = num_players_1
        self.num_players_2 = num_players_2
        self.num_p = num_p
        self.num_q = num_q
        self.sim_time = sim_time
    #---- waiting functions----
    def wait(self,x,p):
        #return np.tan(2*p/np.pi)
        return x*(np.exp(p)-1)

    # ----edge cost functions----
    def edgecost(self,x):
        return x + self.wait(x,0.5)
    def edge2(self,x,q):
        return x + self.wait(x,1-q)
    def edge7(self,x,p):
        return x + self.wait(x,p)
    def edge8(self,x,q):
        return x + self.wait(x,q)
    def edge12(self,x,p):
        return x + self.wait(x,1-p)
    def edge(self,x):
        return x 

    # ----OD player strategy costs----
    def cost_1_1(self,x,p,q):
        cost = self.edgecost(x[0])
        cost += self.edgecost(x[5])
        cost += self.edge12(x[11],p)
        cost += self.edgecost(x[12])
        cost += self.edge(x[13])
        return cost
    def cost_1_2(self,x,p,q):
        cost = self.edgecost(x[0])
        cost += self.edge2(x[1],q)
        cost += self.edge7(x[6],p)
        cost += self.edgecost(x[12])
        cost += self.edge(x[13])
        return cost
    def cost_1_3(self,x,p,q):
        cost = self.edgecost(x[0])
        cost += self.edge2(x[1],q)
        cost += self.edgecost(x[2])
        cost += self.edgecost(x[8])
        cost += self.edge(x[13])
        return cost
    
    def cost_1_4(self,x,p,q):
        cost = self.edgecost(x[0])
        cost += self.edgecost(x[5])
        cost += self.edge12(x[11],p)
        cost += self.edge8(x[7],q)
        cost += self.edgecost(x[2])
        cost += self.edgecost(x[8])
        cost += self.edge(x[13])
        return cost

    def cost_2_1(self,x,p,q):
        cost = self.edgecost(x[10])
        cost += self.edgecost(x[4])
        cost += self.edge2(x[1],q)
        cost += self.edgecost(x[2])
        cost += self.edge(x[3])
        return cost
    def cost_2_2(self,x,p,q):
        cost = self.edgecost(x[10])
        cost += self.edge12(x[11],p)
        cost += self.edge8(x[7],q)
        cost += self.edgecost(x[2])
        cost += self.edge(x[3])
        return cost
    def cost_2_3(self,x,p,q):
        cost = self.edgecost(x[10])
        cost += self.edge12(x[11],p)
        cost += self.edgecost(x[12])
        cost += self.edgecost(x[9])
        cost += self.edge(x[3])
        return cost    
    def cost_2_4(self,x,p,q):
        cost = self.edgecost(x[10])
        cost += self.edgecost(x[4])
        cost += self.edge2(x[1],q)
        cost += self.edge7(x[6],p)
        cost += self.edgecost(x[12])
        cost += self.edgecost(x[9])
        cost += self.edge(x[3])
        return cost

    ## -----TL player strategy costs----

    def cost_tl1(self,x,q):
        return self.edge2(x[1],q) + self.edge8(x[7],q) 
    def cost_tl2(self,x,p):
        return self.edge12(x[11],p) + self.edge7(x[6],p)

    def cost_tl(self,x,p,q):
        return self.edge12(x[11],p) + self.edge7(x[6],p) + self.edge2(x[1],q) + self.edge8(x[7],q)

    def social_cost(self,x,p,q):
        cost =  x[1]*self.edge2(x[1],q) + x[6]*self.edge7(x[6],p) + x[7]*self.edge8(x[7],q) + x[11]*self.edge12(x[11],p)
        cost += x[0]*self.edgecost(x[0]) + x[2]*self.edgecost(x[2]) + x[3]*self.edge(x[3]) + x[4]*self.edgecost(x[4])
        cost += x[5]*self.edgecost(x[5]) + x[8]*self.edgecost(x[8]) + x[9]*self.edgecost(x[9]) + x[10]*self.edgecost(x[10])
        cost += x[12]*self.edgecost(x[12]) + x[13]*self.edge(x[13])
        return cost


