#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np  # for array stuff and random
import random
import flopy
import flopy.utils.binaryfile as bf


class pumpingAllocationEnv4(gym.Env):
    def __init__(self):
        self.seed()
        self.observation_space = spaces.Box(0, np.inf, shape=(7,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([11,11])
        
        self.modelname = 'flopy_ex'
        self.mf = flopy.modflow.Modflow(self.modelname, exe_name='C:/Users/hulab/Desktop/RL/mf2005')
        
        self.Lx = 10000                     #邊界長度 (m)
        self.Ly = 10000
        self.ztop = 10.0
        self.zbot = -110.0
        self.nlay = 1                      #number of layers
        self.nrow = 100                    #number of rows
        self.ncol = 100                    #number of columns
        self.delr = self.Lx/self.ncol      #cell width along rows
        self.delc = self.Ly/self.nrow      #cell width along columns
        self.delv = (self.ztop - self.zbot) / self.nlay             #cell width along layers
        self.botm = np.linspace(self.ztop, self.zbot, self.nlay+1)  #bottom elevation of a model layer or a Quasi-3d confining bed
        
        self.itmuni = 3  #itumni: indicates the time unit of model data (0:undefined, 1:seconds, 2:minutes, 3:hours, 4:days, 5:years)
                                    #lenuni: indicates the length unit of model data (0:undefined, 1:feet, 2:meters, 3:centimeters)
        self.hk = 0.36   #水力傳導度 hydraulic conductivity (m/hr)
        self.vka = 0.1
        self.sy = 0.1      # specific yield (無因次)
        self.ss = 0.001    # specific storage
        self.laytyp = 0   # 0: confined,  1: unconfined
        
        self.max_storage = 23500    #最大有效庫容
        self.t_final = 180        #總時間 (旬)
        self.ifRandtt = False     #是否隨機起始月份

        #河川入流量(萬噸)(假設隨季節浮動)
        river101 = np.array([799.5, 1645.5, 1122.8, 846.6, 807.2, 677.7, 672.4, 674.5, 1395.4, 855.6, 787.7, 826.2,
                             556.2, 4629.1, 3457.6, 2578.8, 3025.4, 13556.2, 4908.6, 6528.6, 4072.4, 3284.4, 4215.8, 3255.6, 
                             3534.4, 1669.1, 1218.6, 12024.2, 3612, 2454.1, 2326.3, 5538.5, 3203.9, 2583.3, 2581.6, 2074.6])
        river102 = np.array([1927.4, 1648.8, 1258.9, 963.5, 818.1, 649.1, 710.2, 668.1, 837.1, 2743.9, 4058.3, 3352.3,
                             3539.9, 6223.2, 6537.5, 2507.2, 2008.6, 2021, 2299.2, 29947.2, 5130.2, 2518.6, 3591.5, 30721.8,
                             8921.7, 3523.3, 16867.4, 19136.6, 6837.1, 3674.9, 2469.8, 1718.8, 1511, 1206.9, 3345.7, 3390.1])
        river103 = np.array([1795.6,1346.2,1200.3,1440.1,2086.7,1280,1572.8,2286.7,1611.2,1513.4,1151.8,1124.7,
                             4238.8,3483.1,6170.1,5575,4035.2,4214.9,4833.3,3553.4,14592.3,4415,3209.8,2552.4,
                             1869.3,1417.9,4770.4,1827.2,3792.2,1549.9,1145.1,1600.7,1168.3,1311.4,1150.6,1334.5])
        river104 = np.array([1038.8,852.2,866.7,606.4,778.8,529,470.2,926.8,1822.8,887.4,1038.4,834.5,
                             2525.4,1887.4,4678.6,1790.4,1690.3,2746.6,4567.5,7178.5,4746.7,27317.3,8986.1,9484.9,
                             8977.7,3371.3,21319.5,8857.6,3645.2,2756.8,1966.8,1578.2,1027.7,1024.4,938,901.4])
        river105 = np.array([1379.1,2447,6738.8,5012,2198.8,1515.1,1353.1,8667.6,8028.6,3006.1,4694,3014,
                             2331.4,3112.5,2592.9,4246.9,4649.1,5869.8,5985.7,6446.2,4116.2,2444.3,2955,1875.7,
                             2237.2,11217.2,30627.9,9508.5,7300.3,3882,2457.2,1809.2,2568.1,1708.5,1462.9,1324])
        river106 = np.array([1013.1, 939.7, 933.5, 838.1, 749.1, 1023.9, 1166.9, 1211.0, 1566.4, 1492.4, 1801.3, 5169.1, 
                            2207.5, 3141.9, 3650.3, 18996.7, 15899.4, 5402.1, 7994.3, 3639.5, 5628.9, 3633.3, 1958.2, 1677.7, 
                            1767.8, 1713.6, 1257.3, 2120.7, 13143.8, 3677.5, 2074.2, 2241.6, 2021.7, 1978.8, 1953.2, 1628.2 ])
        river107 = np.array([4418.6, 3129.4, 2168, 3179.3, 2956.1, 1761, 2269.4, 1719.6, 1284, 931.5, 932, 1060, 
                            1172.8, 985.9, 825, 1372.3, 2361.6, 3117, 2755.3, 11469, 3089, 1729.1, 2053.2, 6773, 
                            5635.1, 4962.4, 7730, 4698.5, 3866.6, 2367.6, 1851.8, 1381.9, 1164, 836.8, 771.3, 935])
        river108 = np.array([1117.3, 776.6, 704.4, 540.6, 454.9, 389, 3520.5, 2565.3, 1487.8, 946.4, 1988.8, 2352.5,
                            5767.9, 5340.7, 4556.8, 2438.5, 10567.9, 5242.1, 5323.1, 2556.8, 3479.6, 11307.8, 6969.6, 5678.6,
                            3103, 1937.4, 4701.6, 10945.4, 2285.8, 1629.3, 1744.8, 999, 1101.2, 2097.6, 1110.5, 1083.4])
        river109 = np.array([915.4, 610.2, 744.1, 548, 536.4, 369.8, 684.3, 976.4, 1389.6, 1262.7, 1093.8, 826.3,
                            602.9, 2794.4, 7561.1, 2465.8, 1533.3, 1175.9, 2042.2, 1232.6, 2277.7, 1746.5, 1486.9, 1467,
                            1660, 1548.3, 1140.8, 957, 1593.6, 2198.7, 1014.1, 1614.4, 1129, 2839.9, 1775.9, 1679.3])
        river110 = np.array([1062.9, 1058.6, 759.9, 547.5, 797.7, 405.1, 548.2, 394.3, 628.8, 391.2, 372.1, 585.3,
                            449, 336.2, 2153.1, 6819.5, 4164.5, 4439.4, 2242.5, 1484.6, 28267.9, 17129.5, 11652.5, 6132.4,
                            2891.3, 7208.4, 2627.4, 1592.1, 15388.1, 4817, 2449.3, 1728.8, 1486.7, 1147, 1120.8, 1310.8])
       
        self.rivers=[river101, river102, river103, river104, river105, river106, river107, river108, river109, river110]
        self.river_p = [0.1]*10        #入流情境機率
        
        #self.demands1 = np.array([1300,2600,2900,2700,2700,2800,3100,3000,2700,2800,2600,1300])
        self.demands1 = np.array([1201.95,1201.95,1201.95,2415.01,2598.18,2677.66,2812.45,2862.56,2948.96,2734.69,2665.57,2677.66,
                                2601.63,2529.06,2521.28,2587.81,2639.65,2691.49,2940.32,2969.7,3006.85,2948.96,2916.99,2902.3,
                                2713.09,2652.61,2687.17,2767.52,2767.52,2715.68,2605.95,2341.57,2110.02,1201.95,1201.95,1201.95])
        self.demands2 = np.array([1201.95,1201.95,1201.95,1915.01,2098.18,2177.66,2312.45,2362.56,2448.96,2234.69,2165.57,2177.66,
                                2101.63,2029.06,2021.28,2087.81,2139.65,2191.49,2940.32,2969.7,3006.85,2948.96,2916.99,2902.3,
                                3213.09,3152.61,2687.17,2767.52,2767.52,2715.68,2605.95,2341.57,2110.02,1201.95,1201.95,1201.95])
        self.demands = [self.demands1,self.demands2]
        self.demands_p = [0.5]*2
      
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
        
    def reset(self):

        self.nper = 2              # the number of stress periods in the simulation
        self.nstp = [1]+[1]*1      # the number of time steps in a stress period.
        self.perlen = [1]+[240]*1  # the length of a stress period
        self.steady = [True]+[False]*1
        
        self.ibound = np.ones((self.nlay, self.nrow, self.ncol), dtype=np.int32)   #boundary variable
        self.ibound[:,:,0] = -1
        self.ibound[:,:,-1] = 0
        self.ibound[:,0,:] = 0
        self.ibound[:,-1,:] = 0

        self.head = np.zeros((self.nlay, self.nrow, self.ncol), dtype=np.float32)   #initial head at the beginning of the simulation
        self.head[:,:,0] = 0
        #self.head[:,:,-1] = 80
        
        self.t = 0    #第幾旬
        self.tt = random.choice([0,6,9,15,27]) if self.ifRandtt else 0      #隨機初始起始月份
        #self.month = (self.tt//3)%12+1
        scenerio = np.random.choice([0,1,2,3,4,5,6,7,8,9],p=self.river_p)
        self.river = self.rivers[scenerio]   #隨機選擇入流情境
        self.inflow = self.river[self.t+self.tt] + random.randint(-2,2)*50 
        
        self.S = 18000    # 初始水庫水量

        d_scenerio = np.random.choice([0,1],p=self.demands_p)
        self.dmd = self.demands[d_scenerio]   #隨機選擇
        self.demand = self.dmd[self.t+self.tt]+random.randint(-4,4)*50         

        norm_S = 2*(self.S)/self.max_storage -1
        norm_Zmin =  2*(0 - (-100))/(0-(-100))-1
        #norm_month = 2*(self.month-1)/(12-1) -1
        norm_demand = 2*(self.demand-np.min(self.demands)+200)/(np.max(self.demands)-np.min(self.demands)+400) -1
        self.norm_inflow_t_1 = 0
        self.norm_inflow_t_2 = 0
        self.norm_inflow_t_3 = 0
        self.norm_inflow_t_4 = 0

        return np.array([norm_S, norm_Zmin, norm_demand, self.norm_inflow_t_1, self.norm_inflow_t_2, self.norm_inflow_t_3, self.norm_inflow_t_4 ])
    
 
    def step(self, action):
        pump = action[1]*216     #抽水
        GW = pump*20*240/10000        #10口井抽一旬的水量(萬噸)
        #stress period:[ layer, row, column, Q(m^3/hour)]
        pumping_wells = [[0,73,73,-pump],[0,14,32,-pump],[0,18,41,-pump],[0,20,66,-pump],[0,17,92,-pump],
                         [0,37,30,-pump],[0,14,47,-pump],[0,40,60,-pump],[0,38,72,-pump],[0,45,90,-pump],
                         [0,32,81,-pump],[0,50,50,-pump],[0,63,35,-pump],[0,60,65,-pump],[0,58,82,-pump],
                         [0,83,95,-pump],[0,75,40,-pump],[0,77,63,-pump],[0,88,80,-pump],[0,70,95,-pump]]
        Q = {1: pumping_wells}
        
        recharge=(self.inflow-np.min(self.rivers))/(np.max(self.rivers)-np.min(self.rivers))*(0.001-0.0002)+0.0002
        rech={0: 0.00005, 1: recharge}
        
        #================================MODFLOW_2005=====================================
        if self.t != 0:
            self.nper = 1
            self.nstp = [1]      
            self.perlen = [240]   
            self.steady = [False]
            Q = {0: pumping_wells}
            rech = {0: recharge}

        dis = flopy.modflow.ModflowDis(
            self.mf, self.nlay, self.nrow, self.ncol, delr=self.delr, delc=self.delc, top=self.ztop, botm=self.botm[1:],
            nper=self.nper, nstp=self.nstp, perlen=self.perlen, itmuni=self.itmuni, steady=self.steady
        )
        bas = flopy.modflow.ModflowBas(self.mf, ibound=self.ibound, strt=self.head)
        lpf = flopy.modflow.ModflowLpf(self.mf, laytyp=self.laytyp, hk=self.hk, sy=self.sy, ss=self.ss)
        gmg = flopy.modflow.ModflowGmg(self.mf)
        wel = flopy.modflow.ModflowWel(self.mf, stress_period_data=Q)
        rch = flopy.modflow.ModflowRch(self.mf, rech=rech)
        stress_period_data = {}
        for kper in range(self.nper):
            for kstp in range(self.nstp[kper]):
                stress_period_data[(kper, kstp)] = [
                    "save head",
                    "save drawdown",
                    "save budget",
                    "print head",
                    "print budget",
                ]
        oc = flopy.modflow.ModflowOc(
            self.mf, stress_period_data=stress_period_data, compact=True
        )
        
        self.mf.write_input()   # Write the model input files
        success, buff = self.mf.run_model(silent=True)    # Run the model
        if not success:
            raise Exception("MODFLOW did not terminate normally.")
        
        hds = bf.HeadFile(self.modelname+'.hds')
        times = hds.get_times()
        self.head = hds.get_data(totim=times[-1])
        
        #================計算 Reward=====================================
        SW = action[0]*300
        if self.S < SW:
            SW = self.S
            reward = -2
        else:
            reward,reward1 = 0,0
            WDR = (self.demand - SW - GW)/self.demand
            reward1 = 1-3*abs(WDR)
            if (self.demand - SW - GW) >=-300 and (self.demand - SW - GW)<=0:
                reward1 += 1
            reward+=reward1

        self.sw=SW

        #地下水位不應低於安全水位
        Z = self.head[0,:,:]
        Z[0,:]=Z[1,:]
        Z[-1,:]=Z[-2,:]
        Z[:,-1]=Z[:,-2]
        if Z.min() < 0:
            reward -= 2
        
        self.Zmean=Z.mean()
    
        if self.S/self.max_storage >= 0.9 and action[1]==0:
            reward +=1 

        done = False     
        if self.t==self.t_final-1 :
            done = True
        else:
            self.t += 1

        self.norm_inflow_t_4 = self.norm_inflow_t_3
        self.norm_inflow_t_3 =  self.norm_inflow_t_2
        self.norm_inflow_t_2 = self.norm_inflow_t_1
        self.norm_inflow_t_1 = 2*(self.inflow-np.min(self.rivers)+100)/(np.max(self.rivers)-np.min(self.rivers)+200) -1
           
        self.S = min(self.S-SW+self.inflow, self.max_storage) 
        self.month=((self.t+self.tt)//3)%12+1
        
        scenerio = np.random.choice([0,1,2,3,4,5,6,7,8,9],p=self.river_p)
        self.river = self.rivers[scenerio] if self.t %36==0 else self.river
        d_scenerio = np.random.choice([0,1],p=self.demands_p)
        self.dmd = self.demands[d_scenerio] if self.t %36==0 else self.dmd 
        
        self.demand = self.dmd[(self.t+self.tt)%36]+random.randint(-4,4)*50     
        self.inflow = self.river[(self.t+self.tt)%36] + random.randint(-2,2)*50 

        #state 標準化至[-1,1]
        norm_S = 2*(self.S)/self.max_storage -1
        norm_Zmin =  2*(Z.min() - (-100))/(0-(-100))-1
        norm_month = 2*(self.month-1)/(12-1) -1
        norm_demand = 2*(self.demand-np.min(self.demands)+200)/(np.max(self.demands)-np.min(self.demands)+400) -1
        #norm_inflow = 2*(self.inflow-np.min(self.rivers)+100)/(np.max(self.rivers)-np.min(self.rivers)+600) -1

        return np.array([norm_S, norm_Zmin, norm_demand, self.norm_inflow_t_1, self.norm_inflow_t_2, self.norm_inflow_t_3, self.norm_inflow_t_4 ]), reward, done, {}