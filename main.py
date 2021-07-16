## Housing size distribution retrieved from https://www.statista.com/statistics/242189/disitribution-of-households-in-the-us-by-household-size/
## Students make up 25% of the US population https://www.census.gov/newsroom/press-releases/2019/school-enrollment.html#:~:text=3%2C%202019%20%E2%80%94%20The%20number%20of,population%20age%203%20and%20older.
## Workers make up ~57% of the US population https://www.statista.com/statistics/192398/employment-rate-in-the-us-since-1990/#:~:text=The%20employment%2Dpopulation%20ratio%20represents,rate%20stood%20at%2056.8%20percent.

from COVIDModel import COVIDModel
import matplotlib.pyplot as plt

import random

def main():

    props = { 'n': 250,
              'topology':'scale free',
              'saturation':.05,
              'dimensions':1,
              'weight': 1.,
              'unfriend': .0,
              'friend': .0,
              'update': 1.,
              'type_dist': {'S': .98, 'E':.02, 'I':.0, 'R':.0},
              'housing_dist': {1: .28, 2: .35, 3: .15,
                               4: .12, 5: .05, 6: .02,
                               7: .02, 8:.01},
              'resistance_param': .0,
              'wearing': False,
              'recover': .1,
              'num_businesses': 4,
              'max_cohabitation': 8,
              'business_type_dist': { 'nec' : 0.5,
                                      'serv' : 0.5
                                    },
              'half_life': 10,
              'perc_working': .7,
              'i_to_s': .1813,
              'e_to_i': .1813,
              'mask_to_mask':.01,
              'mask_to_nomask':.03,
              'nomask_to_mask':.01,
              'nomask_to_nomask':.03,
#              'mask_to_mask':.0,
#              'mask_to_nomask':.0,
#              'nomask_to_mask':.0,
#              'nomask_to_nomask':.0,
              'risk_mod':1.,
              'inf_punishment':100,
              'delta':.5
#              'file': 'test.xml'
              }

    CM = COVIDModel( props )
    CM._save('learningtest')
    props['file'] = 'learningtest.xml'
    
    numtests = 50
    policies = ['control1','control2','control3']
    for pol in policies:
        for i in range( numtests ):
            del CM
            CM = COVIDModel( props )
            CM.training_episode(newmodel=False, ep=i, epsilon=1., policy=pol)

#    epsilon = 1.
#    for i in range(175,200):
#        del CM
#        CM = COVIDModel( props )
#        epsilon = 1. * ( .95 ** i )
#        if i > 0: CM.training_episode(newmodel=False, ep=i, epsilon=epsilon)
#        else: CM.training_episode(newmodel=True, ep=i, epsilon=epsilon)

if __name__ == '__main__':
    main()
