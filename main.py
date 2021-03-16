## Housing size distribution retrieved from https://www.statista.com/statistics/242189/disitribution-of-households-in-the-us-by-household-size/

from COVIDModel import COVIDModel
import matplotlib.pyplot as plt

import random

def main():

    props = { 'n': 200,
              'topology':'scale free',
              'saturation':.5,
              'dimensions':1,
              'weight': 1.,
              'unfriend': 1.,
              'friend': .1,
              'update': 1.,
              'type_dist': {'S':.95,'E':.0,'I':.05,'R':.0},
              'housing_dist': {1: .28, 2: .35, 3: .15,
                               4: .12, 5: .05, 6: .02,
                               7: .02, 8:.01},
              'resistance_param': .0,
              'wearing':False,
              'transmit':.05,
              'recover':.1,
              'num_businesses':20,
              'max_cohabitation':8,
              'business_type_dist':{'SCHOOL':.25,
                                    'HOSPITAL':.25,
                                    'SHOP':.25,
                                    'GROCERY':.25
                                    }
              }

    CM = COVIDModel( props )

    CM.debug()

if __name__ == '__main__':
    main()
