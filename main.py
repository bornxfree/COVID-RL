from COVIDModel import COVIDModel
import matplotlib.pyplot as plt

import random

def main():

    props = { 'n': 100,
              'topology':'scale free',
              'saturation':.5,
              'dimensions':1,
              'weight': 1.,
              'unfriend': 1.,
              'friend': .1,
              'update': 1.,
              'type_dist': {'S':.95,'E':.0,'I':.05,'R':.0},
              'resistance_param': .0,
              'file': 'mytest.xml'}

    CM = COVIDModel( props )

    CM.debug()

if __name__ == '__main__':
    main()
