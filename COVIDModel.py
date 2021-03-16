from SocialNetwork import *
import random as rnd
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense

class COVIDModel( SocialNetwork ):

    def __init__( self, props ):

        super(COVIDModel, self).__init__( props )

        self.init_mask_wearing()
        self.init_businesses()
        self.assign_work_locations()
        self.init_homes( method='distribution' )
        self.init_xy()

    ## Initialize a list of booleans, one for each node, determining whether
    ## the node is wearing a mask or not.
    def init_mask_wearing( self ):
        if 'wearing' in self._properties:
            self._properties[ 'iswearing' ] = [ self._properties['wearing'] for \
                                                i in range( self._properties['n'] ) ]

    ## Initialize all the businesses in the network.
    def init_businesses( self ):
        
        ## This only executes if there is a dictionary passed in the
        ## 'business_type_dist' property.
        if 'business_type_dist' in self._properties:
            d = self._properties['business_type_dist']

            ## Make sure proportions sum to 1.
            if abs( sum( [ d[key] for key in d ] ) - 1. ) > .000001:
                print( 'Business type proportions must sum to 1.' )
                return

            num = self._properties['num_businesses']
            self._properties['businesses'] = []
            self._properties['agents_by_location'] = {}
            
            ## A small bit of extra machinery to make sure that there are no
            ## disparities in the number of businesses because of rounding.
            maxtype = None
            maxprop = 0.0
            nums = {}
            total = 0
            for key in d:

                if d[key] > maxprop:
                    maxprop = d[key]
                    maxtype = key
                
                num_curr = int( d[key] * num )
                nums[key] = num_curr
                for i in range( num_curr ):
                    self._properties['businesses'].append( '%s_%d' % ( key, i ) )
                    total += 1

            while total < num:
                nextnum = nums[maxtype]
                self._properties['businesses'].append( '%s_%d' % ( maxtype, nextnum ) )
                nums[maxtype] += 1
                total += 1

            self._properties['businesses'].sort()

            ## Build a dictionary with keys = location names and values = lists of node indexes (integers).
            ## Need to update this each time an agent changes location.
            for loc in self._properties['businesses']:
                self._properties['agents_by_location'][loc] = []

    def assign_work_locations( self ):

        self._properties['work_locations'] = []

        ## Assign workplaces to agents
        for i in range( self._properties['n'] ):                
            pass

    def find_nearest( self, business, node ):

        dists = []
        my_xy = self._properties['agent_xy'][node]

        bs = [ b for b in self._properties['businesses'] if business in b ]

        mindist = 999999999
        minidx = -1
        
        ## Iterate through businesses of the correct type, calculating distances
        ## on the way.
        for idx in bs:
            their_xy = self._properties['locations'][idx]
            dists.append( np.linalg.norm( my_xy - their_xy ) )
            if dists[-1] < mindist:
                mindist = dists[-1]
                minidx = idx

        print( 'Closest %s: %s\tDistance: %s' % ( business, minidx, mindist ) )

    ## Assign home locations to each node.
    def init_homes( self, method='alone' ):

        ## Create a property to store each agent's home location.
        self._properties['home_locations'] = [ '' for i in range( self._properties['n'] ) ]

        ## Assign all nodes a unique home location.
        if method == 'alone':
            for i in range( self._properties['n'] ):
                 self._properties['home_locations'].append( 'HOME_%d' % i )
                 self._properties['agents_by_location']['HOME_%d' % i] = [i]

        ## Assign nodes homes based on a housing distribution provided in the
        ## property dictionary.  The distribution should have keys that
        ## represent how many people live in a house (1, 2, 3...), and values
        ## that say what percentage of homes should have that number of
        ## occupants.
        elif method == 'distribution':

            d = self._properties['housing_dist']

            if abs( sum( [ d[key] for key in d ] ) - 1. ) > .000001:
                print( 'Home occupancy proportions must sum to 1.' )
                return

            ## Make a list of occupancies, weighted proportional to the
            ## numbers in the provided distribution.
            occupancies = []
            for key in d:
                num = int( d[key] * 100 )
                occupancies.extend( [ key for i in range( num ) ] )
            rnd.shuffle( occupancies )
            
            ## Start plucking nodes off the list and putting them each in
            ## houses until none are left.
            allnodes = [ i for i in range( self._properties['n'] ) ]
            count = 0
            while allnodes:
                currnum = rnd.choice( occupancies )
                members = set()
                if currnum < len(allnodes):
                    while len( members ) < currnum:
                        members.add( rnd.choice( allnodes ) )
                else:
                    members = set( allnodes )
                home_code = 'HOME_%d' % count
                self._properties['agents_by_location'][home_code] = []
                for m in members:
                    allnodes.remove( m )
                    self._properties['home_locations'][m] = home_code
                    self._properties['agents_by_location'][home_code].append( m )
                count += 1

        ## Set each agent's starting location to its home.
        self._properties['agent_locations'] = self._properties['home_locations']

        ## Make sure that agents are connected to all others with whome they
        ## live.
        for loc in self._properties['home_locations']:
            mylist = self._properties['agents_by_location'][loc]
            self.close_group( mylist )

    ## Create a triadic closure among the nodes in nodelist.
    def close_group( self, nodelist ):

        if len( nodelist ) == 1: return
        for i in range( len( nodelist ) - 1 ):
            for j in range( i, len( nodelist ) ):
                self.connect( nodelist[i], nodelist[j] )

    ## Assign 2D coordinates to all nodes.
    def init_xy( self ):
        
        ## This is a dictionary with keys = place names (businesses and homes)
        ## and values = two-element lists containing x-y coordinates.
        self._properties['locations'] = {}
        
        ## This is a list keeping track of each agent's current x-y position.
        self._properties['agent_xy'] = [ np.array( [ rnd.random(), rnd.random() ] ) for \
                                         i in range( self._properties['n'] ) ]

        ## Assign fixed locations to each business.
        for b in self._properties['businesses']:
            self._properties['locations'][b] = np.array( [ rnd.random(), rnd.random() ] )

        ## Assign fixed locations to each home.
        for h in self._properties['home_locations']:
            node = int( h.split('_')[1] )
            self._properties['locations'][h] = self._properties['agent_xy'][node]

    ## Change the location of node to newloc.
    def update_location( self, node, newloc ):
        
        ## Check that the location exists.
        keys = [ key for key in self._properties['locations'] ]
        if newloc not in keys:
            print( 'No such location: [%s]' % newloc )
            return
        
        try:
            ## Update node's location, moving it from its old place to the new
            ## one.  Then change node's x-y coordinates to those of the new
            ## location.
            curr_location = self._properties['agent_locations'][node]
            self._properties['agents_by_location'][curr_location].remove( node )
            self._properties['agent_locations'][node] = newloc
            self._properties['agents_by_location'][newloc].append( node )
            self._properties['agent_xy'][node] = self._properties['locations'][newloc]
        except:
            print( 'Problem encountered removing node from its current location.' )

    ## This function will be called to update the overall state of the network.
    ## Mostly just a wrapper for other functions.
    def update( self ):
        pass

    ## This function definition will override the one in the base class.
    ## This is where we define which agents will act and what they will do.
    ## Choose where to go and then whether to mask.
    def act( self ):
        pass

    ## This is the function to handle one interaction between agents.
    def interact( self, node1, node2 ):
        pass

    ## This is the function that will deal with friending in between steps,
    ## if applicable.
    def network_effects( self ):
        pass

    ## This function needs to return a vector representing the node's current
    ## state.
    def get_state( self, node ):
        pass

    def get_reward( self, node ):
        ## Homophily
        ## Satisfaction of basic needs
        ## Satisfaction of social needs
        ## Non-material needs (doctor)
        pass

    ## Returns the percentage of nodes in the network of type 'I'.
    def get_global_prevalence( self ):

        num_nodes = self._properties['n']
        num_infected = self._properties['types'].count('I')
        return num_infected / num_nodes

    ## Returns the percentage of nodes in the neighborhood of node of type 'I'.
    def get_local_prevalence( self, node ):

        neighborhood = list( self.get_neighbors( node ) )
        num_nodes = len( neighborhood )
        num_infected = len( [ self._properties['types'][i] for i in neighborhood if \
                              self._properties['types'][i] == 'I' ] )
        return num_infected / num_nodes

    ## Come back to whether to view certain businesses' risks.
    def get_risk_perception( self, node ):
        ## global prevalence
        ## local prevalence
        pass

    def get_needs_perception( self, node ):
        pass

    def build_learning_model( self ):

        num_features = len( self.get_state( 0 ) )
        num_actions = 1

        self.model = Sequential()
        self.model.add(Dense(256, input_dim=num_features, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(num_actions, activation='relu'))
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    def train( self ):

        self.build_learning_model()

    def debug( self ):

        cmd = ''

        while cmd != 'q':

            cmd = input( '>>> ' )

            if cmd == 'q': return
            
            cmdline = cmd.split()

            if cmdline[0] == 'help':

                print( 'COVIDModel Commands:' )
                print( '\tshow iswearing <list of nodes (optional)>' )
                print( '\tshow locations <list of location types (optional)>' )
                print( '\tshow businesses <list of business types (optional)>' )
                print( '\tshow housing_distribution' )
                print( '\tshow home_locations <list of nodes (optional)>' )
                print( '\tshow agent_locations <list of nodes (optional)>' )
                print( '\tshow agent_xy <list of nodes (optional)>' )
                print( '\tshow agents_by_location <list of locations (optional)>' )
                print( '\tset location <node> <location>' )
                print( '\tfind_nearest <location type> <node>' )

                super( COVIDModel, self ).debug( frominherited=True, mycommand=cmdline )
                continue

            if cmdline[0] == 'draw':
                nx.draw_networkx( self._graph )
                plt.show()
                continue

            if len(cmdline) > 2:
                try:
                    mylist = [int(i) for i in cmdline[2:]]
                except:
                    mylist = cmdline[2:]
            else:
                if cmdline[1] == 'agents_by_location':
                    mylist = [ key for key in self._properties['agents_by_location'] ]
                elif cmdline[1] == 'businesses':
                    mylist = list( range( self._properties['num_businesses'] ) )
                else:
                    mylist = list( range( self._properties['n'] ) )

            if cmdline[0] == 'show':

                if cmdline[1] == 'iswearing':

                    for i in mylist:
                        print( 'Node %d wearing mask: %s' % ( i, self._properties['iswearing'][i] ) )

                elif cmdline[1] == 'locations':

                    if len( cmdline ) == 2:
                        for i in self._properties['home_locations']:
                            print( '%s:    x = %s ; y = %s' % ( i, *self._properties['locations'][i] ) )

                        for i in self._properties['businesses']:
                            print( '%s:    x = %s ; y = %s' % ( i, *self._properties['locations'][i] ) )

                    else:
                        mylist = cmdline[2:]
                        for t in mylist:
                            if t == 'HOME':
                                for i in self._properties['home_locations']:
                                    loc = self._properties['locations'][i]
                                    print( '%s:    x = %s ; y = %s' % ( i, loc[0], loc[1] ) )
                            else:
                                blist = [ i for i in self._properties['businesses'] if t in i ]
                                for i in blist:
                                    loc = self._properties['locations'][i]
                                    print( '%s:    x = %s ; y = %s' % ( i, loc[0], loc[1] ) )

                elif cmdline[1] in [ 'businesses', 'home_locations', 'agent_locations',
                                     'agent_xy' ]:

                    for b in mylist:
                        print( '%d: %s' % ( b, self._properties[cmdline[1]][b] ) )

                elif cmdline[1] == 'agents_by_location':

                    for loc in mylist:
                        print( '[%s]: %s' % ( loc, self._properties['agents_by_location'][loc] ) )

                elif cmdline[1] == 'housing_distribution':
                    
                    self.show_housing_distribution()

                else:
                    super( COVIDModel, self ).debug( frominherited=True, mycommand=cmdline )

            elif cmdline[0] == 'set':

                if cmdline[1] == 'location':
                    self.update_location( int( cmdline[2] ), cmdline[3] )

            elif cmdline[0] == 'find_nearest':

                self.find_nearest( cmdline[1], int( cmdline[2] ) )

            else:
                super( COVIDModel, self ).debug( frominherited=True, mycommand=cmdline )

    def show_housing_distribution( self ):
        nums = {}
        for loc in self._properties['agents_by_location']:
            if loc.split('_')[0] == 'HOME':
                num = len( self._properties['agents_by_location'][loc] )
                if num in nums:
                    nums[num] += 1
                else:
                    nums[num] = 1
        keys = sorted( [i for i in nums] )
        plt.bar( keys, [nums[key] for key in keys] )
        plt.show()
