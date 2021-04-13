from SocialNetwork import *

import matplotlib.pyplot as plt
import networkx          as nx
import numpy             as np
import random            as rnd

import math
import time

from keras.layers import Dense
from keras.models import Sequential, load_model
from statistics   import mean

state_codes = { 'S' : [ 0, 0 ],
                'E' : [ 1, 0 ],
                'I' : [ 0, 1 ],
                'R' : [ 1, 1 ] }
action_codes = { 0 : [ 0, 0 ],
                 1 : [ 1, 0 ],
                 2 : [ 0, 1 ],
                 3 : [ 1, 1 ] }

#def sigmoid( x, factor, shift ):
#    return 1 / ( 1 + ( math.exp( -factor * x + shift ) ) )

def sigmoid( x, factor, shift ):
    s = 1 / ( 1 + ( math.exp( -factor * x + shift ) ) )
    lim = shift / factor * 3
    if x > lim:
        s += ( x - lim ) * .01 * factor
    return s

class COVIDModel( SocialNetwork ):

    def __init__( self, props ):

        super(COVIDModel, self).__init__( props )
        
        if 'file' not in props:

            self.init_mask_wearing()
            self.init_businesses()
            self.init_homes( method='distribution' )
    #        self.assign_work_locations()
            self.init_xy()
            self.init_steps_since()
            
            self._properties['npi'] = 0.

    ## Initialize a list of booleans, one for each node, determining whether
    ## the node is wearing a mask or not.
    def init_mask_wearing( self ):
        if 'wearing' in self._properties:
            self._properties[ 'iswearing' ] = [ self._properties['wearing'] for \
                                                i in range( self._properties['n'] ) ]

    def init_steps_since( self ):
        self._properties['steps_since'] = [ [ 0, 0, 0 ] for i in range( self._properties['n'] ) ]

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
                
    def reset_home_locations( self ):
        
        for i in range( self._properties['n'] ):
            self.update_location( i, self._properties['home_locations'][i] )

    def assign_work_locations( self ):
        
#        print( self._properties['agents_by_location'] )

        n = self._properties['n']
        num_working = int( self._properties['perc_working'] * n )
        self._properties['work_locations'] = [ '' for i in range( n ) ]
        nodes = list( range( n ) )
        rnd.shuffle( nodes )

        ## Assign workplaces to agents
        ## 75% of single person households work
        single_homes = [ key for key in self._properties['agents_by_location'] if \
                         len( self._properties['agents_by_location'][key] ) == 1 ]
        rnd.shuffle( single_homes )
        single_homes = single_homes[:int(.75 * len(single_homes))]
        
        total = 0
        
        for h in single_homes:
            agent = self._properties['agents_by_location'][h][0]
            self._properties['work_locations'][agent] = rnd.choice( self._properties['businesses'] )
            total += 1
            
        multiple_homes = [ key for key in self._properties['agents_by_location'] if \
                           len( self._properties['agents_by_location'][key] ) > 1 ]
        while total < num_working:
            h = rnd.choice( multiple_homes )
            agent = rnd.choice( self._properties['agents_by_location'][h] )
            if self._properties['work_locations'][agent] == '':
                self._properties['work_locations'][agent] = rnd.choice( self._properties['businesses'] )
                total += 1        

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
            dists.append( np.linalg.norm( np.array( my_xy ) - np.array( their_xy ) ) )
            if dists[-1] < mindist:
                mindist = dists[-1]
                minidx = idx

#        print( 'Closest %s: %s\tDistance: %s' % ( business, minidx, mindist ) )
        return minidx

    ## Assign home locations to each node.
    def init_homes( self, method='alone' ):

        ## Create a property to store each agent's home location.
        self._properties['home_locations'] = [ '' for i in range( self._properties['n'] ) ]
        self._properties['agent_locations'] = [ '' for i in range( self._properties['n'] ) ]

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
                    self._properties['agent_locations'][m] = home_code
                    self._properties['agents_by_location'][home_code].append( m )
                count += 1

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
        
        self.act()
        self.update_conditions()
        
    def update_conditions( self ):
        
        for i in range( self._properties['n'] ):
            
            t = self._properties['types'][i]
            p = rnd.random()
            if t == 'E':
                if p < self._properties['e_to_i']:
                    self._properties['types'][i] = 'I'
            elif t == 'I':
                if 'i_to_r' in self._properties:
                    if p < self._properties['i_to_r']:
                        self._properties['types'][i] = 'R'
                elif 'i_to_s' in self._properties:
                    if p < self._properties['i_to_s']:
                        self._properties['types'][i] = 'S'
    
    def assign_interaction_groups( self, nodes ):
        
        ret = {}
        
        ## ret needs to contain lists of interaction groups
        ## The key will be which location they choose to socialize at.
        ## The value will be the indexes of the nodes that are interacting
        ## there.
        ## For now, group size is constant at 2.
        sg = self._graph.subgraph( nodes )
        
        mynodes = list( sg.nodes() )
        myedges = [ e for e in list( sg.edges() ) if e[0] != e[1] ]
        
        while myedges:

            edge = rnd.choice( myedges )
            
            loc1 = self._properties['home_locations'][edge[0]]
            loc2 = self._properties['home_locations'][edge[1]]
            where = rnd.choice( [ loc1, loc2 ] )
            
            self.update_location( edge[0], where )
            self.update_location( edge[1], where )
            
            mynodes.remove( edge[0] )
            mynodes.remove( edge[1] )
            myedges = [ i for i in myedges if (edge[0] not in i) and \
                        (edge[1] not in i) ]
            
            ret[where] = [ edge[0], edge[1] ]
        
        return ret, mynodes

    ## This function definition will override the one in the base class.
    ## This is where we define which agents will act and what they will do.
    ## Choose where to go and then whether to mask.
    def act( self, training=False, eps=1. ):
        
        ## Action codes:
        ##   1 : necessities
        ##   2 : services
        ##   3 : social
        ##   4 : stay home
        
        ## Track statistics with these        
        n = self._properties['n']
        
        demand_idxs = [ -1 for i in range( self._properties['n'] ) ]
        chose_social = []
        
        action_observations = []
        mask_observations   = []
        action_choices      = []
        mask_choices        = []
        start_rewards       = []
        end_rewards         = []
        
        for i in range( n ):
            
            #print('Agent %d action' % i)
            
            start_rewards.append( self.get_reward( i ) )

            if rnd.random() > eps:
                
#                print( 'Getting trained action' )
                
                action_s = self.get_state_for_action( i )
                action_vec = self.action_model.predict( action_s.reshape( 1, len(action_s) ) )
                action_idx = np.argmax( action_vec )
                action_observations.append( action_s )
                
                mask_s = self.get_state_for_masking( i, action_idx )
                mask_vec = self.mask_model.predict( mask_s.reshape( 1, len(mask_s) ) )
                mask_idx = np.argmax( mask_vec )
                mask_observations.append( mask_s )
            
            else:
                
#                print( 'Getting random action' )
                
                action_vec = rnd.choice( [ [1,0,0,0],
                                           [0,1,0,0],
                                           [0,0,1,0],
                                           [0,0,0,1] ] )
                mask_vec = rnd.choice( [ [1,0],
                                         [0,1] ] )
                
                action_idx = np.argmax( action_vec )
                mask_idx = np.argmax( mask_vec )
                
                action_s = self.get_state_for_action( i )
                action_observations.append( action_s )
                
                mask_s = self.get_state_for_masking( i, action_idx )
                mask_observations.append( mask_s )
                
            action_choices.append( action_idx )
            mask_choices.append( mask_idx )
            
#            print('action: ', idx)
            
            ## Find new location based on choice above
            ## idx == 0: necessities
            ## idx == 1: services
            ## idx == 2: social
            ## idx == 3: NOP, stay home
            if action_idx == 0:
                self.update_location( i, self.find_nearest( 'nec', i ) )
                demand_idxs[i] = 0
            elif action_idx == 1:
                self.update_location( i, self.find_nearest( 'serv', i ) )
                demand_idxs[i] = 1
            elif action_idx == 2:
                chose_social.append( i )
            else:
                self.update_location( i, self._properties['home_locations'][i] )
            
            ## Mask if decided to
            if mask_idx == 0:
                self._properties['iswearing'][i] = False
            else:
                self._properties['iswearing'][i] = True
                
            ## END for loop choosing which actions to take
            
        ## Assign any nodes that chose social interactions to interaction pairs.
        ## Some nodes will not be matched -- matching algorithm is random.
        groups, unmatched = self.assign_interaction_groups( chose_social )

        for group in groups:
            for member in groups[group]:
                demand_idxs[member] = 2
        
        ## Execute actions chosen, modify steps_since for each agent
        for i in range( n ):
            
            ## Send any unmatched agents trying to socialize home
            matched = i not in unmatched
            if not matched:
                unmatched.remove( i )
                self.update_location( i, self._properties['home_locations'][i] )
                action_choices[i] = 3
            
            idx = demand_idxs[i]
#            print( 'Agent %d : Action %d' % ( i, idx ) )
            if idx == 0:
                self._properties['steps_since'][i][0] = 0
                self._properties['steps_since'][i][1] += 1
                self._properties['steps_since'][i][2] += 1
            elif idx == 1:
                self._properties['steps_since'][i][0] += 1
                self._properties['steps_since'][i][1] = 0
                self._properties['steps_since'][i][2] += 1
            elif idx == 2 and matched:
                self._properties['steps_since'][i][0] += 1
                self._properties['steps_since'][i][1] += 1
                self._properties['steps_since'][i][2] = 0
            else:
                self._properties['steps_since'][i][0] += 1
                self._properties['steps_since'][i][1] += 1
                self._properties['steps_since'][i][2] += 1
                
            ## With steps_since updated, we can now calculate new demand and
            ## reward values.
            
        for loc in self._properties['agents_by_location']:
            ## have agents at the same location interact
            self.group_interact( loc )
            
        ## Once interactions are finished, we will know new disease states.
        ## This allows us to recalculate reward as a result of the agents'
        ## choices.
        for i in range( n ):
            end_rewards.append( self.get_reward( i ) )
            
        ## Return all information needed to train the models.
        ret = [ action_observations, mask_observations, action_choices,
                mask_choices, start_rewards, end_rewards ]
        
        return ret

    ## This is the function to handle one interaction between agents.
    def interact( self, node1, node2 ):
        
#        print( '%d interacting with %d' % ( node1, node2 ) )
        
        health1 = self._properties['types'][node1]
        health2 = self._properties['types'][node2]
        resistance = 0
        ## if neither person is exposed or infected, there's no transmission
        if ((health1 !=  'E' and health1 != 'I') and (health2 != 'E' and health2 != 'I')):
            return
        ## find the resistance between the two nodes
        if (self._properties['iswearing'][node1]) and (self._properties['iswearing'][node2]):
            resistance = self._properties['mask_to_mask']
        elif (self._properties['iswearing'][node2]) and not (self._properties['iswearing'][node1]):
            resistance = self._properties['nomask_to_mask']
        elif (self._properties['iswearing'][node1]) and not (self._properties['iswearing'][node2]):
            resistance = self._properties['mask_to_nomask']
        else:
            resistance = self._properties['nomask_to_nomask']
            
        ## calculate random number and if it's less than transmission probability, infect someone
        transmit = rnd.random()
        if (transmit <= resistance):
            if (health1 == 'S'):
                self._properties['types'][node1] = 'E'
            elif (health2 == 'S'):
                self._properties['types'][node2] = 'E'
                
    ## Have each pair of colocated agents interact
    def group_interact( self, location ):
        
        agents = self._properties['agents_by_location'][location]
        if len( agents ) in [ 0, 1 ]: return
        for i in range( len( agents ) - 1 ):
            for j in range( 1, len( agents ) ):
                self.interact( agents[i], agents[j] )

    ## This is the function that will deal with friending in between steps,
    ## if applicable.
    def network_effects( self ):
        pass

    def get_state_for_masking( self, node, action ):
        
        t = self._properties['types'][node]
        state = []
        state.extend( state_codes[t] )
        state.append( self.get_risk_perception() )
        if action == 3: state.append( 0 )
        else: state.append( self.get_needs_perception( node )[action] )
        state.extend( action_codes[action] )
        return np.array( state )
    
    def get_state_for_action( self, node ):
        
        state_codes = { 'S' : [ 0, 0 ],
                        'E' : [ 1, 0 ],
                        'I' : [ 0, 1 ],
                        'R' : [ 1, 1 ] }
        
        t = self._properties['types'][node]
        state = []
        state.extend( state_codes[t] )
        state.append( self.get_risk_perception() )
        state.extend( self.get_needs_perception( node ) )
        return np.array( state )
    
    def get_demand( self, node ):
        
        shift = self._properties['half_life']
        steps_since = self._properties['steps_since'][node]
        ## slow, medium, fast
        return [ sigmoid( steps_since[0], 2, shift ),
                 sigmoid( steps_since[1], 1, shift ),
                 sigmoid( steps_since[2], .5, shift ) ]

    def get_reward( self, node ):
        
        ## Get demand levels
        demands = self.get_needs_perception( node )
        
        ## Calculate raw demand score
        r = len( demands ) - sum( demands )
        
        ## Decrease demand in the event of infected state
        if self._properties['types'][node] == 'I': r -= 10.
        
        ## Decrease demand if wearing a mask
        if self._properties['iswearing'][node]: r *= .75
        
        ## Return modified demand, altered by any NPI measures in place.
        return r + self._properties['npi']
    
    def get_average_reward( self, mode='default' ):
        
        nodes = list( range( self._properties['n'] ) )
        if mode == 'mask':
            nodes = [ i for i in range(len(nodes)) if self._properties['types'][i] ]
        elif mode == 'nomask':
            nodes = [ i for i in range(len(nodes)) if not self._properties['types'][i] ]
        if not nodes: return 0.
        return mean( [ self.get_reward(i) for i in nodes ] )
    
    def get_NPI_level( self ):
        ## Need to implement the central controller
        pass

    ## Returns the percentage of nodes in the network of type 'I'.
    def get_global_prevalence( self ):

        num_nodes = self._properties['n']
        types = self._properties['types']
        num_infected = types.count('I') + types.count('E')
        return num_infected / num_nodes

    ## Returns the percentage of nodes in the neighborhood of node of type 'I'.
    def get_local_prevalence( self, node ):

        neighborhood = list( self.get_neighbors( node ) )
        num_nodes = len( neighborhood )
        num_infected = len( [ self._properties['types'][i] for i in neighborhood if \
                              self._properties['types'][i] == 'I' ] )
        return num_infected / num_nodes

    ## Get an aggregate figure for the agent's risk perception.
    def get_risk_perception( self ):
        ## Only using global prevalence right now.
        return self._properties['risk_mod'] * self.get_global_prevalence()

    ## Returns demand vector.
    def get_needs_perception( self, node ):
        return self.get_demand( node )

    def build_learning_model( self ):

        num_actions = 4

        self.action_model = Sequential()
        self.action_model.add( Dense( 256, input_dim=len( self.get_state_for_action(0) ),
                                      activation='relu' ) )
        self.action_model.add( Dense( 256, activation='relu') )
        self.action_model.add( Dense( num_actions, activation='relu') )
        self.action_model.compile( loss='mse', optimizer='adam', metrics=['accuracy'] )
        
        ## Output layer is two nodes, one for mask, one for not.
        self.mask_model = Sequential()
        self.mask_model.add( Dense( 256, input_dim=len( self.get_state_for_masking(0,3) ),
                             activation='relu' ) )
        self.mask_model.add( Dense( 256, activation='relu') )
        self.mask_model.add( Dense( 2, activation='sigmoid') )
        
        self.action_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        self.mask_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    def train( self, num_steps=200, num_episodes=10, animation=False,
               animation_params={} ):
        
        alpha = .5
        decay = .99
        gamma = .9
        
        ## Need to save first graph here so we can reload it each training
        ## episode.        
        self._save( 'training_graph' )
        
        if animation:
            
            ## Check parameters
            if 'mode' not in animation_params:
                animation_params['mode'] = 'network'
            if 'wait' not in animation_params:
                animation_params['wait'] = .01
            
            fig, ax = plt.subplot()
            plt.ion()
            plt.show()
            
        start = time.clock()
        
        for ep in range( num_episodes ):
            
            ## Track statistics with these
            num_nec_interactions    = []
            num_serv_interactions   = []
            num_social_interactions = []
            num_nops                = []
            num_masks               = []
            num_susceptible         = []
            num_exposed             = []
            num_infected            = []
            num_recovered           = []
            avg_rewards             = []
            avg_rewards_mask        = []
            avg_rewards_nomask      = []
            
            epsilon = 1.
            
            act_time = 0
            update_time = 0
            stats_time = 0
            train_time = 0
            
            print( 'Episode: %d' % ep )
            
            init_start = time.clock()
            
            if ep == 0:
                ## On the first episode, initialize the learning model
                self.build_learning_model()
            else:
                ## Each episode thereafter, just reset the graph to its original
                ## state and load the pre-built models.
                self.action_model = load_model( 'action_model' )
                self.mask_model = load_model( 'mask_model' )
                
                self.init_mask_wearing()
                self.init_steps_since()
                self.mix_network()
                self.reset_home_locations()
                
            init_end = time.clock()
            init_dur = init_end - init_start
            print( 'Time to initialize graph: ', init_dur )
                
            stats_start = time.clock()
            num_nec_interactions.append( 0 )
            num_serv_interactions.append( 0 )
            num_social_interactions.append( 0 )
            num_nops.append( 0 )
            num_masks.append( 0 )
            
            num_susceptible.append( self._properties['types'].count( 'S' ) )
            num_exposed.append( self._properties['types'].count( 'E' ) )
            num_infected.append( self._properties['types'].count( 'I' ) )
            num_recovered.append( self._properties['types'].count( 'R' ) )
            
            avg_rewards.append( self.get_average_reward() )
            avg_rewards_mask.append( self.get_average_reward( mode='mask' ) )
            avg_rewards_nomask.append( self.get_average_reward( mode='nomask' ) )
            
            stats_end = time.clock()
            stats_dur = stats_end - stats_start
            stats_time += stats_dur
                
            result_file = open( 'SEIR_Results_ep%d.csv' % ep, 'w' )
            
            for step in range( num_steps ):
                
                act_start = time.clock()
                package = self.act( training=True, eps=epsilon )
                act_end = time.clock()
                act_dur = act_end - act_start
                act_time += act_dur
                
                update_start = time.clock()
                self.update_conditions()
                update_end = time.clock()
                update_dur = update_end - update_start
                update_time += update_dur
                
                stats_start = time.clock()
                
                action_observations = package[0]
                mask_observations   = package[1]
                action_choices      = package[2]
                mask_choices        = package[3]
                start_rewards       = package[4]
                end_rewards         = package[5]
                
                num_nec_interactions.append( package[6] )
                num_serv_interactions.append( package[7] )
                num_social_interactions.append( package[8] )
                num_nops.append( package[9] )
                num_masks.append( sum( mask_choices ) )
                
                num_susceptible.append( self._properties['types'].count( 'S' ) )
                num_exposed.append( self._properties['types'].count( 'E' ) )
                num_infected.append( self._properties['types'].count( 'I' ) )
                num_recovered.append( self._properties['types'].count( 'R' ) )
                
                avg_rewards.append( self.get_average_reward() )
                avg_rewards_mask.append( self.get_average_reward( mode='mask' ) )
                avg_rewards_nomask.append( self.get_average_reward( mode='nomask' ) )
                
                new_action_states = [ self.get_state_for_action( i ) for i in \
                                      range( self._properties['n'] ) ]
                new_mask_states = [ self.get_state_for_masking( i, action_choices[i] ) for i in \
                                    range( self._properties['n'] ) ]
                
                stats_end = time.clock()
                stats_dur = stats_end - stats_start
                stats_time += stats_dur
                
                labels = []
                mask_labels = []
                
                train_start = time.clock()
                
                for i in range( self._properties['n'] ):
                    
                    ## Calculate Q-update for action network
                    ## New value = old value + alpha * temporal difference
                    ## temporal difference = target - old value
                    ## target = current reward + discount factor * estimated optimal reward
                    ## estimated optimal reward = argmax( predict( current state ) )
                    
                    new_state = new_action_states[i]
                    new_pred = self.action_model.predict( new_state.reshape( 1, len( new_state ) ) )
                    est_optimal = np.max( new_pred )
                    target = end_rewards[i] + ( gamma * est_optimal )
                    td = target - start_rewards[i]
                    newval = start_rewards[i] + ( alpha * td )
                    
                    label = new_pred
                    for j in range( len( new_pred ) ):
                        if j == action_choices[i]: label[j] = newval
                        else: label[j] = 0.
                    labels.append( label )
                    
                    new_mask_state = new_mask_states[i]
                    new_mask_pred = self.mask_model.predict( new_mask_state.reshape( 1, len(new_mask_state) ) )
                    est_optimal = np.max( new_mask_pred )
                    target = end_rewards[i] + ( gamma * est_optimal )
                    td = target - start_rewards[i]
                    newval = start_rewards[i] + ( alpha * td )
                    
                    mask_label = new_mask_pred
                    for j in range( len( new_mask_pred ) ):
                        if j == mask_choices[i]: mask_label[j] = newval
                        else: mask_label[j] = 0.
                    mask_labels.append( mask_label )
                    
                self.action_model.fit( np.array( action_observations ).reshape(self._properties['n'], len( action_observations[0] )),
                                       np.array( labels ).reshape(self._properties['n'], 4), epochs=10, verbose=0 )
                self.mask_model.fit( np.array( mask_observations ).reshape(self._properties['n'], len( mask_observations[0] )),
                                     np.array( mask_labels ).reshape(self._properties['n'], 2), epochs=10, verbose=0 )
                
                train_end = time.clock()
                train_dur = train_end - train_start
                train_time += train_dur
                
                epsilon *= decay
                
            end = time.clock()
            dur = end - start
            print( 'Episode duration: %s seconds' % dur )
            
            print( 'Time spent acting: %s' % act_time )
            print( 'Time spent updating: %s' % update_time )
            print( 'Time spent collecting stats: %s' % stats_time )
            print( 'Time spent training: %s' % train_time )
            
            write_start = time.clock()
            
            self.action_model.save( 'action_model' )
            self.mask_model.save( 'mask_model' )
            
            num_susceptible = [ str(k) for k in num_susceptible ]
            num_exposed = [ str(k) for k in num_exposed ]
            num_infected = [ str(k) for k in num_infected ]
            num_recovered = [ str(k) for k in num_recovered ]
            num_nec_interactions = [ str(k) for k in num_nec_interactions ]
            num_serv_interactions = [ str(k) for k in num_serv_interactions ]
            num_social_interactions = [ str(k) for k in num_social_interactions ]
            num_nops = [ str(k) for k in num_nops ]
            num_masks = [ str(k) for k in num_masks ]
            avg_rewards = [ str(k) for k in avg_rewards ]
            avg_rewards_mask = [ str(k) for k in avg_rewards_mask ]
            avg_rewards_nomask = [ str(k) for k in avg_rewards_nomask ]
            
            result_file.write( 'S,' )
            result_file.write( ','.join( num_susceptible ) + '\n' )
            result_file.write( 'E,' )
            result_file.write( ','.join( num_exposed ) + '\n' )
            result_file.write( 'I,' )
            result_file.write( ','.join( num_infected ) + '\n' )
            result_file.write( 'R,' )
            result_file.write( ','.join( num_recovered ) + '\n' )
            
            result_file.write( 'nec,' )
            result_file.write( ','.join( num_nec_interactions ) + '\n' )
            result_file.write( 'serv,' )
            result_file.write( ','.join( num_serv_interactions ) + '\n' )
            result_file.write( 'social,' )
            result_file.write( ','.join( num_social_interactions ) + '\n' )
            result_file.write( 'nop,' )
            result_file.write( ','.join( num_nops ) + '\n' )
            result_file.write( 'masks,' )
            result_file.write( ','.join( num_masks ) + '\n' )
            
            result_file.write( 'rew,' )
            result_file.write( ','.join( avg_rewards ) + '\n' )
            result_file.write( 'rew_mask,' )
            result_file.write( ','.join( avg_rewards_mask ) + '\n' )
            result_file.write( 'rew_nomask,' )
            result_file.write( ','.join( avg_rewards_nomask ) + '\n' )
            
            result_file.close()
            
            write_end = time.clock()
            write_dur = write_end - write_start
            print( 'Time spent writing: %s' % write_dur )
            
    def training_episode( self, newmodel=True, num_steps=1000, ep=0 ):
        
        ## Track statistics with these
        num_nec_mask_interactions    = []
        num_serv_mask_interactions   = []
        num_social_mask_interactions = []
        num_mask_nops                = []
        num_nec_nomask_interactions    = []
        num_serv_nomask_interactions   = []
        num_social_nomask_interactions = []
        num_nomask_nops                = []
        num_susceptible         = []
        num_exposed             = []
        num_infected            = []
        num_recovered           = []
        avg_rewards             = []
        avg_rewards_mask        = []
        avg_rewards_nomask      = []
        
        epsilon = 1.
        alpha = .5
        decay = .99
        gamma = .9
        
        act_time = 0
        update_time = 0
        stats_time = 0
        train_time = 0
        
        start = time.clock()
        
        init_start = time.clock()
            
        if newmodel:
            ## On the first episode, initialize the learning model
            self.build_learning_model()
        else:
            ## Each episode thereafter, just reset the graph to its original
            ## state and load the pre-built models.
            self.action_model = load_model( 'action_model' )
            self.mask_model = load_model( 'mask_model' )
            
            self.init_mask_wearing()
            self.init_steps_since()
            self.mix_network()
            self.reset_home_locations()
            
        init_end = time.clock()
        init_dur = init_end - init_start
        print( 'Time to initialize graph: ', init_dur )
        
        stats_start = time.clock()
        num_nec_mask_interactions.append( 0 )
        num_serv_mask_interactions.append( 0 )
        num_social_mask_interactions.append( 0 )
        num_mask_nops.append( 0 )
        num_nec_nomask_interactions.append( 0 )
        num_serv_nomask_interactions.append( 0 )
        num_social_nomask_interactions.append( 0 )
        num_nomask_nops.append( 0 )
        
        num_susceptible.append( self._properties['types'].count( 'S' ) )
        num_exposed.append( self._properties['types'].count( 'E' ) )
        num_infected.append( self._properties['types'].count( 'I' ) )
        num_recovered.append( self._properties['types'].count( 'R' ) )
        
        avg_rewards.append( self.get_average_reward() )
        avg_rewards_mask.append( self.get_average_reward( mode='mask' ) )
        avg_rewards_nomask.append( self.get_average_reward( mode='nomask' ) )
        
        stats_end = time.clock()
        stats_dur = stats_end - stats_start
        stats_time += stats_dur
        
        result_file = open( 'current_results\\10penalty_SEIRS_Results_ep%d.csv' % ep, 'w' )
        
        for step in range( num_steps ):
                
            act_start = time.clock()
            package = self.act( training=True, eps=epsilon )
            act_end = time.clock()
            act_dur = act_end - act_start
            act_time += act_dur
            
            update_start = time.clock()
            self.update_conditions()
            update_end = time.clock()
            update_dur = update_end - update_start
            update_time += update_dur
            
            stats_start = time.clock()
            
            action_observations = package[0]
            mask_observations   = package[1]
            action_choices      = package[2]
            mask_choices        = package[3]
            start_rewards       = package[4]
            end_rewards         = package[5]
            
            mask_nec    = 0
            nomask_nec  = 0
            mask_serv   = 0
            nomask_serv = 0
            mask_soc    = 0
            nomask_soc  = 0
            mask_nop    = 0
            nomask_nop  = 0
            
            for i in range( self._properties['n'] ):
                
                if mask_choices[i] == 0:
                    
                    if action_choices[i] == 0:
                        nomask_nec += 1
                    elif action_choices[i] == 1:
                        nomask_serv += 1
                    elif action_choices[i] == 2:
                        nomask_soc += 1
                    elif action_choices[i] == 3:
                        nomask_nop += 1
                        
                elif mask_choices[i] == 1:
                    
                    if action_choices[i] == 0:
                        mask_nec += 1
                    elif action_choices[i] == 1:
                        mask_serv += 1
                    elif action_choices[i] == 2:
                        mask_soc += 1
                    elif action_choices[i] == 3:
                        mask_nop += 1
            
            num_nec_mask_interactions.append( mask_nec )
            num_serv_mask_interactions.append( mask_serv )
            num_social_mask_interactions.append( mask_soc )
            num_mask_nops.append( mask_nop )
            num_nec_nomask_interactions.append( nomask_nec )
            num_serv_nomask_interactions.append( nomask_serv )
            num_social_nomask_interactions.append( nomask_soc )
            num_nomask_nops.append( nomask_nop )
            
            num_susceptible.append( self._properties['types'].count( 'S' ) )
            num_exposed.append( self._properties['types'].count( 'E' ) )
            num_infected.append( self._properties['types'].count( 'I' ) )
            num_recovered.append( self._properties['types'].count( 'R' ) )
            
            avg_rewards.append( self.get_average_reward() )
            avg_rewards_mask.append( self.get_average_reward( mode='mask' ) )
            avg_rewards_nomask.append( self.get_average_reward( mode='nomask' ) )
            
            new_action_states = [ self.get_state_for_action( i ) for i in \
                                  range( self._properties['n'] ) ]
            new_mask_states = [ self.get_state_for_masking( i, action_choices[i] ) for i in \
                                range( self._properties['n'] ) ]
            
            stats_end = time.clock()
            stats_dur = stats_end - stats_start
            stats_time += stats_dur
            
            labels = []
            mask_labels = []
            
            train_start = time.clock()
            
            for i in range( self._properties['n'] ):
                
                ## Calculate Q-update for action network
                ## New value = old value + alpha * temporal difference
                ## temporal difference = target - old value
                ## target = current reward + discount factor * estimated optimal reward
                ## estimated optimal reward = argmax( predict( current state ) )
                
                new_state = new_action_states[i]
                new_pred = self.action_model.predict( new_state.reshape( 1, len( new_state ) ) )
                est_optimal = np.max( new_pred )
                target = end_rewards[i] + ( gamma * est_optimal )
                td = target - start_rewards[i]
                newval = start_rewards[i] + ( alpha * td )
                
                label = new_pred
                for j in range( len( new_pred ) ):
                    if j == action_choices[i]: label[j] = newval
                    else: label[j] = 0.
                labels.append( label )
                
                new_mask_state = new_mask_states[i]
                new_mask_pred = self.mask_model.predict( new_mask_state.reshape( 1, len(new_mask_state) ) )
                est_optimal = np.max( new_mask_pred )
                target = end_rewards[i] + ( gamma * est_optimal )
                td = target - start_rewards[i]
                newval = start_rewards[i] + ( alpha * td )
                
                mask_label = new_mask_pred
                for j in range( len( new_mask_pred ) ):
                    if j == mask_choices[i]: mask_label[j] = newval
                    else: mask_label[j] = 0.
                mask_labels.append( mask_label )
                
            self.action_model.fit( np.array( action_observations ).reshape(self._properties['n'], len( action_observations[0] )),
                                   np.array( labels ).reshape(self._properties['n'], 4), epochs=10, verbose=0 )
            self.mask_model.fit( np.array( mask_observations ).reshape(self._properties['n'], len( mask_observations[0] )),
                                 np.array( mask_labels ).reshape(self._properties['n'], 2), epochs=10, verbose=0 )
            
            train_end = time.clock()
            train_dur = train_end - train_start
            train_time += train_dur
            
            epsilon *= decay
            
        end = time.clock()
        dur = end - start
        print( 'Episode duration: %s seconds' % dur )
        
        print( 'Time spent acting: %s' % act_time )
        print( 'Time spent updating: %s' % update_time )
        print( 'Time spent collecting stats: %s' % stats_time )
        print( 'Time spent training: %s' % train_time )
        
        write_start = time.clock()
        
        self.action_model.save( 'action_model' )
        self.mask_model.save( 'mask_model' )
        
        num_susceptible = [ str(k) for k in num_susceptible ]
        num_exposed = [ str(k) for k in num_exposed ]
        num_infected = [ str(k) for k in num_infected ]
        num_recovered = [ str(k) for k in num_recovered ]
        num_nec_mask_interactions = [ str(k) for k in num_nec_mask_interactions ]
        num_serv_mask_interactions = [ str(k) for k in num_serv_mask_interactions ]
        num_social_mask_interactions = [ str(k) for k in num_social_mask_interactions ]
        num_mask_nops = [ str(k) for k in num_mask_nops ]
        num_nec_nomask_interactions = [ str(k) for k in num_nec_nomask_interactions ]
        num_serv_nomask_interactions = [ str(k) for k in num_serv_nomask_interactions ]
        num_social_nomask_interactions = [ str(k) for k in num_social_nomask_interactions ]
        num_nomask_nops = [ str(k) for k in num_nomask_nops ]
        avg_rewards = [ str(k) for k in avg_rewards ]
        avg_rewards_mask = [ str(k) for k in avg_rewards_mask ]
        avg_rewards_nomask = [ str(k) for k in avg_rewards_nomask ]
        
        result_file.write( 'S,' )
        result_file.write( ','.join( num_susceptible ) + '\n' )
        result_file.write( 'E,' )
        result_file.write( ','.join( num_exposed ) + '\n' )
        result_file.write( 'I,' )
        result_file.write( ','.join( num_infected ) + '\n' )
        result_file.write( 'R,' )
        result_file.write( ','.join( num_recovered ) + '\n' )
        
        result_file.write( 'nec_mask,' )
        result_file.write( ','.join( num_nec_mask_interactions ) + '\n' )
        result_file.write( 'serv_mask,' )
        result_file.write( ','.join( num_serv_mask_interactions ) + '\n' )
        result_file.write( 'social_mask,' )
        result_file.write( ','.join( num_social_mask_interactions ) + '\n' )
        result_file.write( 'nop_mask,' )
        result_file.write( ','.join( num_mask_nops ) + '\n' )
        result_file.write( 'nec_nomask,' )
        result_file.write( ','.join( num_nec_nomask_interactions ) + '\n' )
        result_file.write( 'serv_nomask,' )
        result_file.write( ','.join( num_serv_nomask_interactions ) + '\n' )
        result_file.write( 'social_nomask,' )
        result_file.write( ','.join( num_social_nomask_interactions ) + '\n' )
        result_file.write( 'nop_nomask,' )
        result_file.write( ','.join( num_nomask_nops ) + '\n' )
        
        result_file.write( 'rew,' )
        result_file.write( ','.join( avg_rewards ) + '\n' )
        result_file.write( 'rew_mask,' )
        result_file.write( ','.join( avg_rewards_mask ) + '\n' )
        result_file.write( 'rew_nomask,' )
        result_file.write( ','.join( avg_rewards_nomask ) + '\n' )
        
        result_file.close()
        
        write_end = time.clock()
        write_dur = write_end - write_start
        print( 'Time spent writing: %s' % write_dur )

    def debug( self ):

        cmd = ''

        while cmd != 'q':

            cmd = input( '>>> ' )
            
            ## Quit
            if cmd == 'q': return
            
            cmdline = cmd.split()
            numtokens = len( cmdline )
            allnodes = list( range( self._properties['n'] ) )
            
            ## Go through one or more time steps
            if cmdline[0] == 'c':
                if numtokens == 1:
                    print( 'Updating 1 step.' )
                    self.update()
                if numtokens > 1:
                    try:
                        steps = int( cmdline[1] )
                        print( 'Updating %d steps.' % steps )
                        for i in range( steps ): self.update()
                    except:
                        print( 'Usage: c <number of steps>' )
                continue

            ## Print out a list of admissible commands
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

            ## General display command.  Details depend on the argument.
            if cmdline[0] == 'show':

                ## Show who is wearing masks at the current time step
                if cmdline[1] == 'iswearing':
                    if numtokens == 2: mylist = allnodes
                    else: mylist = [ int(i) for i in cmdline[2:] ]
                    for i in mylist:
                        print( 'Node %d wearing mask: %s' % ( i, self._properties['iswearing'][i] ) )

                ## Show locations of homes / businesses
                elif cmdline[1] == 'locations':

                    ## Show all locations
                    if numtokens == 2:
                        for i in self._properties['home_locations']:
                            print( '%s:    x = %s ; y = %s' % ( i, *self._properties['locations'][i] ) )
                        for i in self._properties['businesses']:
                            print( '%s:    x = %s ; y = %s' % ( i, *self._properties['locations'][i] ) )

                    ## Show only locations of specified types
                    else:
                        mylist = cmdline[2:]
                        for t in mylist:
                            if t == 'HOME':
                                for i in self._properties['home_locations']:
                                    loc = self._properties['locations'][i]
                                    print( '%s:    x = %s ; y = %s' % ( i, *loc ) )
                            else:
                                blist = [ i for i in self._properties['businesses'] if t in i ]
                                for i in blist:
                                    loc = self._properties['locations'][i]
                                    print( '%s:    x = %s ; y = %s' % ( i, *loc ) )

                ## Show contents of dictionaries for different agents
                elif cmdline[1] in [ 'home_locations', 'agent_locations',
                                     'agent_xy', 'steps_since' ]:
                    
                    if numtokens == 2: mylist = allnodes
                    else: mylist = [ int(i) for i in cmdline[2:] ]

                    for b in mylist:
                        print( '%d: %s' % ( b, self._properties[cmdline[1]][b] ) )

                ## Show all locations and any agents in them
                elif cmdline[1] == 'agents_by_location':
                    
                    if numtokens == 2:
                        mylist = [ loc for loc in self._properties['agents_by_location'] ]
                    else:
                        mylist = cmdline[2:]

                    for loc in mylist:
                        curr_occ = self._properties['agents_by_location'][loc]
                        if not curr_occ: continue
                        print( '[%s]: %s' % ( loc, curr_occ ) )

                ## Show a histogram of number of houses by occupancy
                elif cmdline[1] == 'housing_distribution':
                    self.show_housing_distribution()
                    
                ## Draw a plot of the social network
                elif cmdline[1] == 'network':
                    self.show_network()
                    
                ## Show current reward values for certain nodes
                elif cmdline[1] == 'reward':
                    
                    if numtokens == 2: mylist = allnodes
                    else: mylist = [ int(i) for i in cmdline[2:] ]
                    
                    for node in mylist:
                        print( 'Reward %d:' % node, self.get_reward( node ) )
                
                ## Show current demand values for certain nodes
                elif cmdline[1] == 'demand':
                    
                    if numtokens == 2: mylist = allnodes
                    else: mylist = [ int(i) for i in cmdline[2:] ]
                    
                    for node in mylist:
                        print( 'Demand %d:' % node, self.get_demand( node ) )
                        
                ## Show current risk values for certain nodes
                elif cmdline[1] == 'risk':
                    
                    if numtokens == 2: mylist = allnodes
                    else: mylist = [ int(i) for i in cmdline[2:] ]
                    
                    print( 'Risk:', self.get_risk_perception() )
                    
                ## Show current infected/exposed percentage
                elif cmdline[1] == 'global':
                    print( 'Percent infected:', self.get_global_prevalence() )

                ## Pass command line on to base class method
                else:
                    super( COVIDModel, self ).debug( frominherited=True, mycommand=cmdline )

            ## Set the value of a parameter.
            ## BE CAREFUL.
            elif cmdline[0] == 'set':

                if cmdline[1] == 'location':
                    self.update_location( int( cmdline[2] ), cmdline[3] )
                    
                else:
                    print( 'Setting property [%s] to value [%s]' % (cmdline[1],
                                                                    cmdline[2]) )
                    if cmdline[1] in ['half_life', 'i_to_r', 'e_to_i',
                                      'mask_to_mask', 'mask_to_nomask',
                                      'nomask_to_mask', 'nomask_to_nomask',
                                      'risk_mod']:
                        self._properties[ cmdline[1] ] = float( cmdline[2] )
                    else:
                        self._properties[ cmdline[1] ] = cmdline[2]

            elif cmdline[0] == 'find_nearest':
                minidx = self.find_nearest( cmdline[1], int( cmdline[2] ) )
                print( 'Closest %s: %s' % ( cmdline[1], minidx ) )

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
        
    def show_network( self ):
        
        mycolors = []
        
        if 'type_dist' in self._properties:
            colors = {}
            
#            for t in self._properties['type_dist']:
#                colors[t] = ( rnd.random(), rnd.random(), rnd.random(), 0.5 )
            
            colors = { 'S' : 'b',
                       'E' : 'y',
                       'I' : 'r',
                       'R' : 'g' }
            
            for i in range( len( self._properties['types'] ) ):
                mycolors.append( colors[self._properties['types'][i]] )
                
        else:
            mycolors = [ 'b' for i in range( self._properties['n'] ) ]
            
        nx.draw_networkx( self._graph, node_color=mycolors )
        plt.show()