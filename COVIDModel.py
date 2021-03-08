from SocialNetwork import *

class COVIDModel( SocialNetwork ):

    def __init__( self, props ):

        super(COVIDModel, self).__init__( props )

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

    def get_global_prevalence( self ):
        pass

    def get_local_prevalence( self, node ):
        pass

    ## Come back to whether to view certain businesses' risks.
    def get_risk_perception( self, node ):
        ## global prevalence
        ## local prevalence
        pass

    def get_needs_perception( self, node ):
        pass

    def train( self ):
        pass

    def debug( self ):

        cmd = ''

        while cmd != 'q':

            cmd = input( '>>> ' )
            cmdline = cmd.split()

            if cmdline[0] == 'test':
                print( 'The test worked!!!  I am in the inherited class!' )

            else:
                print( 'I am calling the base class instead.' )
                super( COVIDModel, self ).debug( frominherited=True, mycommand=cmdline )
