from sample_players import DataPlayer
import random

HEURISTICS = [0,1,2,3,4,5,6,7,8,9,10]
AVG_ROUNDS = 20.0

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def __init__(self, player_id, heuristic=0):
        self.round = 0
        
        super().__init__(player_id)
        if heuristic == 0:
            self.heuristic = self.base_score
        elif heuristic == 1:
            self.heuristic = self.agg_score1
        elif heuristic == 2:
            self.heuristic = self.agg_score2
        elif heuristic == 3:
            self.heuristic = self.sq_score1
        elif heuristic == 4:
            self.heuristic = self.sq_score2
        elif heuristic == 5:
            self.heuristic = self.sq_score3
        elif heuristic == 6:
            self.heuristic = self.rnd_score1
        elif heuristic == 7:
            self.heuristic = self.rnd_score2
        elif heuristic == 8:
            self.heuristic = self.rnd_score3
        elif heuristic == 9:
            self.heuristic = self.rnd_score4
        else:
            self.heuristic = self.rnd_score5
    
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        best_move = random.choice(state.actions())
        self.queue.put(best_move)
        
        self.node_count = 0
        if not self.context:
            self.context = {'node': [self.node_count], 'round': self.round}
        else:
            self.context['node'].append(self.node_count)
            self.round = self.context['round']  = self.context['round'] + 1
        self.deepening_ab(state)
        
    def deepening_ab(self, state, depth_limit=8):
        best_move, best_score = None, float("-inf")
        for depth in range(2, depth_limit+1):
            move, score = self.alphabeta(state, depth)
            if score > best_score:
                best_score = score
                best_move = move
                self.queue.put(best_move)
        return best_move
    
    def alphabeta(self, state, depth=4):
        alpha, beta = float("-inf"), float("inf")
        best_move, best_score = None, float("-inf")
        
        for a in state.actions():
            v = self.min_value(state.result(a), alpha, beta, depth)
            alpha = max(alpha, v)
            if v > best_score:
                best_score = v
                best_move = a
                self.queue.put(best_move)
        return best_move, best_score
    
    def min_value(self, state, alpha, beta, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        self.context['node'][-1] = self.node_count = self.node_count+1
        
        if state.terminal_test():
            return state.utility(self.player_id)
        
        if depth <= 0:
            return self.score(state) #state.utility(self.player_id)
        
        v = float("inf")
        for a in state.actions():
            v = min(v, self.max_value(state.result(a), alpha, beta, depth-1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def max_value(self, state, alpha, beta, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        self.context['node'][-1] = self.node_count = self.node_count+1
        
        if state.terminal_test():
            return state.utility(self.player_id)
        
        if depth <= 0:
            return self.score(state)
        
        v = float("-inf")
        for a in state.actions():
            v = max(v, self.min_value(state.result(a), alpha, beta, depth-1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
        
    def score(self, state):
        if self.heuristic:
            return self.heuristic(state)
        else:
            return self.base_score(state)

    def base_score(self, state):
        return self.mult_score(state, 1.0, 1.0)
        
    def agg_score1(self, state):
        return self.mult_score(state, 1.0, 2.0)
    
    def agg_score2(self, state):
        return self.mult_score(state, 1.0, 3.0)
        
    def def_score1(self, state):
        return self.mult_score(state, 2.0, 1.0)
    
    def def_score2(self, state):
        return self.mult_score(state, 3.0, 1.0)
    
    def mult_score(self, state, delta = 1.0, gamma = 2.0):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return delta * len(own_liberties) - gamma * len(opp_liberties)
    
    def sq_score1(self, state):
        return self.sq_score(state, 1.0, 1.0)
        
    def sq_score2(self, state):
        return self.sq_score(state, 1.0, 1.5)
    
    def sq_score3(self, state):
        return self.sq_score(state, 1.0, 2.0)
        
    def sq_score(self, state, delta = 1.0, gamma = 1.0):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return delta * len(own_liberties)**2 - gamma * len(opp_liberties)**2

    def rnd_score1(self, state):
        ratio = self.round / AVG_ROUNDS
        if ratio > 0.5:
            return self.agg_score1(state)
        else:
            return self.def_score1(state)
        
    def rnd_score2(self, state):
        ratio = self.round / AVG_ROUNDS
        if ratio > 0.5:
            return self.agg_score2(state)
        else:
            return self.def_score2(state)
        
    def rnd_score3(self, state):
        ratio = self.round / AVG_ROUNDS
        if ratio > 0.5:
            return self.sq_score(state, 1.0, 1.5)
        else:
            return self.def_score2(state)
    
    def rnd_score4(self, state):
        ratio = self.round / AVG_ROUNDS
        if ratio > 0.5:
            return self.def_score1(state)
        else:
            return self.agg_score1(state)
        
    def rnd_score5(self, state):
        ratio = self.round / AVG_ROUNDS
        if ratio > 0.5:
            return self.def_score2(state)
        else:
            return self.sq_score(state, 1.0, 1.5)