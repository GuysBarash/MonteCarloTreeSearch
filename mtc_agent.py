# Name: Guy Barash
# ID  : 301894234
import hashlib
import json
import math
import os
import pickle
import random
import sys
import numpy as np
from datetime import datetime

import pddlsim
from pddlsim.local_simulator import LocalSimulator

from tqdm import tqdm
import time


def clear_policy(path):
    if os.path.exists(path):
        os.remove(path)


def get_id_for_dict(dict):
    unique_str = ''.join(["'%s':'%s';" % (key, sorted(list(val))) for (key, val) in sorted(dict.items())])
    return hashlib.sha1(unique_str).hexdigest()


current_node_id = 0


def get_node_id():
    global current_node_id

    current_node_id += 1
    return current_node_id


class treeNode:
    def __init__(self, state, parent=None, originating_action=None, terminal=False, exploration_constant=None):
        self.exploraction_constant = exploration_constant
        self.node_id = get_node_id()
        self.state = state
        self.state_sig = None
        self.originating_action = originating_action
        self.isTerminal = terminal
        self.isDeadEnd = False
        self.goalRemain = 10000
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.parent_sig = None
        self.numVisits = 0
        self.totalReward = 0
        self.wins = 0
        self.dead_ends = 0
        self.total_steps = 0
        self.actions = dict()
        self.depth = 0


class MyExecutor(object):

    def get_scores_of_node(self, node):
        actions = list()
        scores = list()
        rewards = list()
        child_nodes = list()
        for action, child_sig in node.children_sig.items():
            child = self.mtc_guide.get(child_sig)
            if node.numVisits <= 0:
                reward = 0
                score = float(2 ** 30)

            elif child.numVisits <= 0:
                reward = 0
                score = float(2 ** 30)
            else:
                reward = float(child.totalReward) / child.numVisits
                exploration_factor = math.sqrt(math.log(node.numVisits) / child.numVisits)
                score = reward + (self.mtc_exploration_constant * exploration_factor)

            child_nodes.append(child)
            actions.append(action)
            scores.append(score)
            rewards.append(reward)

        return child_nodes, actions, scores, rewards

    def __init__(self, exploration_rate_start=None, policy_path=None, train_mode=True, steps_until_reset=65):
        self.train_mode = train_mode
        self.mtc_root = None
        self.mtc_root_sig = None
        self.mtc_exploration_constant = 13 * math.sqrt(2.0)
        self.mtc_max_depth = 30
        self.mtc_iteration_limit = 10

        self.mtc_guide = dict()

        self.completion_reward = 10.0
        self.predicat_completion_reward = +4.0
        self.dead_end_reward = -2.0
        self.depth_limit_reward = 0.0
        self.step_penelty = -0.01
        self.loop_penelty = -2.0

        self.current_step = 0
        self.steps_cap = 150
        self.total_agent_runs = 0
        self.policy_path = policy_path

        self.traveled_path = dict()

        if os.path.exists(self.policy_path):
            self.import_policy()

    def initialize(self, services):
        self.services = services
        self.domain = services.parser.domain_name
        self.problem = services.parser.problem_name
        self.is_probalistic_game = any([type(action) == pddlsim.parser_independent.ProbabilisticAction
                                        for (name, action) in
                                        self.services.valid_actions.provider.parser.actions.items()])

        self.is_hidden_info_game = len(self.services.valid_actions.provider.parser.revealable_predicates) > 0
        self.initial_state = self.services.perception.get_state()
        self.initial_goals_count = self.unfulfilled_goals_count(self.initial_state)

        self.mtc_root = None

    def next_action(self):
        current_state = self.services.perception.get_state()
        current_state_sig = self.state_to_str(current_state)
        goals_reached = self.services.goal_tracking.reached_all_goals()
        self.current_step += 1
        if current_state_sig in self.traveled_path:
            print("--> potential loop <---")
        self.traveled_path[current_state_sig] = self.current_step

        if goals_reached:
            self.export()
            return None
        if self.current_step > self.steps_cap:
            self.export()
            return None

        self.mtc_search(current_state)
        self.mtc_display_choices(current_state_sig)
        optimal_action = self.mtc_recommend(current_state_sig)
        return optimal_action

    ##############################################################################################
    ###################################     MTC functions   ###################################
    ##############################################################################################
    def mtc_search(self, initial_state):
        if self.mtc_root_sig is None:
            self.mtc_root = self.mtc_generate_node(state=initial_state)
            self.mtc_root_sig = self.mtc_root.state_sig
            self.current_node_sig = self.mtc_root_sig
        else:
            self.current_node_sig = self.state_to_str(initial_state)
            self.current_node = self.mtc_guide.get(self.current_node_sig)

        time.sleep(0.1)
        for i in tqdm(range(self.mtc_iteration_limit)):
            reward, chosen_path = self.mtc_executeRound(self.current_node_sig)
            # print("Round {}/{}\tReward: {}\tAction: {}".format(i + 1, self.mtc_iteration_limit, reward,
            #                                                    chosen_path))
        time.sleep(0.1)
        return None

    def mtc_executeRound(self, start_node_sig=None):
        if start_node_sig is None:
            start_node_sig = self.mtc_root_sig
        node_sig, initial_action = self.mtc_selectNode(start_node_sig)
        reward, initial_action_rollout, steps_to_termination, termination_reason, termination_state_sig = self.mtc_rollout(
            node_sig)
        self.mtc_backpropogate(termination_state_sig, reward, steps_to_termination, termination_reason)
        return reward, initial_action

    def mtc_rollout(self, state_sig):
        # Random search policy, without repetitions
        state_node = self.mtc_guide.get(state_sig)
        state = state_node.state
        hit_goal = self.hit_goal(state)
        dead_end = self.hit_dead_end(state)
        traveled_path = dict()
        current_mtc_depth = 0
        initial_action = None
        chosen_action = None
        path_reward = 0
        rollout_step = 0
        termination_reason = 'RUNNING'
        initial_goal_remain = self.unfulfilled_goals_count(state)

        while True:
            original_state_sig = self.state_to_str(state)
            if hit_goal:
                termination_reason = 'GOAL'
                path_reward += self.completion_reward
                break
            if dead_end:
                # DEAD END
                path_reward += self.dead_end_reward
                termination_reason = 'DEAD END'
                break

            # pos = list(state['at'])[0][1]
            # if pos == 't1':
            #     j = 3

            rollout_step += 1
            valid_actions = self.get_valid_action_from_state(state)
            valid_destinations_states = [(t_action, self.apply_action_to_state(t_action, state))
                                         for t_action in valid_actions]

            valid_destinations_states_sigs = [(t_action, t_state, self.state_to_str(t_state))
                                              for t_action, t_state in valid_destinations_states]

            # valid_destinations_states_sigs = [(t_action, t_state, t_state_sig)
            #                                   for t_action, t_state, t_state_sig in valid_destinations_states_sigs
            #                                   if t_state_sig not in traveled_path]
            if len(valid_destinations_states_sigs) == 0:
                # DEAD END
                path_reward += self.dead_end_reward
                termination_reason = 'DEAD END'
                break
                # if current_mtc_depth > self.mtc_max_depth:
                #     # depth cap
                #     termination_reason = 'STEP CAP'
                #     path_reward += self.depth_limit_reward
                #     break
                j = 3
            else:
                chosen_action, chosen_state, chosen_state_sig = random.choice(valid_destinations_states_sigs)
                traveled_path[original_state_sig] = True
                next_state_node = self.mtc_guide.get(chosen_state_sig, None)
                if next_state_node is None:
                    next_state_node = self.mtc_generate_node(state=chosen_state, parent=state_node,
                                                             originating_action=chosen_action)

                # Handle loops
                if chosen_state_sig in traveled_path:
                    path_reward += self.loop_penelty
                    termination_reason = 'LOOP'
                    break

                state = chosen_state
                state_node = next_state_node
                state_sig = self.state_to_str(state)

                hit_goal = self.hit_goal(state)
                dead_end = self.hit_dead_end(state)
                current_mtc_depth += 1
                if initial_action is None:
                    initial_action = chosen_action

        termination_state_sig = original_state_sig
        return path_reward, chosen_action, rollout_step, termination_reason, termination_state_sig

    def mtc_backpropogate(self, termination_state_sig, termination_reward, steps_to_termination, termination_reason):
        c_steps_to_termination = steps_to_termination
        node = self.mtc_guide.get(termination_state_sig)
        propagation_size = 0
        current_step_penelty = 0
        c_steps_to_termination = 0
        while True:
            propagation_size += 1
            node.numVisits += 1
            node.total_steps = c_steps_to_termination
            node.wins += int(termination_reason == 'GOAL')
            node.dead_ends += int(termination_reason == 'DEAD END')

            # if node.state_sig == self.current_node_pointer.state_sig:
            #     break

            parent_node_sig = node.parent_sig
            parent_node = self.mtc_guide.get(parent_node_sig)
            predicates_solved_reward = 0
            if parent_node_sig is not None:
                predicates_solved_reward = (parent_node.goalRemain - node.goalRemain) * self.predicat_completion_reward
            # Calculate reward
            reward = termination_reward + current_step_penelty + predicates_solved_reward
            node.totalReward += reward

            node = parent_node
            c_steps_to_termination += 1
            current_step_penelty += self.step_penelty

            if node is None:
                break

        # print("Prop --> {}".format(propagation_size))

    def mtc_selectNode(self, node_sig):
        node = self.mtc_guide.get(node_sig)
        hit = 0
        traveled_path = dict()
        while not node.isTerminal:
            if node.state_sig in traveled_path:
                # Stuck in a loop
                node_sig, action = self.mtc_expand(node)
                return node_sig, action
            else:
                traveled_path[node.state_sig] = True

            if node.isFullyExpanded:
                node_sig, action = self.mtc_getBestChild(node)
                node = self.mtc_guide.get(node_sig)
            else:
                node_sig, action = self.mtc_expand(node)
                return node_sig, action
        return node_sig, None

    def mtc_expand(self, node):
        actions = self.get_valid_action_from_state(node.state)
        for action in actions:
            if action not in node.children_sig.keys():
                # Action is not explored
                new_state = self.apply_action_to_state(action, node.state)
                newNode = self.mtc_generate_node(state=new_state, parent=node, originating_action=action)
                node.children_sig[action] = newNode.state_sig
                if len(actions) == len(node.children_sig):
                    node.isFullyExpanded = True
                return newNode.state_sig, action

        node_sig, action = self.mtc_getBestChild(node)
        return node_sig, action

    def mtc_getBestChild(self, node, exploit=False):
        if type(node) is type('str'):
            j = 3

        child_nodes, actions, scores, rewards = self.get_scores_of_node(node)
        if exploit:
            chosen_node_idx = np.argmax(rewards)
        else:
            chosen_node_idx = np.argmax(scores)

        chosen_node = child_nodes[chosen_node_idx]
        chosen_node_sig = chosen_node.state_sig
        chosen_action = actions[chosen_node_idx]
        return chosen_node_sig, chosen_action

    def mtc_recommend(self, current_node_sig=None):
        if current_node_sig is None:
            current_node_sig = self.current_node_pointer
        current_node = self.mtc_guide.get(current_node_sig)
        chosen_node, chosen_action = self.mtc_getBestChild(current_node, exploit=True)
        return chosen_action

    def mtc_display_choices(self, current_node_sig):

        current_node = self.mtc_guide.get(current_node_sig)
        current_predicates = current_node.goalRemain
        print('_______________________________________________________________________________')
        print('_______________________________________________________________________________')
        child_nodes, actions, scores, rewards = self.get_scores_of_node(current_node)
        max_reward_idx = np.argmax(rewards)
        max_score_idx = np.argmax(scores)

        for child_idx, child in enumerate(child_nodes):
            expected_predicates = child.goalRemain
            best_score_pointer = ''
            if child_idx == max_reward_idx:
                best_score_pointer = '@\t'
            best_reward_pointer = ''
            if child_idx == max_score_idx:
                best_reward_pointer = 'O\t'

            visits = child.numVisits
            if visits == 0:
                visits = -1
            action = actions[child_idx]
            msg = ''
            # msg += '[Expended: {}]\t'.format(child.isFullyExpanded)
            msg += "Score: {:>+.6f}\t".format(scores[child_idx])
            msg += "Rewards: {:>+.5f}\t".format(rewards[child_idx])
            msg += "Wins: {}/{} ({:>.3f})\t".format(child.wins, visits, float(child.wins) / visits)
            # msg += "DeadEnds: {}/{}\t".format(child.dead_ends, visits)
            # msg += "Steps: {:>5.2f}\t".format(float(child.total_steps) / child.numVisits)
            msg += "Expected goals: {}-->{}\t".format(current_predicates, expected_predicates)
            msg += "{}{}{}".format(best_reward_pointer, best_score_pointer, action)
            print(msg)
        print('_______________________________________________________________________________')

    ##############################################################################################
    ###################################     node handling###   ###################################
    ##############################################################################################

    def mtc_generate_node(self, state, parent=None, originating_action=None):
        state_sig = get_id_for_dict(state)
        state_node = self.mtc_guide.get(state_sig, None)
        if state_node is None:
            state_node = treeNode(state=state, parent=parent, originating_action=originating_action,
                                  exploration_constant=self.mtc_exploration_constant)
            if parent is not None:
                parent_sig = parent.state_sig
                state_node.parent_sig = parent_sig
                state_node.depth = parent.depth + 1
            state_node.state_sig = state_sig
            is_dead_end = self.hit_dead_end(state)
            goals_remaining = self.unfulfilled_goals_count(state)
            is_terminal = is_dead_end or (goals_remaining == 0)
            state_node.isDeadEnd = is_dead_end
            state_node.isTerminal = is_terminal
            state_node.isFullyExpanded = is_terminal
            state_node.goalRemain = goals_remaining
            self.mtc_guide[state_node.state_sig] = state_node
        else:
            pass

        return state_node

    ##############################################################################################
    ###################################     domain functions   ###################################
    ##############################################################################################

    def export(self):
        if not self.train_mode:
            return False

        output = dict()
        sys.setrecursionlimit(10000)
        output['mtc_tree'] = pickle.dumps(self.mtc_root)
        output['total_agent_runs'] = self.total_agent_runs
        if self.train_mode:
            with open(self.policy_path, 'w') as ffile:
                json.dump(output, ffile, indent=4)

        return True

    def import_policy(self):
        with open(self.policy_path) as json_file:
            output = json.load(json_file)

        self.mtc_root_x = pickle.loads(output['mtc_tree'])
        self.total_agent_runs = output['total_agent_runs'] + 1

    def get_valid_action_from_state(self, state):
        possible_actions = []
        for (name, action) in self.services.valid_actions.provider.parser.actions.items():
            for candidate in self.services.valid_actions.provider.get_valid_candidates_for_action(state, action):
                possible_actions.append(action.action_string(candidate))
        return possible_actions

    def get_valid_action_from_state_with_probabilities(self, state):
        possible_actions = []
        action_outcomes_probabilities = list()
        for (name, action) in self.services.valid_actions.provider.parser.actions.items():
            for candidate in self.services.valid_actions.provider.get_valid_candidates_for_action(state, action):
                possible_actions.append(action.action_string(candidate))
                action_outcomes_probabilities.append(action.prob_list)

        return possible_actions, action_outcomes_probabilities

    def apply_probabilistic_action_to_state(self, action_sig, state, chosen_outcome=None):
        action_name, param_names = self.services.parser.parse_action(action_sig)
        action = self.services.parser.actions[action_name]
        params = map(self.services.parser.get_object, param_names)
        param_mapping = action.get_param_mapping(params)

        if chosen_outcome is None:
            chosen_outcome = action.choose_random_effect()

        possible_outcomes = len(action.prob_list)
        if chosen_outcome >= possible_outcomes:
            print("BAD CHOSEN OUTCOME")

        index = chosen_outcome
        for (predicate_name, entry) in action.to_delete(param_mapping, index):
            predicate_set = state[predicate_name]
            if entry in predicate_set:
                predicate_set.remove(entry)

        for (predicate_name, entry) in action.to_add(param_mapping, index):
            state[predicate_name].add(entry)

    def apply_action_to_state(self, action, state):
        state_copy = self.services.parser.copy_state(state)
        self.services.parser.apply_action_to_state(action, state_copy, check_preconditions=False)
        return state_copy

    def find_unfulfilled_subgoal(self, subgoal, state):
        if subgoal.test(state):
            return None
        else:
            if type(subgoal) == pddlsim.parser_independent.Literal:
                return subgoal
            else:
                for subsubgoal in subgoal.parts:
                    l = self.find_unfulfilled_subgoal(subsubgoal, state)
                    if l is not None:
                        # Got one
                        return l
                    else:
                        # Keep looking
                        pass
                return None

    def unfulfilled_goals_count(self, state, goal=None):
        if goal is None:
            goal = self.services.goal_tracking.uncompleted_goals[0]

        if goal.test(state):
            return 0
        else:
            if type(goal) is pddlsim.parser_independent.Literal:
                return 1
            else:
                counter = 0
                for subsubgoal in goal.parts:
                    counter += self.unfulfilled_goals_count(state, subsubgoal)
                return counter

    def is_terminal(self, state):
        goal = self.services.goal_tracking.uncompleted_goals[0]
        hit_goal = goal.test(state)
        dead_end = len(self.get_valid_action_from_state(state)) == 0
        return hit_goal or dead_end

    def hit_goal(self, state):
        goal = self.services.goal_tracking.uncompleted_goals[0]
        hit_goal = goal.test(state)
        return hit_goal

    def hit_dead_end(self, state):
        valid_actions = self.get_valid_action_from_state(state)
        no_children = len(valid_actions) == 0
        if no_children:
            # Classic dead end
            return True

        state_sig = self.state_to_str(state)
        state_node = self.mtc_guide.get(state_sig, None)
        if state_node is None:
            # Current node is not explored
            return False

        return False

    ###############################################################################################
    ###############################################################################################

    def state_to_str(self, state):
        # state_copy = self.services.parser.copy_state(state)
        # state_copy.pop('=')
        # ret = str(state_copy)
        # ret = state_copy['at'].pop()[1]
        ret = get_id_for_dict(state)
        return ret

    def action_to_str(self, action):
        ret = str(action)
        return ret

    def state_action_to_str(self, state, action):
        ret = '{}___{}'.format(self.state_to_str(state), self.action_to_str(action))
        return ret


if __name__ == '__main__':
    # Get args
    worlds = dict()
    worlds['satellite'] = (r"C:\school\cognitive\cognitive_project\satellite_domain_multi.pddl",
                           r"C:\school\cognitive\cognitive_project\satellite_problem_multi.pddl")
    worlds['freecell'] = (r"C:\school\cognitive\cognitive_project\freecell_domain.pddl",
                          r"C:\school\cognitive\cognitive_project\freecell_problem.pddl")
    worlds['rover'] = (r"C:\school\cognitive\cognitive_project\rover_domain.pddl",
                       r"C:\school\cognitive\cognitive_project\rover_problem.pddl")
    worlds['simple'] = (r"C:\school\cognitive\cognitive_project\domain_simple.pddl",
                        r"C:\school\cognitive\cognitive_project\problem_simple.pddl")
    worlds['simple_web'] = (r"C:\school\cognitive\cognitive_project\domain_simple.pddl",
                            r"C:\school\cognitive\cognitive_project\problem_simple_web.pddl")

    current_world = 'rover'
    args = sys.argv
    run_mode_flag = 'L'
    domain_path = worlds[current_world][0]  # args[2]
    problem_path = worlds[current_world][1]  # args[3]
    policy_path = 'POLICYFILE'
    train_mode = 'L' in run_mode_flag

    clear_policy(policy_path)

    if train_mode:
        # Train mode
        global_start_time = datetime.now()
        iterations = 80
        moving_average_window = 30
        results_moving_average = list()
        rewards_moving_average = list()
        for i in range(iterations):
            simulator = LocalSimulator()
            executor = MyExecutor(policy_path=policy_path, train_mode=True)
            ret_message = simulator.run(domain_path, problem_path, executor)
            print(ret_message)

    # Test mode
    simulator = LocalSimulator()
    executor = MyExecutor(policy_path=policy_path, train_mode=False, steps_until_reset=270)
    ret_message = simulator.run(domain_path, problem_path, executor)
    print(ret_message)
