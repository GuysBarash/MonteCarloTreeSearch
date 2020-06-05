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
        self.committed_path = list()
        self.exploration_constant = exploration_constant
        self.node_id = get_node_id()
        self.state = state
        self.state_sig = None
        self.isTerminal = terminal
        self.isDeadEnd = False
        self.goalRemain = 10000
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.parent_sig = None
        self.parent_action = originating_action
        self.numVisits = 0
        self.totalReward = 0
        self.wins = 0
        self.dead_ends = 0
        self.total_steps = 0
        self.actions = dict()
        self.depth = 0

    def add_action(self, action):
        if action in self.actions:
            j = 3
        else:
            self.actions[action] = dict()
            self.actions[action]['Visits'] = 0
            self.actions[action]['Rewards'] = 0

    def add_action_reward(self, action, reward, termination_reason=None):
        if action not in self.actions:
            self.add_action(action)

        self.actions[action]['Visits'] = self.actions[action]['Visits'] + 1
        self.actions[action]['Rewards'] = self.actions[action]['Rewards'] + reward
        if termination_reason is not None:
            self.actions[action][termination_reason] = self.actions[action].get(termination_reason, 0) + 1


class MyExecutor(object):

    def __init__(self, exploration_rate_start=None, policy_path=None, train_mode=True, steps_until_reset=65):
        self.train_mode = train_mode
        self.mtc_root = None
        self.mtc_root_sig = None

        if train_mode:
            self.mtc_exploration_constant = 13 * math.sqrt(2.0)
            self.mtc_iteration_limit = 50
        else:
            self.mtc_exploration_constant = 5 * math.sqrt(2.0)
            self.mtc_iteration_limit = 2

        self.mtc_guide = dict()

        self.completion_reward = 20.0
        self.predicat_completion_reward = +7.0
        self.dead_end_reward = -2.0
        self.depth_limit_reward = 0.0
        self.step_penelty = -0.01
        self.step_in_loop_penelty = 0.0
        self.history_repeat_penalty = -0.05
        self.loop_break_penelty = -4.0

        self.visit_limit_to_avoid_loops = 4
        self.depth_limit = 40
        self.current_step = 0
        self.steps_cap = 150
        self.total_agent_runs = 0
        self.policy_path = policy_path

        self.traveled_path_histogram = dict()
        self.traveled_path = list()
        self.traveled_path_actions = list()

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
        if current_state_sig in self.traveled_path_histogram:
            print("--> potential loop <---")
        self.traveled_path_histogram[current_state_sig] = self.current_step

        if goals_reached:
            self.export()
            return None
        if self.current_step > self.steps_cap:
            self.export()
            return None

        avg_depth = self.mtc_search(current_state)
        self.mtc_display_choices(current_state_sig, avg_depth)
        optimal_action = self.mtc_recommend(current_state_sig)
        self.traveled_path += [current_state_sig]
        self.traveled_path_actions += [optimal_action]
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
        depths = list()
        for current_mtc_idx in tqdm(range(self.mtc_iteration_limit), desc='[Step: {}]'.format(self.current_step)):
            reward, chosen_path, path_depth = self.mtc_executeRound(self.current_node_sig)
            depths += [path_depth]
            # print("Round {}/{}\tReward: {}\tAction: {}".format(i + 1, self.mtc_iteration_limit, reward,
            #                                                    chosen_path))
        time.sleep(0.1)

        avg_depth = np.mean(depths)
        return avg_depth

    def mtc_executeRound(self, start_node_sig):
        initial_simulated_node_sig, initial_action = self.mtc_selectNode(start_node_sig)
        first_step_token = (start_node_sig, initial_action, initial_simulated_node_sig)
        reward, initial_action_rollout, steps_to_termination, termination_reason, simulation_traveled_path = self.mtc_rollout(
            initial_simulated_node_sig)

        simulation_traveled_path = [first_step_token] + simulation_traveled_path
        path_depth = len(simulation_traveled_path)
        self.mtc_backpropogate(simulation_traveled_path, reward, steps_to_termination, termination_reason)
        return reward, initial_action, path_depth

    def mtc_rollout(self, state_sig):
        # Random search policy, without repetitions
        state_node = self.mtc_guide.get(state_sig)
        state = state_node.state
        hit_goal = self.hit_goal(state)
        dead_end = self.hit_dead_end(state)
        entered_loop = False
        traveled_path_histogram = dict()
        simulation_traveled_path = list()
        current_mtc_depth = 0
        chosen_action = None
        path_reward = 0
        rollout_step = 0
        termination_reason = 'RUNNING'
        # Terminations types: GOAL , LOOP , DEAD END
        initial_goal_remain = state_node.goalRemain

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

            if entered_loop:
                # Loop
                path_reward += self.loop_break_penelty
                termination_reason = 'LOOP'
                break

            if rollout_step > self.depth_limit:
                # Depth limit
                path_reward += self.depth_limit_reward
                termination_reason = 'DEPTH CAP'
                break

            # Calculate step options
            rollout_step += 1
            valid_actions = self.get_valid_action_from_state(state)
            chosen_action = random.choice(valid_actions)
            chosen_state = self.apply_action_to_state(chosen_action, state)
            chosen_state_sig = self.state_to_str(chosen_state)
            traveled_path_histogram[original_state_sig] = traveled_path_histogram.get(original_state_sig, 0) + 1
            simulation_traveled_path.append((original_state_sig, chosen_action, chosen_state_sig))
            next_state_node = self.mtc_guide.get(chosen_state_sig, None)
            if next_state_node is None:
                next_state_node = self.mtc_generate_node(state=chosen_state, parent=state_node,
                                                         originating_action=chosen_action)
            else:
                # This path has been traveresed
                # maybe replace the random policy?
                pass

            # Handle loops
            if chosen_state_sig in traveled_path_histogram:
                path_reward += self.step_in_loop_penelty
                visits_to_current_state = traveled_path_histogram[chosen_state_sig]
                if visits_to_current_state > self.visit_limit_to_avoid_loops:
                    entered_loop = True

            state = chosen_state
            state_node = next_state_node
            state_sig = self.state_to_str(state)

            hit_goal = self.hit_goal(state)
            dead_end = self.hit_dead_end(state)
            current_mtc_depth += 1

        return path_reward, chosen_action, rollout_step, termination_reason, simulation_traveled_path

    def mtc_backpropogate(self, traveled_path, termination_reward, steps_to_termination, termination_reason):
        if termination_reason == 'DEPTH CAP':
            # update just current action
            parent_node_sig, parent_action, _ = traveled_path[0]
            _, _, node_sig = traveled_path[-1]
            node = self.mtc_guide.get(node_sig)
            parent_node = self.mtc_guide.get(parent_node_sig)

            predicates_solved_reward = (parent_node.goalRemain - node.goalRemain) * self.predicat_completion_reward
            reward = self.depth_limit_reward + predicates_solved_reward
            parent_node.add_action_reward(action=parent_action,
                                          reward=reward,
                                          termination_reason=termination_reason)


        else:
            propagation_size = 0
            current_step_penelty = 0
            c_steps_to_termination = 0
            traveled_path.reverse()

            for parent_node_sig, parent_action, node_sig in traveled_path:
                node = self.mtc_guide[node_sig]
                parent_node = self.mtc_guide.get(parent_node_sig)
                parent_node.numVisits += 1
                propagation_size += 1
                predicates_solved_reward = (parent_node.goalRemain - node.goalRemain) * self.predicat_completion_reward
                revisiting_old_path_penalty = self.traveled_path_histogram.get(node_sig,
                                                                               0.0) * self.history_repeat_penalty

                reward = termination_reward + current_step_penelty + predicates_solved_reward + revisiting_old_path_penalty
                parent_node.add_action_reward(action=parent_action,
                                              reward=reward,
                                              termination_reason=termination_reason)
                c_steps_to_termination += 1
                current_step_penelty += self.step_penelty

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
            if action not in node.actions.keys():
                # Action is not explored
                new_state = self.apply_action_to_state(action, node.state)
                newNode = self.mtc_generate_node(state=new_state, parent=node, originating_action=action)
                if len(actions) == len(node.actions):
                    node.isFullyExpanded = True
                return newNode.state_sig, action

        node_sig, action = self.mtc_getBestChild(node)
        return node_sig, action

    def mtc_getBestChild(self, node, exploit=False):
        actions, scores, rewards = self.get_scores_of_node(node)
        if len(actions) <= 0:
            return None, None
        if exploit:
            chosen_node_idx = np.argmax(rewards)
        else:
            chosen_node_idx = np.argmax(scores)

        node_state = node.state
        chosen_action = actions[chosen_node_idx]
        chosen_state = self.apply_action_to_state(chosen_action, node_state)
        chosen_state_sig = self.state_to_str(chosen_state)
        return chosen_state_sig, chosen_action

    def mtc_recommend(self, current_node_sig=None):
        if current_node_sig is None:
            current_node_sig = self.current_node_pointer
        current_node = self.mtc_guide.get(current_node_sig)
        chosen_node, chosen_action = self.mtc_getBestChild(current_node, exploit=True)
        return chosen_action

    def mtc_display_choices(self, current_node_sig, avg_depth=-1):

        current_node = self.mtc_guide.get(current_node_sig)
        current_predicates = current_node.goalRemain
        total_visits = 0
        print('_______________________________________________________________________________')
        print('_______________________________________________________________________________')

        actions, scores, rewards = self.get_scores_of_node(current_node)
        if len(actions) > 0:
            max_reward_idx = np.argmax(rewards)
            max_score_idx = np.argmax(scores)
            for action_idx, action in enumerate(actions):
                action_stats = current_node.actions[action]
                best_score_pointer = ''
                if action_idx == max_reward_idx:
                    best_score_pointer = '@\t'
                best_reward_pointer = ''
                if action_idx == max_score_idx:
                    best_reward_pointer = 'O\t'

                visits = action_stats['Visits']
                total_visits += visits
                wins = action_stats.get('GOAL', 0)
                loops = action_stats.get('LOOP', 0)
                deadends = action_stats.get('DEAD END', 0)
                depth_cap = action_stats.get('DEPTH CAP', 0)
                if visits == 0:
                    visits = -1
                action = actions[action_idx]
                msg = ''
                # msg += '[Expended: {}]\t'.format(child.isFullyExpanded)
                msg += "Score: {:>+.6f}\t".format(scores[action_idx])
                msg += "Rewards: {:>+.5f}\t".format(rewards[action_idx])
                msg += "Wins: {}/{} ({:>.3f})\t".format(wins, visits, float(wins) / visits)
                msg += "loops: {}/{} ({:>.3f})\t".format(loops, visits, float(loops) / visits)
                msg += "dead-ends: {}/{} ({:>.3f})\t".format(deadends, visits, float(deadends) / visits)
                msg += "depth-cap: {}/{} ({:>.3f})\t".format(depth_cap, visits, float(depth_cap) / visits)
                # msg += "DeadEnds: {}/{}\t".format(child.dead_ends, visits)
                # msg += "Steps: {:>5.2f}\t".format(float(child.total_steps) / child.numVisits)
                msg += "{}{}{}".format(best_reward_pointer, best_score_pointer, action)
                print(msg)
        else:
            print("NO AVAILABLE ACTIONS")
        print("-------------------------------------------------------------------------------")
        msg = ''
        msg += 'Current predicates: {}\t'.format(current_predicates)
        msg += 'Total visits: {}\t'.format(total_visits)
        msg += 'Avg depth: {:>.1f}\t'.format(avg_depth)
        msg += 'Tree size: {}\t'.format(current_node_id)
        print(msg)
        print('')
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

    def get_scores_of_node(self, node):
        actions = list()
        scores = list()
        rewards = list()
        for action, action_stats in node.actions.items():
            if node.numVisits <= 0:
                reward = 0
                score = float(2 ** 30)

            elif action_stats['Visits'] <= 0:
                reward = 0
                score = float(2 ** 30)
            else:
                reward = float(action_stats['Rewards']) / action_stats['Visits']
                exploration_factor = math.sqrt(math.log(node.numVisits) / action_stats['Visits'])
                score = reward + (self.mtc_exploration_constant * exploration_factor)

            actions.append(action)
            scores.append(score)
            rewards.append(reward)

        return actions, scores, rewards

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
        iterations = 1
        moving_average_window = 1
        results_moving_average = list()
        rewards_moving_average = list()
        for current_simulation_id in range(iterations):
            simulator = LocalSimulator()
            executor = MyExecutor(policy_path=policy_path, train_mode=True)
            ret_message = simulator.run(domain_path, problem_path, executor)
            print(ret_message)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    else:

        # Test mode
        simulator = LocalSimulator()
        executor = MyExecutor(policy_path=policy_path, train_mode=False, steps_until_reset=270)
        ret_message = simulator.run(domain_path, problem_path, executor)
        print(ret_message)
