import logging
import os
import pickle
import sys
import time
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from matplotlib import pyplot as plt

import numpy as np
from tqdm import tqdm
import wandb
import random

from MCTS import MCTS

import itertools

log = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

class Coach():

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        if self.nnet is not None:
            self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.all_logging_data = []
        self.nn_iteration = None
        self.mcts = MCTS(self.nnet, self.args, self.all_logging_data, self.nn_iteration, cache_data=None)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overridden in loadTrainExamples()
        self.leaf_counter = 0

    def DFSUtil(self, game, board, level, trainExamples, all_cubes, all_cubes_verbose):
        # TODO: Incorporate canonicalBoard & symmetry appropriately when required in the future
        # canonicalBoard = game.getCanonicalForm(board)
        # sym = game.getSymmetries(canonicalBoard, pi)
        # for b, p in sym:
        #     trainExamples.append([b.get_state(), p, None])
        
        # visited.add(v) # no need if we are using a tree

        reward_now = game.getGameEnded(board)
        if reward_now: # reward is not None, i.e., game over
            flattened_list = itertools.chain.from_iterable(board.prior_actions)
            # if board.is_fail():
            #     log.info("March said UNSAT --- skipping this cube (not adding to file)")
            # else:
            all_cubes.append(flattened_list) # adding all cubes
            all_cubes_verbose.append([board.var_elim_till_now,board.ranks_till_now])
            self.leaf_counter += 1
            if self.args.debugging:
                log.info(f"Leaf node: {self.leaf_counter} with reward = {reward_now} and state: {board}")
                log.info(f"Vars eliminated till now: {board.var_elim_till_now}; Ranks till now: {board.ranks_till_now}")
            return reward_now # only leaves have rewards & leaves don't have neighbors
        else: # None
            reward_now = 0 # initialize reward for non-leaf nodes
        # Non-leaf nodes
        # temp = int(level < self.args.tempThreshold)
        if self.args.debugging: log.info(f"-----------------------------------\nDFS level: {level}")
        self.mcts.resetMCTSdict() # reset the MCTS state dict for each node otherwise the exploration gets much deeper
        pi = self.mcts.getActionProb(game, board, temp=0, verbose=self.args.verbose)
        valids = game.getValidMoves(board)

        a = np.random.choice(len(pi), p=pi)
        if self.args.debugging: 
            print(f"a: {a}, board.var2lit[a]: {board.var2lit[a]}, board.ranked_keys: {board.ranked_keys[:10]}")
            print("Board: ", board)
        try: 
            march_rank = board.ranked_keys.index(abs(board.var2lit[a])) + 1
        except Exception as e: # debug based on file (e4_20_mcts_nod_s300_c3_pen02-cdr1138-14664123.out) in Debug dir in Git Large Files
            march_rank = -1
            print("Exception: ", e)
            print("board.valid_literals: ", board.valid_literals, board.march_pos_lit_score_dict)
        if self.args.debugging: 
            log.info(f"DFS best action is {a} with rank {march_rank}, pi = {pi[a]:.3f}, max pi value {max(pi):.3f}, same pi count = {sum(np.array(pi) == pi[a])}")
        wandb.log({"march_rank": march_rank})

        s = game.stringRepresentation(board)
        comp_a = board.get_complement_action(a)
        (next_s_dir1, board) = self.mcts.cache_data[(s, a)]
        (next_s_dir2, board) = self.mcts.cache_data[(s, comp_a)]
        game_copy_dir1 = game.get_copy()
        game_copy_dir2 = game.get_copy()

        # game_copy_dir1 = game.get_copy()
        # next_s_dir1 = game_copy_dir1.getNextState(board, a)

        # comp_a = board.get_complement_action(a) # complement of the literal
        # game_copy_dir2 = game.get_copy()
        # next_s_dir2 = game_copy_dir2.getNextState(board, comp_a)

        assert valids[a] and valids[comp_a], "Invalid action chosen by MCTS"

        for game_n, neighbour in zip((game_copy_dir1, game_copy_dir2), (next_s_dir1, next_s_dir2)): 
            reward_now += self.DFSUtil(game_n, neighbour, level+1, trainExamples, all_cubes, all_cubes_verbose)
        reward_now = reward_now/2 # average reward of the two children
        
        trainExamples.append([board.get_state(), pi, reward_now]) # after all children are visited, add a reward to the current node
        return reward_now # return the reward to the parent

    def executeEpisode(self):
        start_time = time.time()
        trainExamples = []
        all_cubes = []
        all_cubes_verbose = []
        game = self.game.get_copy()
        board = game.getInitBoard()

        self.leaf_counter = 0
        r = self.DFSUtil(game, board, level=1, trainExamples=trainExamples, all_cubes=all_cubes, all_cubes_verbose=all_cubes_verbose)

        time_elapsed = time.time() - start_time
        print("Time taken for cubing: ", round(time_elapsed, 3))

        if self.args.MCTSmode == 0:
            arena_cubes = [list(map(str, l)) for l in all_cubes]
            if os.path.exists(self.args.o):
                log.info(f"{self.args.o} already exists. Replacing old file!")
            f = open(self.args.o, "w")
            f.writelines(["a " + " ".join(l) + " 0\n" for l in arena_cubes])
            f.close()

            log.info("Saved cubes to file")

            if self.args.debugging:
                all_cube_elims = [cubes[0][-1] for cubes in all_cubes_verbose if cubes[0][-1] is not None]
                arena_cubes_v = [list(map(str, l)) for l in all_cubes_verbose]
                f = open(self.args.o+"_verbose", "w")
                f.writelines(["a " + " ".join(l) + f" 0;" + " ".join(lv) + "\n" for l, lv in zip(arena_cubes, arena_cubes_v)])
                f.write(f"Cube eliminations - {all_cube_elims}\nMax: {max(all_cube_elims):.2f}, Mean: {np.mean(all_cube_elims):.2f}, Std: {np.std(all_cube_elims):.2f}")
                f.close()

                log.info("Saved cubes (verbose) to file")

            print("Reward: ", r)
            # with open('trainExamples.pkl', 'wb') as f: # For NN
            #     pickle.dump(trainExamples, f)
            # print("Saved Training examples to trainExamples.pkl")

        return trainExamples

    def nolearnMCTS(self):
        self.mcts = MCTS(self.nnet, self.args, self.all_logging_data, self.nn_iteration, cache_data=None)  # reset search tree every episode
        # if os.path.exists("mcts_cache.pkl"):
        #     with open('mcts_cache.pkl', 'rb') as f:
        #         self.mcts.cache_data = pickle.load(f)
        self.executeEpisode()
        # with open('mcts_cache.pkl', 'wb') as f:
        #     pickle.dump(self.mcts.cache_data, f)
