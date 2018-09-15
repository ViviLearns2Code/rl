import numpy as np

BOARD = np.array([20, 1, 18, 4, 13, 9, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5])
SCORE = np.concatenate((BOARD, BOARD, 2 * BOARD, 3 * BOARD, [25, 50, 0]))

class DartInfo:

    def get_right_neighbor(self, action_index):
        if action_index % 20 == 19:
            return (action_index // 20) * 20
        else:
            return action_index + 1

    def get_left_neighbor(self, action_index):
        if action_index % 20 == 0:
            return (action_index // 20) * 20 + 19
        else:
            return action_index - 1

    def get_probs_outer_single(self, action_index):
        single1_left = self.get_left_neighbor(action_index)
        single1_right = self.get_right_neighbor(action_index)

        double = action_index + 20 * 2
        treble = action_index + 20 * 3
        double_left = single1_left + 20 * 2
        double_right = single1_right + 20 * 2
        treble_left = single1_left + 20 * 3
        treble_right = single1_right + 20 * 3

        return np.array([
            [action_index, SCORE[action_index], 0.92],
            [single1_left, SCORE[single1_left], 0.02],
            [single1_right, SCORE[single1_right], 0.02],
            [double, SCORE[double], 0.01],
            [treble, SCORE[treble], 0.01],
            [double_left, SCORE[double_left], 0.005],
            [double_right, SCORE[double_right], 0.005],
            [treble_left, SCORE[treble_left], 0.005],
            [treble_right, SCORE[treble_right], 0.005]])

    def get_probs_inner_single(self, action_index):
        # 20..39
        single2_left = self.get_left_neighbor(action_index)
        single2_right = self.get_right_neighbor(action_index)
        treble = action_index + 20 * 2
        treble_left = single2_left + 20 * 2
        treble_right = single2_right + 20 * 2
        outer_bullseye = 80

        return np.array([
            [action_index, SCORE[action_index], 0.92],
            [single2_left, SCORE[single2_left], 0.02],
            [single2_right, SCORE[single2_right], 0.02],
            [outer_bullseye, SCORE[outer_bullseye], 0.015],
            [treble, SCORE[treble], 0.015],
            [treble_left, SCORE[treble_left], 0.005],
            [treble_right, SCORE[treble_right], 0.005]])

    def get_probs_double(self, action_index):
        # 40..59
        double_left = self.get_left_neighbor(action_index)
        double_right = self.get_right_neighbor(action_index)
        out = 82
        single1 = action_index - 20 * 2
        single1_left = double_left - 20 * 2
        single1_right = double_right - 20 * 2

        return np.array([
            [action_index, SCORE[action_index], 0.40],
            [double_left, SCORE[double_left], 0.025],
            [double_right, SCORE[double_right], 0.025],
            [single1, SCORE[single1], 0.2],
            [single1_left, SCORE[single1_left], 0.025],
            [single1_right, SCORE[single1_right], 0.025],
            [out, SCORE[out], 0.3]])

    def get_probs_treble(self, action_index):
        # 60..79
        treble_left = self.get_left_neighbor(action_index)
        treble_right = self.get_right_neighbor(action_index)
        single1 = action_index - 20 * 3
        single2 = action_index - 20 * 2
        single1_left = treble_left - 20 * 3
        single1_right = treble_right - 20 * 3
        single2 = action_index - 20 * 2
        single2_left = treble_left - 20 * 2
        single2_right = treble_right - 20 * 2

        return np.array([
            [action_index, SCORE[action_index], 0.3],
            [treble_left, SCORE[treble_left], 0.01],
            [treble_right, SCORE[treble_right], 0.01],
            [single1, SCORE[single1], 0.32],
            [single2, SCORE[single2], 0.32],
            [single1_left, SCORE[single1_left], 0.01],
            [single1_right, SCORE[single1_right], 0.01],
            [single2_left, SCORE[single2_left], 0.01],
            [single2_right, SCORE[single2_right], 0.01]
        ])

    def get_probs_outer_bullseye(self):
        elems = [[80, 25, 0.6], [81, 50, 0.2]]
        for single2 in np.arange(20, 40):
            elems.append([single2, SCORE[single2], 0.01])
        return np.array(elems)

    def get_probs_bullseye(self):
        elems = [[80, 25, 0.4], [81, 50, 0.5]]
        for single2 in np.arange(20, 40):
            elems.append([single2, SCORE[single2], 0.005])
        return np.array(elems)

    def is_double(self, action_index):
        score_type = action_index // 20
        if score_type == 2:
            return True
        else:
            return False

    def is_treble(self, action_index):
        score_type = action_index // 20
        if score_type == 3:
            return True
        else:
            return False

    def is_outer_single(self, action_index):
        score_type = action_index // 20
        if score_type == 0:
            return True
        else:
            return False

    def is_inner_single(self, action_index):
        score_type = action_index // 20
        if score_type == 1:
            return True
        else:
            return False

    def get_prob(self, action_index):
        if self.is_outer_single(action_index):
            # 0...19
            return self.get_probs_outer_single(action_index)
        elif self.is_inner_single(action_index):
            # 20...39
            return self.get_probs_inner_single(action_index)
        elif self.is_double(action_index):
            # 40...59
            return self.get_probs_double(action_index)
        elif self.is_treble(action_index):
            # 60...79
            return self.get_probs_double(action_index)
        elif action_index == 80:
            return self.get_probs_outer_bullseye()
        elif action_index == 81:
            return self.get_probs_bullseye()