import numpy as np

"""
PER for SPG 

PER - https://arxiv.org/pdf/1511.05952.pdf
    see sections 3.3, B.2.1
ACER - https://arxiv.org/abs/1611.01224


- prioritized replay stores experiences with priority p_i, samples experiences with probability proportional to p_i^\alpha / \sum_k p_k^\alpha
- sample with stratified sampling, segmenting range [0, p_total] into k (for mini-batch of k) sub-ranges and sampling uniformly from these
                
- SPG is an actor-critic contextual bandit
    - p_i is computed from critic as r(s,a) - Q(s,a) from equations 2, 6 rather than the TD error
    
https://github.com/rlcode/per

"""

class ReplayBuffer():
    epsilon = 0.01
    alpha = 0.8 # TODO tune this

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
    
    def __get_priority(self, error):
        return (abs(error) + self.epsilon) ** self.alpha

    def add(self, error, experience):
        priority = self.__get_priority(error)
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        batch = []
        
        # stratified sampling
        segment_size = self.tree.total() /  batch_size
        for i in range(batch_size):
            low, high = segment_size * i, segment_size * (i+1)
            priority = np.random.uniform(low, high)
            idx, _, experience = self.tree.get(priority)
            batch.append((idx, experience))
        
        return batch
            
    def update(self, idx, error):
        priority = self.__get_priority(error)
        self.tree.update(idx, priority)

class SumTree:    

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.cursor = 0
        self.n_entries = 0
        

    def total(self):
        return self.tree[0]

    def __propagate(self, idx, change):
        """ propagate a change in priority from a node to the root node"""
        parent_idx = (idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self.__propagate(parent_idx, change)

    def update(self, idx, priority):
        """ set the priority of a node (a leaf node)"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self.__propagate(idx, change)

    def __retrieve(self, idx, val):
        """ Traverse the tree 
            Not clear what guarantees a sum-tree gives, even if 
            priorities (i.e. leaf priorities) are ordered, e.g.:
               10        
              /  \       sample for 1.5, returns 2 
             4    6      sample for 2.5, returns 2 also
            / \  / \     ...
            1  2 3  4

        """
        left_idx, right_idx = 2*idx + 1, 2*idx + 2
        if left_idx >= len(self.tree):
            return idx
        if val <= self.tree[left_idx]:
            return self.__retrieve(left_idx, val)
        else:
            return self.__retrieve(right_idx, val - self.tree[left_idx])

    def get(self, priority):
        """get experience resulting from traversing with particular priority"""
        idx = self.__retrieve(0, priority)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])      

    def add(self, priority, datum):
        idx = self.cursor + self.capacity - 1
        
        self.data[self.cursor] = datum
        self.update(idx, priority)

        self.cursor += 1
        if self.cursor >= self.capacity:
            self.cursor = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

if __name__ == "__main__":

    tree = SumTree(4)
    rng = 5
    for _ in range(4):
        tree.add(round(np.random.random() * rng, 1), None)
    
    print(tree.tree[0])
    print(tree.tree[1], tree.tree[2])
    print(tree.tree[3], tree.tree[4], tree.tree[5], tree.tree[6])
    print()
    
    for _ in range(5):
        sample_priority = np.random.random() * (rng + 1)
        assert tree.get(sample_priority)[1] in tree.tree
        print(sample_priority, '->', tree.get(sample_priority)[1])