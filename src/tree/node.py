import torch


def flatten(data):
    def _flatten(data):
        for element in data:
            if isinstance(element, list):
                yield from _flatten(element)
            else:
                yield element

    return list(_flatten(data))


class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []

    def __repr__(self):
        return self.name

    def add_child(self, code):
        node = Node(name=code, parent=self)
        self.children.append(node)

    def base_counts(self, counts):
        self.base_count = counts.get(self.name, 0) + 1
        for child in self.children:
            child.base_counts(counts)

    def sum_counts(self):
        self.sum_count = self.base_count + sum(
            child.sum_counts() for child in self.children
        )
        return self.sum_count

    def redist_counts(self):
        # This is needed for root node
        self.redist_count = getattr(self, "redist_count", self.sum_count)

        for child in self.children:  # sum of children
            child.redist_count = self.redist_count * (
                child.sum_count / (self.sum_count - self.base_count)
            )
            child.redist_counts()

    def extend_leaves(self, level):
        if not self.children and level > 0:
            self.add_child(self.name)
        for child in self.children:
            child.extend_leaves(level - 1)

    def cutoff_at_level(self, cutoff_level):
        if not self.children:
            return self

        if cutoff_level <= 0:
            new_childs = [self] + [
                child.cutoff_at_level(cutoff_level - 1) for child in self.children
            ]
            self.children = []
            return new_childs
        else:
            self.children = [
                child.cutoff_at_level(cutoff_level - 1) for child in self.children
            ]
            self.children = flatten(self.children)
            for child in self.children:
                child.parent = self
            return self

    def get_tree_matrix(self):
        n_levels = self.get_max_level()
        n_leaves = len(self.get_level(n_levels))
        tree_matrix = torch.zeros((n_levels, n_leaves, n_leaves))
        for level in range(n_levels):
            nodes = self.get_level(level + 1)
            acc = 0
            for i, node in enumerate(nodes):
                n_children = node.num_children_leaves()
                tree_matrix[level, i, acc : acc + n_children] = 1
                acc += n_children
        return tree_matrix

    def create_target_mapping(self, value=-100):
        mapping = {"root": []}  # To handle root errors (for level 1)
        max_level = self.get_max_level()
        for level in range(1, max_level + 1):
            nodes = self.get_level(level)
            for i, node in enumerate(nodes):
                mapping[node.name] = (
                    mapping[node.parent.name][: level - 1]
                    + [i]
                    + [value] * (max_level - level)
                )  # parent mapping + index + padding
        del mapping["root"]  # Remove root
        return mapping

    def print_children(self, *attr, spaces=0):
        print(f" " * spaces, self, [getattr(self, a, "") for a in attr])
        for child in self.children:
            child.print_children(*attr, spaces=spaces + 2)

    def get_level(self, level):
        if self.parent is None and level > self.get_max_level():
            raise IndexError(
                f"Level {level} is too high. Max level is {self.get_max_level()}"
            )
        if level == 0:
            return [self]
        else:
            return flatten([child.get_level(level - 1) for child in self.children])

    def get_max_level(self):
        if not self.children:
            return 0
        else:
            return 1 + max(child.get_max_level() for child in self.children)

    def num_children_leaves(self):
        if not self.children:
            return 1
        return sum([child.num_children_leaves() for child in self.children])

    def get_leaf_counts(self):
        return torch.tensor(
            [c.redist_count for c in self.get_level(self.get_max_level())]
        )

    def get_all_nodes(self):
        if not self.parent:  # This removes root node and category nodes
            return flatten([child.get_all_nodes() for child in self.children])
        return [self] + flatten([child.get_all_nodes() for child in self.children])

    def create_vocabulary(self, base_vocab=None):
        if base_vocab is None:
            base_vocab = {
                "[PAD]": 0,
                "[CLS]": 1,
                "[SEP]": 2,
                "[UNK]": 3,
                "[MASK]": 4,
            }
        target_mapping = self.create_target_mapping()
        base_vocab.update(
            {k: i + len(base_vocab) for i, k in enumerate(target_mapping)}
        )
        return base_vocab
