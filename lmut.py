import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from typing import Optional

class LinearQ(nn.Module):
	def __init__(self, state_dim: int):
		super().__init__()
		self.linear = nn.Linear(state_dim, 1)

	def forward(self, state):
		return self.linear(state)

class ActionTreeNode:
	def __init__(self, state_dim: int, buffer_size: int, depth: int = 0, device: Optional[str] = None):
		self.state_dim = state_dim
		self.depth = depth
		
		if device is None:
			device = "cuda" if torch.cuda.is_available() else "cpu"
		self.device = device
	
		self.is_leaf = True
		
		self.model = LinearQ(state_dim).to(self.device)
		self.buffer = deque(maxlen=buffer_size)
		self.optimizer: Optional[optim.Optimizer] = None
		
		self.split_feature: Optional[int] = None
		self.split_threshold: Optional[float] = None
		self.left: Optional[ActionTreeNode] = None
		self.right: Optional[ActionTreeNode] = None

	
	def route(self, state: np.ndarray):
		if self.is_leaf:
			return self
		
		if state[self.split_feature] < self.split_threshold:
			return self.left.route(state)
		
		return self.right.route(state)
	
	def predict(self, state_t):
		with torch.no_grad():
			return self.model(state_t).item()
		
	def add_sample(self, state, q):
		self.buffer.append((state, q))

	def train_step(self, lr: float, batch_size: int):
		if len(self.buffer) < batch_size:
			return float('inf')
		
		batch = random.sample(self.buffer, batch_size)

		states = torch.tensor(
			np.array([s for s, _ in batch]),
			dtype=torch.float32,
			device=self.device
		)

		targets = torch.tensor(
			np.array([q for _, q in batch]),
			dtype=torch.float32,
			device=self.device
		).unsqueeze(1)

		if self.optimizer is None:
			self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

		criterion = nn.MSELoss()

		preds = self.model(states)

		loss = criterion(preds, targets)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss.item()
	
	def variance(self):
		if len(self.buffer) <= 0:
			return 0.0
		
		return np.var([q for _, q in self.buffer])
	
	def become_internal(self, split_feature: int, split_threshold: float, buffer_size: int, lr: float, batch_size: int):
		self.left = ActionTreeNode(
			self.state_dim,
			buffer_size,
			self.depth + 1,
			device=self.device
		)

		self.right = ActionTreeNode(
			self.state_dim,
			buffer_size,
			self.depth + 1,
			device=self.device
		)

		self.left.model.load_state_dict(self.model.state_dict())
		self.right.model.load_state_dict(self.model.state_dict())

		self.split_feature = split_feature
		self.split_threshold = split_threshold

		for state, q_val in self.buffer:
			if state[split_feature] < split_threshold:
				self.left.add_sample(state, q_val)
			else:
				self.right.add_sample(state, q_val)
		
		self.is_leaf = False
		self.buffer.clear()

		self.left.train_step(lr, batch_size)
		self.right.train_step(lr, batch_size)

	def sample_count(self):
		return len(self.buffer)

class LMUT:
	def __init__(self, state_dim: int, n_actions: int, lr: float = 1e-3, q_mean: float = 0.0, q_std: float = 1.0, buffer_size: int = 3000, device: Optional[str] = None):
		self.state_dim = state_dim
		self.n_actions = n_actions
		self.lr = lr
		self.q_mean = q_mean
		self.q_std = q_std
		self.buffer_size = buffer_size

		if device is None:
			device = "cuda" if torch.cuda.is_available() else "cpu"

		self.device = device

		self.trees: list[ActionTreeNode] = [
			ActionTreeNode(state_dim, buffer_size, device=self.device)
			for _ in range(n_actions)
		]

		self.feature_influence = np.zeros(state_dim)

	def add_split_influence(self, feature: int, inf_value: float):
		self.feature_influence[feature] += inf_value

	def predict_all_actions(self, state: np.ndarray):
		state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
		
		q_vals = []
		for a in range(self.n_actions):
			leaf = self.trees[a].route(state)
			# with torch.no_grad():
				# q = leaf.model(state_t).item() * self.q_std + self.q_mean
			q_scaled = leaf.predict(state_t)
			q = q_scaled * self.q_std + self.q_mean
			q_vals.append(q)
		return q_vals

	def add_transition(self, state: np.ndarray, action: int, teacher_q: float):
		scaled_q = (teacher_q - self.q_mean) / self.q_std
		tree = self.trees[action]
		leaf = tree.route(state)

		# leaf.buffer.append((
			# state, 
			# scaled_q
		# ))
		leaf.add_sample(state, scaled_q)

	# def train_leaf(self, leaf: ActionTreeNode, batch_size: int = 64):
	# 	if len(leaf.buffer) < batch_size:
	# 		return float('inf')
		
	# 	batch = random.sample(leaf.buffer, batch_size)

	# 	states = torch.tensor(np.array([s for s, _ in batch]), dtype=torch.float32, device=self.device)
	# 	targets = torch.tensor(np.array([q for _, q in batch]), dtype=torch.float32, device=self.device).unsqueeze(1)

	# 	if leaf.optimizer is None:
	# 		leaf.optimizer = optim.Adam(leaf.model.parameters(), lr=self.lr)
	# 	criterion = nn.MSELoss()

	# 	preds = leaf.model(states)
	# 	loss = criterion(preds, targets)

	# 	leaf.optimizer.zero_grad()
	# 	loss.backward()
	# 	leaf.optimizer.step()

	# 	return loss.item()
	
	# def _compute_variance(self, leaf: ActionTreeNode):
	# 	if len(leaf.buffer) <= 0:
	# 		return 0.0
	# 	q_vals = [q for _, q in leaf.buffer]
	# 	return np.var(q_vals)

	def _try_split(self, node: ActionTreeNode, min_split: float, batch_size: int):		
		# if len(node.buffer) < batch_size:
		if node.sample_count() < batch_size:
			return False

		states = np.array([s for s, _ in node.buffer])
		q_vals = np.array([q for _, q in node.buffer])

		parent_var = np.var(q_vals)

		best_gain = -1.0
		best_feature = None
		best_thresh = None
		best_left_mask = None
		best_right_mask = None

		n_features = self.state_dim
		for feat in range(n_features):
			feat_vals = states[:, feat]

			thresholds = np.percentile(feat_vals, [25, 50, 75])

			for thresh in thresholds:
				left_mask = feat_vals < thresh
				right_mask = feat_vals >= thresh

				if left_mask.sum() <= 0 or right_mask.sum() <= 0:
					continue

				left_var = np.var(q_vals[left_mask])
				right_var = np.var(q_vals[right_mask])

				w_left = left_mask.sum() / len(q_vals)
				w_right = right_mask.sum() / len(q_vals)

				split_var = w_left * left_var + w_right * right_var
				gain = parent_var - split_var

				if gain > best_gain:
					best_gain = gain
					best_feature = feat
					best_thresh = thresh
					best_left_mask = left_mask
					best_right_mask = right_mask

		if best_gain <= min_split:
			return False
		
		# Obtener los pesos del modelo
		w = node.model.linear.weight.detach().cpu().numpy().flatten()

		sum_w_sq = np.sum(w**2) + 1e-8

		# Término izquierda
		weight_factor = 1.0 + (w[best_feature]**2) / sum_w_sq

		# Término derecha
		left_q = q_vals[best_left_mask]
		right_q = q_vals[best_right_mask]

		left_var = np.var(left_q)
		right_var = np.var(right_q)

		n_left = len(left_q)
		n_right = len(right_q)
		n_total = len(q_vals)

		weighted_child_var = (n_left * left_var + n_right * right_var) / n_total

		var_reduction = parent_var - weighted_child_var

		influence = weight_factor * var_reduction
		self.add_split_influence(best_feature, influence)

		node.become_internal(best_feature, best_thresh, self.buffer_size, self.lr, batch_size)

		# node.is_leaf = False
		# node.split_feature = best_feature
		# node.split_threshold = best_thresh

		# node.left = ActionTreeNode(node.state_dim, node.depth + 1, device=self.device)
		# node.right = ActionTreeNode(node.state_dim, node.depth + 1, device=self.device)

		# node.left.model.load_state_dict(node.model.state_dict())
		# node.right.model.load_state_dict(node.model.state_dict())

		# for state, q_val in node.buffer:
		# 	if state[best_feature] < best_thresh:
		# 		node.left.buffer.append((state, q_val))
		# 	else:
		# 		node.right.buffer.append((state, q_val))
		
		# node.buffer.clear()

		# self.train_leaf(node.left)
		# self.train_leaf(node.right)

		return True

	def update_all_leaves(self, batch_train: int = 64, min_improvement: Optional[None] = None, min_split: float = 0.2, batch_split: int = 128):
		def traverse(node: ActionTreeNode):
			if node.is_leaf:
				# loss = self.train_leaf(node, batch_train)
				loss = node.train_step(self.lr, batch_train)

				if min_improvement is None:
					split_occurred = self._try_split(node, min_split, batch_split)
				else:
					if loss <= min_improvement:
						split_occurred = self._try_split(node, min_split, batch_split)
					else:
						split_occurred = False

				if loss != float('inf'):
					return 1, loss, 1, 1 if split_occurred else 0
				else:
					return 1, 0.0, 0, 1 if split_occurred else 0
				
			else:
				left_lc, left_loss, left_trained, left_splits = traverse(node.left)
				right_lc, right_loss, right_trained, right_splits = traverse(node.right)
				return (left_lc + right_lc, 
						left_loss + right_loss, 
						left_trained + right_trained, 
						left_splits + right_splits)

		total_leaves = 0
		total_loss = 0.0
		total_trained = 0
		total_splits = 0

		for tree in self.trees:
			lc, loss_sum, trained, splits = traverse(tree)

			total_leaves += lc
			total_loss += loss_sum
			total_trained += trained
			total_splits += splits

		avg_loss = total_loss / total_trained if total_trained > 0 else 0.0
		return {'leaf_count': total_leaves, 'avg_loss': avg_loss, 'splits': total_splits}
	
	def print_tree(self, action: int, node: Optional[ActionTreeNode] = None, indent: str = ""):
		if node is None:
			print(f"\n=== Tree for action {action} ===")
			node = self.trees[action]
		
		if node.is_leaf:
			weights = node.model.linear.weight.detach().cpu().numpy().flatten()
			bias = node.model.linear.bias.detach().cpu().item()
			print(f"{indent}[Leaf] depth={node.depth} | y = {weights} * s + {bias:.4f}")
		else:
			print(f"{indent}[Node] depth={node.depth} | split: feature {node.split_feature} < {node.split_threshold:.4f}")
			print(f"{indent}  left:")
			self.print_tree(action, node.left, indent + "    ")
			print(f"{indent}  right:")
			self.print_tree(action, node.right, indent + "    ")

	def compute_current_mse(self):
		total_se = 0.0
		total_n = 0
		for tree in self.trees:
			stack = [tree]
			leaves: list[ActionTreeNode] = []
			while stack:
				node = stack.pop()
				if node.is_leaf:
					leaves.append(node)
				else:
					stack.append(node.left)
					stack.append(node.right)
			for leaf in leaves:
				for state, scaled_q in leaf.buffer:
					state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
					# with torch.no_grad():
						# pred_scaled = leaf.model(state_t).item()
					pred_scaled = leaf.predict(state_t)
					pred_orig = pred_scaled * self.q_std + self.q_mean
					target_orig = scaled_q * self.q_std + self.q_mean
					total_se += (pred_orig - target_orig) ** 2
					total_n += 1

		return total_se / total_n if total_n > 0 else 0.0
	
	def _serialize_node(self, node: ActionTreeNode):
		data = {
			'is_leaf': node.is_leaf,
			'depth': node.depth,
			'state_dim': node.state_dim,
		}
		if node.is_leaf:
			data['model_state'] = node.model.state_dict()
			# data['buffer'] = list(node.buffer)
			# data['optimizer_state'] = node.optimizer.state_dict()
		else:
			data['split_feature'] = int(node.split_feature)
			data['split_threshold'] = float(node.split_threshold)
			data['left'] = self._serialize_node(node.left)
			data['right'] = self._serialize_node(node.right)
		return data

	def _deserialize_node(self, data: dict):
		node = ActionTreeNode(
			state_dim=data['state_dim'],
			buffer_size=self.buffer_size,
			depth=data['depth'],
			device=self.device
		)
		node.is_leaf = data['is_leaf']
		if node.is_leaf:
			node.model.load_state_dict(data['model_state'])
			node.model.to(self.device)
			# for state, q in data['buffer']:
				# node.add_sample(state, q)
			# node.optimizer = optim.Adam(node.model.parameters(), lr=self.lr)
			# node.optimizer.load_state_dict(data['optimizer_state'])
		else:
			node.split_feature = data['split_feature']
			node.split_threshold = data['split_threshold']
			node.left = self._deserialize_node(data['left'])
			node.right = self._deserialize_node(data['right'])
		return node

	def save(self, path: str):
		checkpoint = {
			'config': {
				'state_dim': self.state_dim,
				'n_actions': self.n_actions,
				'lr': self.lr,
				'q_mean': self.q_mean,
				'q_std': self.q_std,
				'buffer_size': self.buffer_size
			},
			'feature_influence': self.feature_influence.tolist(),
			'trees': [self._serialize_node(tree) for tree in self.trees],
		}
		torch.save(checkpoint, path)

	@classmethod
	def load(cls, path: str, device: Optional[str] = None):
		if device is None:
			device = "cuda" if torch.cuda.is_available() else "cpu"

		checkpoint = torch.load(path, map_location=device, weights_only=False)

		cfg = checkpoint['config']
		model = cls(
			state_dim=cfg['state_dim'],
			n_actions=cfg['n_actions'],
			lr=cfg['lr'],
			q_mean=cfg['q_mean'], 
			q_std=cfg['q_std'],
			buffer_size=cfg['buffer_size'],
			device=device
		)
		model.feature_influence = np.array(checkpoint['feature_influence'])
		model.trees = [
			model._deserialize_node(tree_data)
			for tree_data in checkpoint['trees']
		]
		return model
