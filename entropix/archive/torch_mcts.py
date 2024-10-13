# import torch
# from typing import Tuple, List
# import os
# import math

# from entropix.torch_sampler import _sample, calculate_metrics, set_mcts_callback

# class MCTSNode:
#     def __init__(self, token, logits, attention_scores, parent=None):
#         self.token = token
#         self.logits = logits
#         self.attention_scores = attention_scores
#         self.parent = parent
#         self.children = []
#         self.visits = 0
#         self.value = 0
#         self.metrics = None

# class MCTSSearch:
#     def __init__(self, cxfmr, xfmr_weights, model_params, freqs_cis, kvcache):
#         self.cxfmr = cxfmr
#         self.xfmr_weights = xfmr_weights
#         self.model_params = model_params
#         self.freqs_cis = freqs_cis
#         self.kvcache = kvcache
#         self.max_depth = 10
#         self.n_simulations = 50
#         self.exploration_constant = 1.4
#         self.log_file = "mcts_log.txt"
#         self._clear_log_file()
        
#         set_mcts_callback(self.search)

#     def _clear_log_file(self):
#         if os.path.exists(self.log_file):
#             os.remove(self.log_file)

#     def _log_node(self, depth: int, node: MCTSNode):
#         with open(self.log_file, "a") as f:
#             indent = "  " * depth
#             f.write(f"{indent}├─ Token: {node.token.item()}, Visits: {node.visits}, Value: {node.value:.4f}\n")

#     def _evaluate_node(self, node: MCTSNode) -> float:
#         if node.metrics is None:
#             node.metrics = calculate_metrics(node.logits, node.attention_scores)
        
#         ent = node.metrics["logits_entropy"].item()
#         vent = node.metrics["logits_varentropy"].item()
#         agreement = node.metrics["agreement"].item()
#         interaction_strength = node.metrics["interaction_strength"].item()
        
#         # Normalize metrics to [0, 1] range
#         norm_ent = 1 - min(ent / 5.0, 1)
#         norm_vent = 1 - min(vent / 5.0, 1)
        
#         # Combine metrics into a single score
#         score = (
#             norm_ent * 0.3 +
#             norm_vent * 0.3 +
#             agreement * 0.2 +
#             interaction_strength * 0.2
#         )
#         return score

#     def _select_child(self, node: MCTSNode) -> MCTSNode:
#         best_score = float('-inf')
#         best_child = None
        
#         for child in node.children:
#             if child.visits == 0:
#                 return child
            
#             exploit = child.value / child.visits
#             explore = math.sqrt(math.log(node.visits) / child.visits)
#             uct_score = exploit + self.exploration_constant * explore
            
#             if uct_score > best_score:
#                 best_score = uct_score
#                 best_child = child
        
#         return best_child

#     def _expand(self, node: MCTSNode, cur_pos: int) -> MCTSNode:
#         next_token = _sample(node.logits, temperature=1.0, top_p=0.9, top_k=20)
#         next_logits, kvcache, next_attention_scores, _ = self.cxfmr(
#             self.xfmr_weights, self.model_params, next_token,
#             cur_pos + 1, self.freqs_cis[cur_pos + 1:cur_pos + 2], self.kvcache
#         )
#         child = MCTSNode(next_token.squeeze(), next_logits, next_attention_scores, parent=node)
#         node.children.append(child)
#         return child

#     def _simulate(self, node: MCTSNode, cur_pos: int, depth: int) -> float:
#         if depth >= self.max_depth:
#             return self._evaluate_node(node)
        
#         if not node.children:
#             return self._evaluate_node(node)
        
#         selected_child = self._select_child(node)
#         if selected_child.visits == 0:
#             return self._evaluate_node(selected_child)
        
#         value = self._simulate(selected_child, cur_pos + 1, depth + 1)
#         selected_child.visits += 1
#         selected_child.value += value
#         return value

#     def _backpropagate(self, node: MCTSNode, value: float):
#         while node is not None:
#             node.visits += 1
#             node.value += value
#             node = node.parent

#     def search(self, logits: torch.Tensor, attention_scores: torch.Tensor, gen_tokens: torch.Tensor, cur_pos: int) -> torch.Tensor:
#         root = MCTSNode(gen_tokens[:, -1], logits, attention_scores)
        
#         for _ in range(self.n_simulations):
#             leaf = root
#             depth = 0
            
#             # Selection
#             while leaf.children:
#                 leaf = self._select_child(leaf)
#                 depth += 1
            
#             # Expansion
#             if leaf.visits > 0:
#                 leaf = self._expand(leaf, cur_pos + depth)
            
#             # Simulation
#             value = self._simulate(leaf, cur_pos + depth, depth)
            
#             # Backpropagation
#             self._backpropagate(leaf, value)
        
#         best_child = max(root.children, key=lambda c: c.visits)
#         self._log_search_result(root)
#         return best_child.token.unsqueeze(0).unsqueeze(0)

#     def _log_search_result(self, root: MCTSNode):
#         with open(self.log_file, "a") as f:
#             f.write("MCTS Search Result:\n")
#             for child in root.children:
#                 f.write(f"Token: {child.token.item()}, Visits: {child.visits}, Value: {child.value:.4f}\n")

# def initialize_mcts(cxfmr, xfmr_weights, model_params, freqs_cis, kvcache):
#     return MCTSSearch(cxfmr, xfmr_weights, model_params, freqs_cis, kvcache)