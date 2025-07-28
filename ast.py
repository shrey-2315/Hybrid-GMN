from sqlglot.expressions import Identifier, Literal
import sqlglot
from sqlglot import parse_one
import zss

class ASTNode(zss.Node):
    """Custom AST node class extending zss.Node for tree edit distance calculations."""
    def __init__(self, value, children=None):
        """Initializes AST node with value and optional children.
        
        Args:
            value: Node label/value
            children: List of child nodes (default None)
        """
        super().__init__(value)
        if children:
            for child in children:
                self.addkid(child)

def sqlglot_ast_to_zss(ast):
    """Converts sqlglot AST to zss-compatible tree structure.
    
    Args:
        ast: sqlglot Expression object representing parsed SQL
        
    Returns:
        ASTNode tree structure compatible with zss operations
    """
    label_parts = [ast.key]
    val_this = ast.args.get("this")
    if isinstance(val_this, (Identifier, Literal)):
        label_parts.append(str(val_this))
    label = " ".join(label_parts)
    children = []
    for key, val in ast.args.items():
        if isinstance(val, sqlglot.Expression):
            children.append(sqlglot_ast_to_zss(val))
        elif isinstance(val, list):
            children.extend(sqlglot_ast_to_zss(v) for v in val if isinstance(v, sqlglot.Expression))
    return ASTNode(label, children)

def sql_ast_similarity_weighted(sql1, sql2, weights=None):
    """Computes weighted similarity between two SQL queries using tree edit distance.
    
    Args:
        sql1: First SQL query string
        sql2: Second SQL query string
        weights: Dictionary of edit operation weights (default: {"insert":1.0, "remove":1.0, "update":2.0, "keep":0.0})
        
    Returns:
        Normalized similarity score between 0 and 1 (1 = identical)
    """
    weights = weights or {"insert": 1.0, "remove": 1.0, "update": 2.0, "keep": 0.0}
    ast1, ast2 = parse_one(sql1), parse_one(sql2)
    tree1, tree2 = sqlglot_ast_to_zss(ast1), sqlglot_ast_to_zss(ast2)
    def cost_insert(n): return weights["insert"]
    def cost_remove(n): return weights["remove"]
    def cost_update(a, b): return weights["update"] if a.label != b.label else weights["keep"]
    size = max(tree_size(tree1), tree_size(tree2))
    distance = zss.distance(tree1, tree2, get_children=lambda n: n.children,
                            insert_cost=cost_insert, remove_cost=cost_remove, update_cost=cost_update)
    return 1 - (distance / size) if size else 1.0

def tree_size(tree):
    """Calculates the total number of nodes in a tree.
    
    Args:
        tree: Root node of the tree
        
    Returns:
        Integer count of total nodes in the tree
    """
    return 1 + sum(tree_size(child) for child in tree.children) if tree else 0
