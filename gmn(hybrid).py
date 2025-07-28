# === MAIN PIPELINE ===
function main():
    """Main execution pipeline for SQL similarity training and evaluation.
    
    Loads training and test data, parses a sample SQL query to AST,
    trains the model, evaluates performance, and plots results.
    """
    trainer = SQLSimilarityTrainer()
    train_df, test_df = trainer.load_data("train.xlsx", "test.xlsx")
    sample_ast = trainer.ast_parser.parse_sql_to_hybrid_ast(train_df[0]['gt'])
    print(sample_ast)
    trainer.train(train_df, epochs=3)
    metrics, preds, labels = trainer.evaluate(test_df)
    trainer.plot_results(preds, labels, metrics)
end

# === TRAINER CLASS ===
class SQLSimilarityTrainer:
    """Trains and evaluates a GNN model for SQL query similarity prediction.
    
    Handles the complete training pipeline including data loading,
    graph pair preparation, model training, evaluation, and visualization.
    """
    init():
        """Initializes trainer components including parser, graph builder, model, and loss."""
        self.ast_parser = SQLToHybridAST()
        self.graph_builder = HybridGraphBuilder()
        self.model = HybridGNN()
        self.optimizer = Adam(...)
        self.criterion = BCELoss()

    function load_data(train_file, test_file):
        """Loads and validates training and test data from Excel files.
        
        Args:
            train_file: Path to training data Excel file
            test_file: Path to test data Excel file
            
        Returns:
            Tuple of (train_df, test_df) containing loaded and validated dataframes
        """
        read Excel files → validate columns → return dataframes

    function prepare_graph_pairs(df):
        """Converts SQL query pairs in dataframe to graph pairs with labels.
        
        Args:
            df: DataFrame containing 'gt' and 'answer' SQL queries with labels
            
        Returns:
            List of tuples (graph1, graph2, label) ready for model training/evaluation
        """
        for each row:
            ast1 = parse_sql(gt), ast2 = parse_sql(answer)
            graph1 = ast1 → graph, graph2 = ast2 → graph
            label = row["new_labels"]
            append (graph1, graph2, label)
        return list of graph pairs

    function train(data, epochs):
        """Trains the GNN model on prepared graph pairs.
        
        Args:
            data: Training data containing SQL query pairs
            epochs: Number of training iterations over the dataset
        """
        pairs = prepare_graph_pairs(data)
        for epoch:
            for g1, g2, label in pairs:
                pred = model(g1, g2)
                loss = criterion(pred, label)
                backprop and optimizer step

    function evaluate(data):
        """Evaluates model performance on test data.
        
        Args:
            data: Test data containing SQL query pairs
            
        Returns:
            Tuple of (metrics, predictions, labels) where:
            - metrics: Dictionary of evaluation metrics
            - predictions: Model predictions for each test pair
            - labels: Ground truth labels for each test pair
        """
        pairs = prepare_graph_pairs(data)
        for g1, g2, label:
            pred = model(g1, g2)
            collect pred and label
        return metrics, preds, labels

    function plot_results(preds, labels, metrics):
        """Visualizes evaluation results with multiple plots.
        
        Args:
            preds: Model predictions
            labels: Ground truth labels
            metrics: Computed evaluation metrics
        """
        plot ROC, scatter, histogram, summary text

# === SQL TO HYBRID AST ===
class SQLToHybridAST:
    """Converts SQL queries to hybrid AST representation combining relational algebra symbols."""
    init():
        """Initializes SQL to relational algebra symbol mapping."""
        self.mapping ={
            "Select": "π",      # Projection
            "Join": "⨝",        # Join
            "InnerJoin": "⨝",   # Explicit inner join
            "LeftJoin": "⟕",    # Left outer join
            "RightJoin": "⟖",   # Right outer join
            "FullJoin": "⟗",   # Full outer join
            "Where": "σ",       # Selection (WHERE clause)
            "Group": "γ",       # GroupBy
            "Order": "τ",       # Sort
            "Limit": "λ",       # Limit
            "Union": "∪",       # Union
            "Except": "−",      # Set difference
            "Alias": "α",       # Rename
            "Table": "T",       # Table reference
            "Subquery": "S",    # Subquery
        }

    function parse_sql_to_hybrid_ast(sql):
        """Parses SQL query to hybrid AST representation.
        
        Args:
            sql: SQL query string to parse
            
        Returns:
            Dictionary representing the hybrid AST with relational algebra symbols
        """
        parse SQL using sqlglot → call _convert_node

    function _convert_node(node, parent=None):
        """Recursively converts SQL parse node to AST dictionary node.
        
        Args:
            node: SQL parse node to convert
            parent: Parent node (default None for root)
            
        Returns:
            Dictionary representing the AST node with:
            - node_id: Unique identifier
            - ast_type: Node type
            - relational_op: Corresponding relational algebra symbol
            - children: List of child nodes
            - attributes: Node-specific attributes
        """
        node_dict = {
            node_id, ast_type, relational_op,
            children = recursively parsed,
            attributes = name, alias, expressions
        }
        return node_dict

# === GRAPH BUILDER ===
class HybridGraphBuilder:
    """Converts hybrid AST representations to graph neural network inputs."""
    init():
        """Initializes graph builder with feature dimension."""
        self.feature_dim = 128

    function build_graph_from_hybrid_ast(ast):
        """Converts hybrid AST to graph representation for GNN.
        
        Args:
            ast: Hybrid AST dictionary to convert
            
        Returns:
            PyG Data object with:
            - x: Node feature matrix
            - edge_index: Graph connectivity in COO format
        """
        nodes, edges = extract from ast
        x = encode each node into 128-dim vector
        edge_index = build edge tensor
        return Data(x, edge_index)

    function _create_node_features(node):
        """Encodes AST node features into fixed-dimensional vector.
        
        Args:
            node: AST node to encode
            
        Returns:
            128-dimensional feature vector representing the node
        """
        encode ast_type, operator_type, relational_op, words

# === GNN MODEL ===
class HybridGNN(nn.Module):
    """Graph Neural Network for SQL query similarity prediction."""
    init():
        """Initializes GNN layers and similarity MLP."""
        GCN layers → conv1, conv2, conv3
        similarity_mlp = Linear → ReLU → Linear → Sigmoid

    function forward(graph1, graph2):
        """Computes similarity score between two SQL query graphs.
        
        Args:
            graph1: First query graph
            graph2: Second query graph
            
        Returns:
            Predicted similarity score between 0 and 1
        """
        emb1 = encode_graph(graph1)
        emb2 = encode_graph(graph2)
        combined = concat(emb1, emb2)
        similarity = similarity_mlp(combined)
        return similarity

    function encode_graph(graph):
        """Encodes graph into fixed-dimensional embedding.
        
        Args:
            graph: Input graph to encode
            
        Returns:
            Graph-level embedding vector
        """
        x = apply GCN layers
        pooled = global_mean_pool(x)
        return pooled

# === METRIC COMPUTATION ===
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

def compute_metrics(preds, labels):
    """Computes core metrics for SQL similarity evaluation.
    
    Args:
        preds: List/array of predicted similarity scores (0-1)
        labels: List/array of ground truth similarity scores (0-1)
    
    Returns:
        Dictionary with:
        - roc_auc: Area under ROC curve (if binary labels)
        - pearson: Pearson correlation coefficient
        - spearman: Spearman rank correlation
    """
    metrics = {
        'pearson': pearsonr(labels, preds)[0],
        'spearman': spearmanr(labels, preds)[0]
    }
    
    # Only compute ROC if labels are binary
    if len(set(labels)) == 2:
        metrics['roc_auc'] = roc_auc_score(labels, preds)
    
    return metrics
