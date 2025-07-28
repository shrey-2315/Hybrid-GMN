# === MAIN PIPELINE ===
function main():
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
    init():
        self.ast_parser = SQLToHybridAST()
        self.graph_builder = HybridGraphBuilder()
        self.model = HybridGNN()
        self.optimizer = Adam(...)
        self.criterion = BCELoss()

    function load_data(train_file, test_file):
        read Excel files → validate columns → return dataframes

    function prepare_graph_pairs(df):
        for each row:
            ast1 = parse_sql(gt), ast2 = parse_sql(answer)
            graph1 = ast1 → graph, graph2 = ast2 → graph
            label = row["new_labels"]
            append (graph1, graph2, label)
        return list of graph pairs

    function train(data, epochs):
        pairs = prepare_graph_pairs(data)
        for epoch:
            for g1, g2, label in pairs:
                pred = model(g1, g2)
                loss = criterion(pred, label)
                backprop and optimizer step

    function evaluate(data):
        pairs = prepare_graph_pairs(data)
        for g1, g2, label:
            pred = model(g1, g2)
            collect pred and label
        return metrics, preds, labels

    function plot_results(preds, labels, metrics):
        plot ROC, scatter, histogram, summary text

# === SQL TO HYBRID AST ===
class SQLToHybridAST:
    init():
        mapping = {"Select": "π", "Join": "⨝", ...}

    function parse_sql_to_hybrid_ast(sql):
        parse SQL using sqlglot → call _convert_node

    function _convert_node(node, parent=None):
        node_dict = {
            node_id, ast_type, relational_op,
            children = recursively parsed,
            attributes = name, alias, expressions
        }
        return node_dict

# === GRAPH BUILDER ===
class HybridGraphBuilder:
    init():
        feature_dim = 128

    function build_graph_from_hybrid_ast(ast):
        nodes, edges = extract from ast
        x = encode each node into 128-dim vector
        edge_index = build edge tensor
        return Data(x, edge_index)

    function _create_node_features(node):
        encode ast_type, operator_type, relational_op, words

# === GNN MODEL ===
class HybridGNN(nn.Module):
    init():
        GCN layers → conv1, conv2, conv3
        similarity_mlp = Linear → ReLU → Linear → Sigmoid

    function forward(graph1, graph2):
        emb1 = encode_graph(graph1)
        emb2 = encode_graph(graph2)
        combined = concat(emb1, emb2)
        similarity = similarity_mlp(combined)
        return similarity

    function encode_graph(graph):
        x = apply GCN layers
        pooled = global_mean_pool(x)
        return pooled

# === METRIC COMPUTATION ===
function compute_metrics(preds, labels):
    roc_auc = compute if binary
    pearson = correlation(preds, labels)
    spearman = correlation(preds, labels)
    return dict of metrics
