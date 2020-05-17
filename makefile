full_adj_mx:
	python3 gen_adj_mx.py --sensor_ids_filename data/sensor_graph/full_ids.txt --distances_filename data/sensor_graph/full_distances.csv --output_pkl_filename data/sensor_graph/full_adj_mx.pkl

part_adj_mx:
	python3 gen_adj_mx.py --sensor_ids_filename data/sensor_graph/part_ids.txt --distances_filename data/sensor_graph/part_distances.csv --output_pkl_filename data/sensor_graph/part_adj_mx.pkl

fab_full_data:
	python3 generate_training_data.py --output_dir data/fab_full --traffic_df_filename data/fab_full.h5 --edge_weights_filename data/fab/full_graph{}.csv

fab_part_data:
	python3 generate_training_data.py --output_dir data/fab_part --traffic_df_filename data/fab_part.h5 --edge_weights_filename data/fab/part_graph{}.csv

toy_full_data:
	python3 generate_training_data.py --output_dir data/toy_full --traffic_df_filename data/toy_full.h5 --edge_weights_filename data/toy/full_graph{}.csv --subdatasets 4

toy_part_data:
	python3 generate_training_data.py --output_dir data/toy_part --traffic_df_filename data/toy_part.h5 --edge_weights_filename data/toy/part_graph{}.csv --subdatasets 4

fab_pwn_train:
	python3 train.py --data data/fab_full --adjdata data/sensor_graph/full_adj_mx.pkl --do_graph_conv --addaptadj --randomadj --num_nodes 168 --cat_feat_gc --scale_dim 20 --upscale_output --pwn --save logs/fab_pwn

fab_gwn_train:
	python3 train.py --data data/fab_full --adjdata data/sensor_graph/full_adj_mx.pkl --do_graph_conv --addaptadj --randomadj --num_nodes 168 --cat_feat_gc --save logs/fab_gwn

fab_embedding_train:
	python3 train.py --data data/fab_full --adjdata data/sensor_graph/full_adj_mx.pkl --do_graph_conv --aptonly --addaptadj --randomadj --num_nodes 20 --cat_feat_gc --scale_dim 168 --downscale_input --upscale_output --save logs/fab_embedding

fab_partition_train:
	python3 train.py --data data/fab_part --adjdata data/sensor_graph/part_adj_mx.pkl --do_graph_conv --addaptadj --randomadj --num_nodes 20 --cat_feat_gc --save logs/fab_partition

toy_pwn_train:
	python3 train.py --device cpu --data data/toy_full --adjdata data/sensor_graph/full_adj_mx.pkl --do_graph_conv --addaptadj --randomadj --num_nodes 168 --cat_feat_gc --scale_dim 20 --upscale_output --pwn --save logs/toy_pwn

toy_gwn_train:
	python3 train.py --device cpu --data data/toy_full --adjdata data/sensor_graph/full_adj_mx.pkl --do_graph_conv --addaptadj --randomadj --num_nodes 168 --cat_feat_gc --save logs/toy_gwn

toy_embedding_train:
	python3 train.py --device cpu --data data/toy_full --adjdata data/sensor_graph/full_adj_mx.pkl --do_graph_conv --aptonly --addaptadj --randomadj --num_nodes 20 --cat_feat_gc --scale_dim 168 --downscale_input --upscale_output --save logs/toy_embedding

toy_partition_train:
	python3 train.py --device cpu --data data/toy_part --adjdata data/sensor_graph/part_adj_mx.pkl --do_graph_conv --addaptadj --randomadj --num_nodes 20 --cat_feat_gc --save logs/toy_partition

